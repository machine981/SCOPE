import argparse
import json
import os
import sys
import time
import itertools
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from openai import OpenAI
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from verl.utils.reward_score import _default_compute_score


def teacher_generate_on_wrong_samples(
    df: pd.DataFrame,
    teacher_ip_pool: list,
    teacher_model_name: str,
    n_samples: int,
    temperature: float = 0.6,
    max_tokens: int = 32768,
):
    # Create API client pool
    client_pool = [
        OpenAI(base_url=f"http://{ip}:8000/v1", api_key="xxx")
        for ip in teacher_ip_pool
    ]
    client_iter = itertools.cycle(client_pool)
    n_workers = len(teacher_ip_pool) * 10

    def generate_single(prompt_messages, max_retries=3):
        for attempt in range(max_retries):
            try:
                client = next(client_iter)
                resp = client.chat.completions.create(
                    model=teacher_model_name,
                    messages=prompt_messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return resp.choices[0].message.content
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"  [WARN] Generation failed: {e}")
                    return ""
                time.sleep(1)
        return ""

    # Generate n_samples for each prompt
    n = len(df)
    responses = [[None] * n_samples for _ in range(n)]
    total_tasks = n * n_samples
    print(f"Teacher generation: {n} prompts x {n_samples} = {total_tasks} calls")

    prompts = df["prompt"].tolist()
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        future_to_idx = {}
        for i, prompt in enumerate(prompts):
            prompt_messages = prompt.tolist() if hasattr(prompt, "tolist") else prompt
            for k in range(n_samples):
                fut = executor.submit(generate_single, prompt_messages)
                future_to_idx[fut] = (i, k)

        for future in tqdm(as_completed(future_to_idx), total=total_tasks, desc="Generation"):
            i, k = future_to_idx[future]
            try:
                responses[i][k] = future.result()
            except Exception:
                responses[i][k] = ""

    # Evaluate accuracy
    all_scores = []
    for i, (_, row_data) in tqdm(enumerate(df.iterrows()), total=n, desc="Evaluation"):
        gt = row_data["reward_model"]["ground_truth"]
        ds = row_data["data_source"]
        row_scores = []
        for k in range(n_samples):
            resp = responses[i][k] or ""
            try:
                score = float(_default_compute_score(ds, resp, gt))
            except Exception:
                score = 0.0
            row_scores.append(score)
        all_scores.append(row_scores)

    return responses, all_scores


def run_exp1(
    wrong_samples_path: str,
    teacher_ip_pool: list,
    teacher_model_name: str,
    n_samples: int,
    output_dir: str,
    temperature: float = 0.6,
    max_tokens: int = 32768,
):
    print(f"\n{'='*60}")
    print("Exp1: Teacher avg@n on wrong samples")
    print(f"{'='*60}")

    df = pd.read_parquet(wrong_samples_path)
    print(f"Loaded: {len(df)} samples")

    has_ppl_bin = "ppl_bin" in df.columns
    if has_ppl_bin:
        print(f"PPL bins: {df['ppl_bin'].nunique()}")
        for b in sorted(df['ppl_bin'].unique()):
            print(f"  Bin {b}: {(df['ppl_bin']==b).sum()} samples, avg PPL={df[df['ppl_bin']==b]['teacher_ppl'].mean():.2f}")

    # Teacher generation
    responses, all_scores = teacher_generate_on_wrong_samples(
        df=df,
        teacher_ip_pool=teacher_ip_pool,
        teacher_model_name=teacher_model_name,
        n_samples=n_samples,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    # Compute metrics
    avg_scores = [np.mean(s) for s in all_scores]
    pass_at_1 = [s[0] for s in all_scores]
    pass_at_n = [float(max(s) > 0) for s in all_scores]

    df = df.copy()
    df["teacher_responses"] = [json.dumps(r) for r in responses]
    df["teacher_scores"] = [json.dumps(s) for s in all_scores]
    df["teacher_avg_score"] = avg_scores
    df["teacher_pass_at_1"] = pass_at_1
    df["teacher_pass_at_n"] = pass_at_n

    print(f"\nOverall: pass@1={np.mean(pass_at_1):.4f}, avg@{n_samples}={np.mean(avg_scores):.4f}")

    # Stats by PPL bin
    bin_results = {}
    if has_ppl_bin:
        bin_label_map = {
            0: "Q1_lowest_ppl", 1: "Q2_low_ppl",
            2: "Q3_high_ppl",   3: "Q4_highest_ppl",
        } if df['ppl_bin'].nunique() == 4 else {
            0: "A_low_ppl", 1: "B_high_ppl"
        }

        print(f"\nBy PPL bin:")
        for bin_id, label in sorted(bin_label_map.items()):
            mask = df["ppl_bin"] == bin_id
            bin_df = df[mask]
            if len(bin_df) == 0:
                continue
            b_avg = bin_df["teacher_avg_score"].mean()
            b_p1 = bin_df["teacher_pass_at_1"].mean()
            b_pn = bin_df["teacher_pass_at_n"].mean()
            b_ppl = bin_df["teacher_ppl"].mean()
            print(f"  {label}: n={len(bin_df)}, PPL={b_ppl:.2f}, pass@1={b_p1:.4f}, avg@{n_samples}={b_avg:.4f}")
            bin_results[label] = {
                "bin_id": bin_id,
                "size": len(bin_df),
                "teacher_ppl_mean": float(b_ppl),
                "pass_at_1": float(b_p1),
                f"avg_at_{n_samples}": float(b_avg),
                f"pass_at_{n_samples}": float(b_pn),
            }

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    result_path = os.path.join(output_dir, f"teacher_avg{n_samples}_on_wrong_samples.parquet")
    df.to_parquet(result_path, index=False)
    print(f"\nResults saved to: {result_path}")

    stats = {
        "total_samples": len(df),
        "n_samples": n_samples,
        "teacher_model": teacher_model_name,
        "overall": {
            "pass_at_1": float(np.mean(pass_at_1)),
            f"avg_at_{n_samples}": float(np.mean(avg_scores)),
            f"pass_at_{n_samples}": float(np.mean(pass_at_n)),
        },
        "by_ppl_bin": bin_results,
    }
    summary_path = os.path.join(output_dir, f"teacher_avg{n_samples}_on_wrong_samples_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"Summary saved to: {summary_path}")

    return df, stats


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wrong_samples_path", type=str,
                        default="exp2_results/sample2000_rollout1_bins4/wrong_samples_with_ppl.parquet")
    parser.add_argument("--teacher_ip_pool", type=str, default="")
    parser.add_argument("--teacher_model_name", type=str, default="Skywork-OR1-7B")
    parser.add_argument("--n_samples", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--max_tokens", type=int, default=32768)
    parser.add_argument("--output_dir", type=str, default="./exp1_results")
    return parser.parse_args()


def main():
    args = parse_args()
    teacher_ip_pool = [ip.strip() for ip in args.teacher_ip_pool.split(",") if ip.strip()]
    print(f"Teacher IP pool: {teacher_ip_pool}")

    run_exp1(
        wrong_samples_path=args.wrong_samples_path,
        teacher_ip_pool=teacher_ip_pool,
        teacher_model_name=args.teacher_model_name,
        n_samples=args.n_samples,
        output_dir=args.output_dir,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )


if __name__ == "__main__":
    main()
