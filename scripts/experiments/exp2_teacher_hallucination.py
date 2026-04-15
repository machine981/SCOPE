import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from openai import OpenAI
from tqdm import tqdm
import itertools
from concurrent.futures import ThreadPoolExecutor, as_completed

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from verl.utils.api_interface import vllmAPIModelInterface
from verl.utils.reward_score import _default_compute_score


def prepare_data(data_path: str, sample_size: int, output_dir: str, seed: int = 42):
    df = pd.read_parquet(data_path)
    print(f"Total samples: {len(df)}")

    if sample_size < len(df):
        df = df.sample(n=sample_size, random_state=seed).reset_index(drop=True)

    os.makedirs(output_dir, exist_ok=True)
    sample_path = os.path.join(output_dir, "train_sample.parquet")
    df.to_parquet(sample_path, index=False)
    print(f"Saved: {sample_path}")
    return sample_path


def label_student_responses(student_gen_path: str, output_dir: str):
    df = pd.read_parquet(student_gen_path)
    print(f"Loaded: {len(df)} prompts")

    # Expand trajectories to individual rows
    def extract_all_responses(val):
        if isinstance(val, (list, np.ndarray)):
            val = list(val)
            result = []
            for item in val:
                if isinstance(item, (list, np.ndarray)):
                    result.extend([str(x) for x in item if x])
                elif isinstance(item, str) and item:
                    result.append(item)
            return result if result else [""]
        return [""]

    rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Expand trajectories"):
        responses = extract_all_responses(row["responses"])
        for resp in responses:
            new_row = row.to_dict()
            new_row["student_response"] = resp
            rows.append(new_row)

    expanded_df = pd.DataFrame(rows).reset_index(drop=True)
    n_prompts = len(df)
    n_total = len(expanded_df)
    n_samples = n_total // n_prompts
    print(f"Expanded: {n_prompts} x {n_samples} = {n_total}")

    # Label with accuracy score
    scores = []
    for _, row in tqdm(expanded_df.iterrows(), total=len(expanded_df), desc="Label"):
        try:
            gt = row["reward_model"]["ground_truth"]
            score = _default_compute_score(row["data_source"], row["student_response"], gt)
            scores.append(float(score))
        except Exception:
            scores.append(0.0)

    expanded_df["student_score"] = scores
    print(f"Accuracy: {np.mean(scores):.4f}")

    labeled_path = os.path.join(output_dir, "student_gen_labeled.parquet")
    expanded_df.to_parquet(labeled_path, index=False)

    # Filter wrong samples (score = 0)
    wrong_df = expanded_df[expanded_df["student_score"] == 0.0].copy().reset_index(drop=True)
    print(f"Wrong samples: {len(wrong_df)}")

    return expanded_df, wrong_df


def truncate_to_prefix(response: str, truncate_ratio: float = 0.7) -> str:
    if not response or len(response.strip()) == 0:
        return response

    target_len = int(len(response) * truncate_ratio)

    # Find nearest newline before target_len
    newline_pos = response.rfind('\n', 0, target_len)
    if newline_pos > 0:
        return response[:newline_pos].rstrip()

    # Fallback: find nearest newline after target_len
    newline_pos = response.find('\n', target_len)
    if newline_pos > 0:
        return response[:newline_pos].rstrip()

    return response[:target_len].rstrip()


def build_prefix_inputs(wrong_df: pd.DataFrame, tokenizer, truncate_ratio: float = 0.7):
    prefix_token_ids_list = []
    prompt_lengths = []
    prefixes = []

    for _, row in tqdm(wrong_df.iterrows(), total=len(wrong_df), desc="Build prefixes"):
        prompt_messages = row["prompt"]
        if hasattr(prompt_messages, "tolist"):
            prompt_messages = prompt_messages.tolist()

        prompt_ids = tokenizer.apply_chat_template(
            prompt_messages,
            add_generation_prompt=True,
            tokenize=True,
        )

        prefix_text = truncate_to_prefix(row["student_response"], truncate_ratio)
        prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=False)

        prefix_token_ids_list.append(prompt_ids + prefix_ids)
        prompt_lengths.append(len(prompt_ids))
        prefixes.append(prefix_text)

    return prefix_token_ids_list, prompt_lengths, prefixes


def compute_teacher_ppl(
    wrong_df: pd.DataFrame,
    teacher_ip_pool: list,
    teacher_model_name: str,
    student_tokenizer_path: str,
    ppl_dir: str,
    teacher_max_model_len: int = 34000,
):
    if len(wrong_df) == 0:
        print("[WARN] No wrong samples")
        return wrong_df

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(student_tokenizer_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Build full sequences (prompt + response)
    full_token_ids_list = []
    full_prompt_lengths = []
    valid_indices = []

    for idx, (_, row) in enumerate(tqdm(wrong_df.iterrows(), total=len(wrong_df), desc="Tokenize")):
        prompt_messages = row["prompt"]
        if hasattr(prompt_messages, "tolist"):
            prompt_messages = prompt_messages.tolist()
        prompt_ids = tokenizer.apply_chat_template(
            prompt_messages, add_generation_prompt=True, tokenize=True,
        )
        response_ids = tokenizer.encode(row["student_response"], add_special_tokens=False)
        total_len = len(prompt_ids) + len(response_ids)
        if total_len > teacher_max_model_len:
            continue
        full_token_ids_list.append(prompt_ids + response_ids)
        full_prompt_lengths.append(len(prompt_ids))
        valid_indices.append(idx)

    n_filtered = len(wrong_df) - len(valid_indices)
    if n_filtered > 0:
        print(f"[INFO] Filtered {n_filtered} too-long samples")
    wrong_df = wrong_df.iloc[valid_indices].reset_index(drop=True)

    # Compute PPL via API
    client = vllmAPIModelInterface(
        model_name=teacher_model_name,
        ip_pool=teacher_ip_pool,
        top_k=1,
    )
    n_workers = len(teacher_ip_pool) * 10
    print(f"Computing PPL for {len(full_token_ids_list)} sequences...")
    full_results = client.get_batch_answers(full_token_ids_list, max_workers=n_workers)

    # Extract PPL from logprobs
    nll_list = []
    ppl_list = []
    for i, (res, prompt_len) in enumerate(zip(full_results, full_prompt_lengths)):
        all_logprobs = res.get("prompt_logprobs", [])
        response_len = len(full_token_ids_list[i]) - prompt_len
        if len(all_logprobs) == 0 or response_len <= 0:
            nll_list.append(float("inf"))
            ppl_list.append(float("inf"))
            continue
        response_logprobs = all_logprobs[-response_len:]
        valid_lps = [lp for lp in response_logprobs
                     if lp is not None and not np.isinf(lp) and not np.isnan(lp)]
        if len(valid_lps) == 0:
            nll_list.append(float("inf"))
            ppl_list.append(float("inf"))
            continue
        nll = -np.mean(valid_lps)
        nll_list.append(nll)
        ppl_list.append(float(np.exp(nll)))

    wrong_df = wrong_df.copy()
    wrong_df["teacher_nll"] = nll_list
    wrong_df["teacher_ppl"] = ppl_list

    valid_mask = np.isfinite(wrong_df["teacher_ppl"])
    wrong_df = wrong_df[valid_mask].reset_index(drop=True)

    print(f"PPL stats: mean={wrong_df['teacher_ppl'].mean():.2f}, median={wrong_df['teacher_ppl'].median():.2f}")

    os.makedirs(ppl_dir, exist_ok=True)
    ppl_path = os.path.join(ppl_dir, "wrong_samples_with_ppl.parquet")
    wrong_df.to_parquet(ppl_path, index=False)
    print(f"Saved: {ppl_path}")

    return wrong_df


def filter_truncated_responses(
    wrong_df: pd.DataFrame,
    student_tokenizer_path: str,
    max_response_length: int = 32768,
):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(student_tokenizer_path, trust_remote_code=True)

    n_before = len(wrong_df)
    truncated_mask = []
    for _, row in tqdm(wrong_df.iterrows(), total=len(wrong_df), desc="Check length"):
        response_ids = tokenizer.encode(row["student_response"], add_special_tokens=False)
        truncated_mask.append(len(response_ids) >= max_response_length)

    truncated_mask = pd.Series(truncated_mask, index=wrong_df.index)
    wrong_df = wrong_df[~truncated_mask].reset_index(drop=True)
    n_filtered = n_before - len(wrong_df)
    print(f"Filtered {n_filtered} truncated samples, remaining {len(wrong_df)}")
    return wrong_df


def split_and_build_prefix(
    wrong_df: pd.DataFrame,
    student_tokenizer_path: str,
    output_dir: str,
    n_bins: int = 4,
    split_method: str = "quartile",
    truncate_ratio: float = 0.7,
):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(student_tokenizer_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Bin by PPL
    ppl = wrong_df["teacher_ppl"]
    bin_labels = []

    if split_method == "median" or n_bins == 2:
        threshold = ppl.median()
        wrong_df = wrong_df.copy()
        wrong_df["ppl_bin"] = (ppl > threshold).astype(int)
        bin_labels = ["A_low_ppl", "B_high_ppl"]
        print(f"Median split (PPL={threshold:.2f}): A={sum(wrong_df['ppl_bin']==0)}, B={sum(wrong_df['ppl_bin']==1)}")

    elif split_method == "quartile" or n_bins == 4:
        quantiles = [ppl.quantile(q) for q in [0.25, 0.5, 0.75]]
        def assign_bin(v):
            if v <= quantiles[0]: return 0
            elif v <= quantiles[1]: return 1
            elif v <= quantiles[2]: return 2
            else: return 3
        wrong_df = wrong_df.copy()
        wrong_df["ppl_bin"] = ppl.apply(assign_bin)
        bin_labels = ["Q1_lowest_ppl", "Q2_low_ppl", "Q3_high_ppl", "Q4_highest_ppl"]
        for b, label in enumerate(bin_labels):
            n = (wrong_df["ppl_bin"] == b).sum()
            mean_ppl = wrong_df[wrong_df["ppl_bin"] == b]["teacher_ppl"].mean()
            print(f"  {label}: n={n}, PPL={mean_ppl:.2f}")

    # Build truncated prefixes
    _, _, prefixes = build_prefix_inputs(wrong_df, tokenizer, truncate_ratio)
    wrong_df["prefix_response"] = prefixes

    prefix_ratios = [len(p)/max(len(r), 1)
                     for p, r in zip(prefixes, wrong_df["student_response"])]
    print(f"Avg prefix ratio: {np.mean(prefix_ratios):.2%}")

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "wrong_samples_with_ppl_prefix.parquet")
    wrong_df.to_parquet(output_path, index=False)
    print(f"Saved: {output_path}")

    return wrong_df, bin_labels


def teacher_continuation(
    wrong_df: pd.DataFrame,
    bin_labels: list,
    teacher_ip_pool: list,
    teacher_model_name: str,
    student_tokenizer_path: str,
    output_dir: str,
    n_continuations: int = 4,
    temperature: float = 0.7,
    max_tokens: int = 2048,
    teacher_max_model_len: int = 34000,
):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(student_tokenizer_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # API client pool
    client_pool = [
        OpenAI(base_url=f"http://{ip}:8000/v1", api_key="xxx")
        for ip in teacher_ip_pool
    ]
    client_iter = itertools.cycle(client_pool)

    def teacher_continue_single(prefix_token_ids: list, dynamic_max_tokens: int, max_retries: int = 3) -> str:
        for attempt in range(max_retries):
            try:
                client = next(client_iter)
                resp = client.completions.create(
                    model=teacher_model_name,
                    prompt=prefix_token_ids,
                    temperature=temperature,
                    max_tokens=dynamic_max_tokens,
                    stop=None,
                )
                return resp.choices[0].text
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"  [WARN] Continuation failed: {e}")
                    return ""
                time.sleep(1)
        return ""

    # Build prefix token IDs with dynamic max_tokens
    prefix_token_ids_list = []
    dynamic_max_tokens_list = []
    for _, row in tqdm(wrong_df.iterrows(), total=len(wrong_df), desc="Tokenize prefix"):
        prompt_messages = row["prompt"]
        if hasattr(prompt_messages, "tolist"):
            prompt_messages = prompt_messages.tolist()
        prompt_ids = tokenizer.apply_chat_template(
            prompt_messages, add_generation_prompt=True, tokenize=True,
        )
        prefix_ids = tokenizer.encode(row["prefix_response"], add_special_tokens=False)
        prefix_token_ids = prompt_ids + prefix_ids
        prefix_token_ids_list.append(prefix_token_ids)
        remaining = teacher_max_model_len - len(prefix_token_ids) - 100
        dynamic_max_tokens_list.append(max(256, min(max_tokens, remaining)))

    # Generate n_continuations for each prefix
    n = len(wrong_df)
    continuations = [[None] * n_continuations for _ in range(n)]

    total_tasks = n * n_continuations
    print(f"Continuation: {n} x {n_continuations} = {total_tasks} calls")

    n_workers = len(teacher_ip_pool) * 10
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        future_to_idx = {}
        for i, (prefix_ids, dyn_max_tok) in enumerate(zip(prefix_token_ids_list, dynamic_max_tokens_list)):
            for k in range(n_continuations):
                fut = executor.submit(teacher_continue_single, prefix_ids, dyn_max_tok)
                future_to_idx[fut] = (i, k)

        for future in tqdm(as_completed(future_to_idx), total=total_tasks, desc="Continuation"):
            i, k = future_to_idx[future]
            try:
                continuations[i][k] = future.result()
            except Exception:
                continuations[i][k] = ""

    # Evaluate recovery rate (accuracy after continuation)
    recovery_scores = []
    for i, row in tqdm(enumerate(wrong_df.iterrows()), total=n, desc="Evaluate"):
        _, row_data = row
        gt = row_data["reward_model"]["ground_truth"]
        ds = row_data["data_source"]
        prefix = row_data["prefix_response"]

        row_scores = []
        for k in range(n_continuations):
            cont = continuations[i][k] or ""
            full_response = prefix + cont
            try:
                score = float(_default_compute_score(ds, full_response, gt))
            except Exception:
                score = 0.0
            row_scores.append(score)
        recovery_scores.append(row_scores)

    wrong_df = wrong_df.copy()
    wrong_df["teacher_continuations"] = [json.dumps(c) for c in continuations]
    wrong_df["recovery_scores"] = [json.dumps(s) for s in recovery_scores]
    wrong_df["mean_recovery_rate"] = [np.mean(s) for s in recovery_scores]

    # Stats by PPL bin
    print(f"\nRecovery rate by PPL bin:")
    bin_results = {}
    for bin_id, label in enumerate(bin_labels):
        mask = wrong_df["ppl_bin"] == bin_id
        bin_df = wrong_df[mask]
        if len(bin_df) == 0:
            continue
        mean_ppl = bin_df["teacher_ppl"].mean()
        recovery_rate = bin_df["mean_recovery_rate"].mean()
        bin_results[label] = {
            "bin_id": bin_id,
            "size": len(bin_df),
            "teacher_ppl_mean": float(mean_ppl),
            "error_recovery_rate": float(recovery_rate),
        }
        print(f"  {label}: n={len(bin_df)}, PPL={mean_ppl:.2f}, recovery={recovery_rate:.4f}")

    output_path = os.path.join(output_dir, "teacher_continuation_results.parquet")
    wrong_df.to_parquet(output_path, index=False)
    print(f"Saved: {output_path}")

    return wrong_df, bin_results


def summarize_results(
    df: pd.DataFrame,
    wrong_df: pd.DataFrame,
    bin_results: dict,
    output_dir: str,
):
    bin_list = sorted(bin_results.values(), key=lambda x: x["bin_id"])

    # Check if recovery rate is monotonically increasing with PPL
    recovery_rates = [b["error_recovery_rate"] for b in bin_list]
    is_monotone = all(recovery_rates[i] <= recovery_rates[i+1]
                      for i in range(len(recovery_rates)-1))

    stats = {
        "total_samples": len(df),
        "wrong_samples": len(wrong_df),
        "student_accuracy": float(df["student_score"].mean()),
        "bins": bin_results,
        "key_finding": {
            "recovery_rates_by_ppl_bin": recovery_rates,
            "monotone_increasing": is_monotone,
            "rate_gap_highest_minus_lowest": (
                recovery_rates[-1] - recovery_rates[0] if len(recovery_rates) >= 2 else 0.0
            ),
            "hypothesis_verified": is_monotone and recovery_rates[-1] > recovery_rates[0],
        },
    }

    summary_path = os.path.join(output_dir, "summary_stats.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"\nRecovery rate gap: {stats['key_finding']['rate_gap_highest_minus_lowest']:+.4f}")
    print(f"Monotone: {'Yes' if is_monotone else 'No'}")
    print(f"Saved: {summary_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=str, required=True, choices=["prepare_data", "analyze"])
    parser.add_argument("--data_path", type=str, default="data/deepmath_new/deepmath_new_train.parquet")
    parser.add_argument("--sample_size", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--student_gen_path", type=str, default="")
    parser.add_argument("--student_model_path", type=str, default="./Models/DeepSeek-R1-Distill-Qwen-1.5B")
    parser.add_argument("--teacher_ip_pool", type=str, default="")
    parser.add_argument("--teacher_model_name", type=str, default="Skywork-OR1-7B")
    parser.add_argument("--n_bins", type=int, default=4)
    parser.add_argument("--split_method", type=str, default="quartile", choices=["median", "quartile"])
    parser.add_argument("--teacher_max_model_len", type=int, default=34000)
    parser.add_argument("--truncate_ratio", type=float, default=0.7)
    parser.add_argument("--n_continuations", type=int, default=4)
    parser.add_argument("--teacher_temperature", type=float, default=0.7)
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--output_dir", type=str, default="./exp2_results")
    parser.add_argument("--ppl_dir", type=str, default="")
    parser.add_argument("--skip_label", action="store_true")
    parser.add_argument("--skip_ppl", action="store_true")
    parser.add_argument("--skip_prefix", action="store_true")
    parser.add_argument("--filter_truncated", action="store_true")
    parser.add_argument("--max_response_length", type=int, default=32768)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.step == "prepare_data":
        prepare_data(
            data_path=args.data_path,
            sample_size=args.sample_size,
            output_dir=args.output_dir,
            seed=args.seed,
        )
        return

    if not args.student_gen_path:
        args.student_gen_path = os.path.join(args.output_dir, "student_gen.parquet")
    if not os.path.exists(args.student_gen_path):
        raise FileNotFoundError(f"Student gen not found: {args.student_gen_path}")

    teacher_ip_pool = [ip.strip() for ip in args.teacher_ip_pool.split(",") if ip.strip()]
    print(f"Teacher IP pool: {teacher_ip_pool}")

    ppl_dir = args.ppl_dir if args.ppl_dir else args.output_dir
    os.makedirs(ppl_dir, exist_ok=True)

    # Step 2: Label student responses
    labeled_path = os.path.join(args.output_dir, "student_gen_labeled.parquet")
    if args.skip_label and os.path.exists(labeled_path):
        labeled_df = pd.read_parquet(labeled_path)
        df = labeled_df
        wrong_df = labeled_df[labeled_df["student_score"] == 0.0].copy().reset_index(drop=True)
    else:
        df, wrong_df = label_student_responses(args.student_gen_path, args.output_dir)

    # Step 4a: Compute teacher PPL
    ppl_path = os.path.join(ppl_dir, "wrong_samples_with_ppl.parquet")
    if args.skip_ppl and os.path.exists(ppl_path):
        wrong_df = pd.read_parquet(ppl_path)
    else:
        wrong_df = compute_teacher_ppl(
            wrong_df=wrong_df,
            teacher_ip_pool=teacher_ip_pool,
            teacher_model_name=args.teacher_model_name,
            student_tokenizer_path=args.student_model_path,
            ppl_dir=ppl_dir,
            teacher_max_model_len=args.teacher_max_model_len,
        )

    # Optional: filter length-truncated responses
    if args.filter_truncated:
        wrong_df = filter_truncated_responses(
            wrong_df=wrong_df,
            student_tokenizer_path=args.student_model_path,
            max_response_length=args.max_response_length,
        )

    # Step 4b: Bin by PPL and build prefixes
    prefix_path = os.path.join(args.output_dir, "wrong_samples_with_ppl_prefix.parquet")
    if args.skip_prefix and os.path.exists(prefix_path):
        wrong_df = pd.read_parquet(prefix_path)
        n_unique_bins = wrong_df["ppl_bin"].nunique()
        bin_labels = ["A_low_ppl", "B_high_ppl"] if n_unique_bins == 2 \
            else ["Q1_lowest_ppl", "Q2_low_ppl", "Q3_high_ppl", "Q4_highest_ppl"]
    else:
        wrong_df, bin_labels = split_and_build_prefix(
            wrong_df=wrong_df,
            student_tokenizer_path=args.student_model_path,
            output_dir=args.output_dir,
            n_bins=args.n_bins,
            split_method=args.split_method,
            truncate_ratio=args.truncate_ratio,
        )

    # Step 5: Teacher continuation
    wrong_df_with_cont, bin_results = teacher_continuation(
        wrong_df=wrong_df,
        bin_labels=bin_labels,
        teacher_ip_pool=teacher_ip_pool,
        teacher_model_name=args.teacher_model_name,
        student_tokenizer_path=args.student_model_path,
        output_dir=args.output_dir,
        n_continuations=args.n_continuations,
        temperature=args.teacher_temperature,
        max_tokens=args.max_tokens,
        teacher_max_model_len=args.teacher_max_model_len,
    )

    # Step 6: Summarize
    summarize_results(
        df=df,
        wrong_df=wrong_df_with_cont,
        bin_results=bin_results,
        output_dir=args.output_dir,
    )

    print(f"\nDone! Results: {args.output_dir}")


if __name__ == "__main__":
    main()
