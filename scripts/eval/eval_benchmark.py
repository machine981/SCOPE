import json
import random
from collections import defaultdict
from functools import partial

import numpy as np

from verl.utils.reward_score import math, gsm8k, math_verify, math_robust
from verl.utils.reward_score.prime_math import compute_score as prime_math_compute_score

def read_jsonl(file_path):
    data_list = []
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        for line in lines:
            temp_json = json.loads(line)
            data_list.append(temp_json)

    return data_list

def write_jsonl(data_list, output_path):
    with open(output_path, 'w', encoding='utf-8') as file:
        for item in data_list:
            file.write(json.dumps(item, ensure_ascii=False))
            file.write('\n')

    print(f"文件生成至{output_path}.")

def compute_score(solution_str, ground_truth, data_source):
    if data_source == "openai/gsm8k":
        res = gsm8k.compute_score(solution_str, ground_truth, method='flexible')
        # res = math_verify.compute_score(solution_str, ground_truth)
        # res = math_robust.compute_score(solution_str, ground_truth)
    elif data_source in ["math500"]:
        res = math_verify.compute_score(solution_str, ground_truth)
        # res = math_robust.compute_score(solution_str, ground_truth)
    elif data_source in ["aime2024", "aime2025", "amc23", "minerva"]:
        # res = math_verify.compute_score(solution_str, ground_truth)
        res = math_robust.compute_score(solution_str, ground_truth)
    elif data_source in ['olympiadbench', "olympiads"]:
        # ground_truth 是 list，取第一个元素；去掉 LaTeX 的 $ 符号
        if isinstance(ground_truth, list):
            ground_truth = ground_truth[0]
        ground_truth = str(ground_truth).strip().strip('$').strip()
        res = prime_math_compute_score(solution_str, ground_truth)
        # prime_math 返回 (is_correct, format_correctness, extracted)，取第一个
        if isinstance(res, tuple):
            res = res[0]
    else:
        raise NotImplementedError(f"No support score function for datasource - {data_source}")

    return res

def bootstrap_metric(data, subset_size, reduce_fns, n_bootstrap=1000, seed=42):
    """
    有放回 bootstrap 采样，估计各 reduce_fn 的均值和标准差。
    与 verl metric_utils.bootstrap_metric 逻辑一致。
    返回 [(mean, std), ...] 对应每个 reduce_fn。
    """
    np.random.seed(seed)
    bootstrap_metric_lsts = [[] for _ in range(len(reduce_fns))]
    for _ in range(n_bootstrap):
        bootstrap_idxs = np.random.choice(len(data), size=subset_size, replace=True)
        bootstrap_data = [data[i] for i in bootstrap_idxs]
        for i, reduce_fn in enumerate(reduce_fns):
            bootstrap_metric_lsts[i].append(reduce_fn(bootstrap_data))
    return [(np.mean(lst), np.std(lst)) for lst in bootstrap_metric_lsts]


def _eval_scores_for_item(item):
    """
    对一条数据的所有候选答案逐一打分，返回 (data_source, score_list)。
    """
    data_source = item['data_source']
    ground_truth = item['answer']
    total_solution_list = item["candidate_generated"] if len(item["candidate_generated"]) > 0 else [item['generated']]
    score_list = [
        compute_score(solution_str=sol, ground_truth=ground_truth, data_source=data_source)
        for sol in total_solution_list
    ]
    return data_source, score_list


def single_eval(item, eval_mode="acc", at_num=None):
    """
    eval_mode: options in ['acc', 'avg@k', 'pass@k']

    avg@k  : mean@N，即所有候选答案的均值（at_num 忽略，始终使用全部候选）。
    pass@k : best@k/mean，用有放回 bootstrap 采样（1000次）从 N 个候选中取 k 个的最大值均值。
             at_num 指定 k；若 at_num=None 则 k=N（等价于 avg@k）。
    acc    : 仅对第一条候选答案评分（兼容旧逻辑）。
    """
    data_source = item['data_source']
    ground_truth = item['answer']
    total_solution_list = item["candidate_generated"] if len(item["candidate_generated"]) > 0 else [item['generated']]
    candidate_solution_num = len(total_solution_list)

    if at_num is not None:
        if candidate_solution_num < at_num:
            raise ValueError(
                f"at_num ({at_num}) exceeds the number of rollouts ({candidate_solution_num})"
            )
    else:
        at_num = candidate_solution_num

    if eval_mode == "acc":
        res = compute_score(
            solution_str=total_solution_list[0],
            ground_truth=ground_truth,
            data_source=data_source,
        )

    elif eval_mode == "avg@k":
        # mean@N：所有候选答案分数的均值，与 verl 的 mean@N 一致
        score_list = [
            compute_score(solution_str=sol, ground_truth=ground_truth, data_source=data_source)
            for sol in total_solution_list
        ]
        res = float(np.mean(score_list))

    elif eval_mode == "pass@k":
        # best@k/mean：有放回 bootstrap，每次从 N 个候选中采样 k 个取 max，重复 1000 次后取均值
        # 与 verl metric_utils.process_validation_metrics 中 best@N/mean 逻辑一致
        score_list = [
            compute_score(solution_str=sol, ground_truth=ground_truth, data_source=data_source)
            for sol in total_solution_list
        ]
        if at_num == candidate_solution_num and at_num == 1:
            # 只有 1 条候选，直接返回该分数
            res = float(score_list[0])
        else:
            [(bon_mean, _)] = bootstrap_metric(
                data=score_list,
                subset_size=at_num,
                reduce_fns=[np.max],
                n_bootstrap=1000,
                seed=42,
            )
            res = float(bon_mean)

    else:
        raise NotImplementedError(f"Not implement eval mode {eval_mode}.")

    return res, data_source, at_num


def eval_result(data, eval_mode="acc", at_num=None, print_avg=False):
    """
    聚合方式与 verl process_validation_metrics 一致：
      1. 对每条 item（每个 prompt）计算指标值
      2. 先在 data_source 内对所有 prompt 取均值（宏平均）
      3. 再跨 data_source 取均值（macro score）
    """
    # data_source -> list of per-prompt scores
    ds2scores = defaultdict(list)
    ds2at_num = {}

    for item in data:
        score, data_source, real_at_num = single_eval(item, eval_mode=eval_mode, at_num=at_num)
        ds2scores[data_source].append(score)
        ds2at_num[data_source] = real_at_num

    total_count = 0
    sum_score = 0
    macro_score = 0
    total_data_source = len(ds2scores)

    for data_source, scores in ds2scores.items():
        single_score = float(np.mean(scores))
        macro_score += single_score
        sum_score += sum(scores)
        total_count += len(scores)
        real_eval_mode = eval_mode.replace('k', str(ds2at_num[data_source]))
        print(f"[{data_source}]--{real_eval_mode}: {single_score * 100:.3g}%")

    if print_avg:
        print(f"Micro Score: {(sum_score / total_count) * 100:.3g}%")
        print(f"Macro Score: {(macro_score / total_data_source) * 100:.3g}%")


if __name__ == "__main__":
    # DATA_PATH = "data"
    RESULT_PATH = "results/math_0408"

    model_dict = {
    'DeepSeek-R1-Distill-Qwen-1.5B': 'models/DeepSeek-R1-Distill-Qwen-1.5B',
    'Qwen3-1.7B-Base': 'models/Qwen3-1.7B-Base',
}



    # dataset_dict = {
    #     "aime2024": 1,
    #     "aime2025": 1,
    #     # "amc23": 1,
    #     # "math500": 1,
    #     # 'minerva': 2,
    #     # "olympiads": 4,
    # }
    dataset_dict = {
        "aime2024": 6,
        "aime2025": 6,
        # "amc23": 5,
        # "math500": 5,
        # 'minerva': 6,
        # "olympiadbench": 12,
    }

    eval_list = []

    max_model_len=32768

    for k, v in model_dict.items():
        print(f"\n===== {k} =====\n")
        for dataset, split_num in dataset_dict.items():
            if 'aime' in dataset:
                max_token=38912
            else:
                max_token=32768

            max_token=16384

            max_token = min(max_token, max_model_len)
            RESULT_JSON = [f"{RESULT_PATH}/{dataset}_{k}_len{max_token}_t0.6_{i}_result.json" for i in range(split_num)]

            total_data_list = []

            for file_path in RESULT_JSON:
                if '2024' in dataset:
                    datasource = 'aime2024'
                elif '2025' in dataset:
                    datasource = 'aime2025'
                else:
                    datasource = dataset
                part_data = read_jsonl(file_path)
                for idx, item in enumerate(part_data):
                    part_data[idx]['data_source'] = datasource

                total_data_list.extend(part_data)

            # for i in [1,2,4,8,16]:
            eval_result(total_data_list, eval_mode="pass@k", at_num=32)

        print("\n")
