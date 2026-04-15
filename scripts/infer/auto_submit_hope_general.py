import os
import subprocess

bash_file = './scripts/infer/infer_general_auto.sh' # bash文件路径

model_dict = {
    # 'DeepSeek-R1-Distill-Qwen-1.5B': './models/DeepSeek-R1-Distill-Qwen-1.5B',
    # 'Qwen3-1.7B-Base': './models/Qwen3-1.7B-Base',
}


dataset_dict = {
    "aime2024": 6,
    "aime2025": 6,
    # "amc23": 5,
    # "math500": 5,
    # 'minerva': 6,
    # "olympiadbench": 12,
}
DATA_PATH = "./data"
RESULT_PATH = "./results/math_0408"

# dataset_dict = {
#     "aime2024": 1,
#     "aime2025": 1,
#     # "amc23": 1,
#     # "math500": 1,
#     # 'minerva': 2,
#     # "olympiads": 4,
# }
# DATA_PATH = "./datasets/math"
# RESULT_PATH = "./results/math_0408"

rollout = 4
max_new_tokens = 32768
temperature = 0.6
top_p = 0.95
top_k = 20

# for base model
max_model_len = 32768
## for instruct model
# max_model_len=40960

# repetition penalty for small models prone to repetition (e.g. 1.7b)
repetition_penalty_for_1_7b = 1.08


for output_name, model_name_or_path in model_dict.items():
    for dataset_key, split_num in dataset_dict.items():
        for split_idx in range(split_num):

            actual_max_new_tokens = min(max_new_tokens, max_model_len)

            json_file = f"{DATA_PATH}/{dataset_key}_{split_idx}.json"
            output_path = f"{RESULT_PATH}/{dataset_key}_{output_name}_len{actual_max_new_tokens}_t{temperature}_{split_idx}_result.json"
            if '1_7b' in output_name:
                repetition_penalty = repetition_penalty_for_1_7b
            else:
                repetition_penalty = 1.0

            cmd = [
                "bash", "-e", bash_file,
                model_name_or_path,
                json_file,
                output_path,
                str(rollout),
                str(actual_max_new_tokens),
                str(max_model_len),
                str(temperature),
                str(top_p),
                str(top_k),
                str(repetition_penalty),
            ]

            print(f"Running: {output_name} | {dataset_key} split {split_idx}")
            print(f"CMD: {' '.join(cmd)}\n")

            result = subprocess.run(cmd, check=True)
