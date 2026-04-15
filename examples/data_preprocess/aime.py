# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the GSM8k dataset to parquet format
"""

import re
import os
import datasets

# from verl.utils.hdfs_io import copy, makedirs
import argparse

import json
import pandas as pd
from transformers import AutoTokenizer
from tqdm import tqdm
from sklearn.model_selection import train_test_split

tokenizer_path = "./model/Qwen3-0.6B"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
MAX_PROMPT_LENGTH = 16384
# SYSTEM_PROMPT = "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>."
data_source = 'aime'


def read_jsonl(file_path):
    data_list = []
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        for line in lines:
            temp_json = json.loads(line)
            data_list.append(temp_json)
    
    return data_list

def preprocess(question_raw, use_chat=True, add_think_token=True, use_system=False):
    post_turn_text = question_raw
    # turn_list = []
    # if use_chat:
    #     if use_system:
    #         turn_list.append({'role':'system','content':SYSTEM_PROMPT},{'role':'user','content':question_raw})
    #     else:
    #         turn_list.append({'role':'user','content':question_raw})
    #     post_turn_text = tokenizer.apply_chat_template(
    #         turn_list, 
    #         add_generation_prompt=True, 
    #         tokenize=False,
    #         enable_thinking=add_think_token
    #     )
    # if add_think_token:
    #     post_turn_text += "<think>\n"
    return post_turn_text

def make_map_fn(split, use_chat=True, add_think_token=True, use_system=False):
    def process_fn(example, idx, split, use_chat=use_chat, add_think_token=add_think_token, use_system=use_system):
        question = example.pop('question')
        # question = preprocess(question_raw, use_chat=use_chat, add_think_token=add_think_token)
        question = question.replace('\boxed{{}}','\\boxed{{}}')
        # tip_prompt = "Put your final answer within \boxed{{}}."
        # question = question + tip_prompt
        answer = example.pop('answer')

        print(question)

        # print('\n\n')
        # # print(question)
        # print(answer)
        # print('\n\n')

        data = {
            "data_source": data_source,  # 需要替换成实际的数据源
            "prompt": [{
                "role": "user",
                "content": question,
            }],
            "ability": "math",
            "reward_model": {
                "style": "rule",
                "ground_truth": answer
            },
            "extra_info": {
                'split': split,
                'index': idx,
            }
        }
        return data

    return process_fn

def process_json_file(data, split):
    process_fn = make_map_fn(split, use_chat=True, add_think_token=True, use_system=False)
    processed_data = [process_fn(example, idx, split) for idx, example in tqdm(enumerate(data))]
    print(f"原始数据数量: {len(processed_data)}")
    filter_processed_data = [item for item in processed_data if len(item['prompt'][0]['content'])<=MAX_PROMPT_LENGTH]
    print(f"过滤后数据数量: {len(filter_processed_data)}")
    
    return filter_processed_data

def save_to_parquet(data, output_file):
    print(f"数据已保存为 {output_file}")
    print(f"data_size: {len(data)}")
    df = pd.DataFrame(data)
    df.to_parquet(output_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_path', default='./maxing/verl-distillation-ori/data')
    parser.add_argument('--hdfs_dir', default=None)

    args = parser.parse_args()
    local_path = args.local_path

    split_ratio = -1
    input_path = './maxing/dpo_training/data/aime_2024_2025_test.json'
    test_data = read_jsonl(input_path)

    # if split_ratio == -1:
    #     train_data = input_data
    # else:
    #     train_data, test_data = train_test_split(input_data, test_size=split_ratio,random_state=42)

    test_processed_data = process_json_file(test_data, split='test')
    output_test_file = os.path.join(args.local_path,f'{data_source}/test.parquet') # 保存为Parquet文件
    save_to_parquet(test_processed_data, output_test_file)

    
    