# -*- coding: utf-8 -*-
import json
import time
import requests
import math
from typing import List, Dict, Any, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import random
from openai import OpenAI
import itertools

import ipaddress

def validate_ip(ip_list):
    invalid_ips = []
    valid_ips = []

    for ip in ip_list:
        if not ip:
            continue
        try:
            ipaddress.ip_address(ip)
            valid_ips.append(ip)
        except ValueError:
            invalid_ips.append(ip)

    # 3. 返回结果
    if len(invalid_ips) == 0:
        return True, valid_ips
    else:
        return False, invalid_ips

class vllmAPIModelInterface:
    def __init__(self,
                 app_id: str = "xxx",
                 model_name: str = "",
                 max_tokens: int = 2000,
                 max_retries: int = 3,
                 timeout: int = 60,
                 top_k: int = 30,
                 ip_pool: list = [],
                 **kwargs):
        self.app_id = app_id
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.timeout = timeout
        self.top_k = top_k

        final_ip_pool = []
        ip_flag, final_ip_pool = validate_ip(ip_pool)

        if len(final_ip_pool) == 0:
            self.ip_pool = ['10.164.79.193']
            print(f"未识别到输入IP池, 采用默认IP池: {self.ip_pool}")
        elif len(final_ip_pool)!=0 and not ip_flag:
            raise ValueError(f"存在非法IP, 非法IP为: {final_ip_pool}")
        else:
            print(f"识别到有效输入IP池, 采用输入IP池: {final_ip_pool}")
            self.ip_pool = final_ip_pool

        self.config = kwargs
        self._initialize_client()

    def _initialize_client(self):
        print(f"API客户端初始化完成 - 模型: {self.model_name}")
        self.client_pool = []
        for item in self.ip_pool:
            client = OpenAI(
                base_url="http://{}:8000/v1".format(item),
                api_key='vocexp'
            )
            self.client_pool.append(client)
        self.client_pool_iterator = itertools.cycle(self.client_pool)
        

    def get_single_answer(self, prompt: str) -> Union[str, Dict[str, Any]]:
        for attempt in range(self.max_retries):
            try:
                result = self._call_api_single(prompt)
                return result
            except Exception as e:
                if attempt == self.max_retries - 1:
                    print(e)
                    error_msg = f"Error: {str(e)}"
                    result = {"prompt": error_msg, "prompt_logprobs": [0.0]}
                    if self.top_k > 1:
                        result["top_logprobs"] = []
                    return result

                time.sleep(1)

    def get_batch_answers(self, prompts: List[str], max_workers: int = 100) -> List[Union[str, Dict[str, Any]]]:
        if not prompts:
            return []

        answers = [None] * len(prompts)
        total = len(prompts)
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_index = {
                executor.submit(self.get_single_answer, prompt): i 
                for i, prompt in enumerate(prompts)
            }
            
            # completed_count = 0
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    answers[index] = future.result()
                except Exception as e:
                    print(e)
                    error_msg = f"Exception: {str(e)}"
                    err_res = {"prompt": error_msg, "prompt_logprobs": [0.0]}
                    if self.top_k > 1:
                        err_res["top_logprobs"] = []
                    answers[index] = err_res

        return answers

    def _call_api_single(self, prompt: str) -> Union[str, Dict[str, Any]]:
        next_client = next(self.client_pool_iterator)

        response = next_client.completions.create(
                model=self.model_name,
                prompt=prompt,
                logprobs=self.top_k,
                temperature=1.0,
                max_tokens=1,
                echo=True,
            )
            
        choice = response.choices[0]

        try:
            if not choice:
                result = {"prompt": "Error: No choices", "prompt_logprobs": [0.0]}
                if self.top_k > 1:
                    result["top_logprobs"] = []
                return result

            # 处理正常情况
            all_token_logprobs = choice.logprobs.token_logprobs
            all_tokens = choice.logprobs.tokens

            if len(all_token_logprobs) > 0:
                all_token_logprobs[0] = 0.0

            prompt_tokens = all_tokens[:-1]
            prompt_logprobs = all_token_logprobs[:-1]

            result = {
                "prompt": prompt_tokens,
                "prompt_logprobs": prompt_logprobs
            }

            # top_k > 1: 额外返回每个位置的 top-k {token_str: logprob} 字典列表
            if self.top_k > 1:
                all_top_logprobs = response.choices[0].logprobs.top_logprobs  # List[Dict[str, float]]
                if all_top_logprobs is not None:
                    result["top_logprobs"] = all_top_logprobs[:-1]  # 去掉最后一个 (max_tokens=1 生成的 token)
                else:
                    result["top_logprobs"] = []

            return result

        except Exception as e:
            print(e)
            if self.top_k > 1:
                return {
                    "prompt": [],
                    "prompt_logprobs": [],
                    "top_logprobs": [],
                }
            else:
                return {
                    "prompt": [],
                    "prompt_logprobs": [],
                }

def load_data_from_jsonl(file_path: str) -> List[Dict]:

    data_list = []
    print(f"正在读取文件: {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                try:
                    item = json.loads(line)
                    if 'input' in item: data_list.append(item)
                except: pass
    except FileNotFoundError:
        print(f"错误: 找不到文件 {file_path}")
        return []
    print(f"成功加载 {len(data_list)} 条有效数据")
    return data_list

def main():
    # 1. 加载测试数据
    input_file_path = '/mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtai/users/songjiaxing/data/testdata/testdata_paotui_nocot_v8.json'
    raw_data = load_data_from_jsonl(input_file_path)

    exp_n = 1

    if not raw_data:
        print("未找到测试数据，请检查路径。")
        return

    # 2. 提取测试 Prompt
    prompts = [item['input'] for item in raw_data][:exp_n]
    
    # 3. 初始化接口
    print("\n=== 开始测试 prompt_logprobs 返回功能 ===")
    api = vllmAPIModelInterface(model_name="Qwen3-8B")
    
    # 4. 调用接口
    start_time = time.time()
    results = api.get_batch_answers(prompts)
    elapsed_time = time.time() - start_time
    
    print(f"推理耗时: {elapsed_time:.4f} 秒\n")

    # 5. 详细验证
    for i, res in enumerate(results):
        print(f"--- 样本 {i+1} 验证 ---")
        if isinstance(res, dict) and "prompt_logprobs" in res:
            tokens = res.get("prompt", [])
            logprobs = res.get("prompt_logprobs", [])

            print(f"Prompt Token 数量: {len(tokens)}")
            print(f"Logprobs 数量: {len(logprobs)}")

            if len(logprobs) > 0:
                # 打印前 exp_n 个示例
                sample_len = min(exp_n, len(logprobs))
                logprobs[0] = 0.0

                # print(logprobs)

                # 计算平均概率
                avg_lp = sum(logprobs) / len(logprobs)
                print(f"平均 Logprob: {avg_lp:.4f}")

                if "topk_indices" in res:
                    print(f"Top-K Indices 形状: {res['topk_indices'].shape}")
                    print(f"Top-K Logprobs 形状: {res['topk_logprobs'].shape}")
                    print(f"前 5 个 Token 的前 3 个 Top-K 索引:\n{res['topk_indices'][:5, :3]}")
                    print(f"前 5 个 Token 的前 3 个 Top-K logprob:\n{res['topk_logprobs'][:5, :3]}")

                print("状态: [正常]")
            else:
                print("状态: [异常] logprobs 为空")
        else:
            print(f"状态: [失败] 返回格式异常: {res}")
        print("-" * 20)
        
if __name__ == "__main__":
    main()
