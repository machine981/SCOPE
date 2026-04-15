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

import torch
from verl import DataProto
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import register, Dispatch
from verl.utils.api_interface import vllmAPIModelInterface
from transformers import AutoTokenizer

class vLLMScorerWorker(Worker):
    """
    外置 vLLM Client 模式的教师模型评分器。
    该 Worker 负责与外部 vLLM 服务通信，获取教师模型的 logprobs。
    """
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        self.role = kwargs.get('role')
        self.kwargs = kwargs

        # 加载 Tokenizer 用于 Tensor -> Text 转换
        # 优先从 actor_rollout_ref.model.path 获取，确保与学生模型一致
        model_path = self.config.get('model', {}).get('path', self.config.actor_rollout_ref.model.path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        self.client = None

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        """
        初始化外置 Client。
        """
        # 从配置中读取 API 参数
        # 兼容 distill.ip_pool 或 distill.server_url 等结构

        ip_pool = self.config.distill.ip_pool
        model_name = self.config.distill.model_name
        top_logprobs_k = self.config.distill.get('top_logprobs_k', 1)
        self.top_logprobs_k = top_logprobs_k

        self.client = vllmAPIModelInterface(
            model_name=model_name,
            ip_pool=ip_pool,
            top_k=top_logprobs_k,
        )

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_logprobs(self, data: DataProto) -> DataProto:
        """
        核心评分逻辑：仅提取 Response 部分的教师 logprobs。
        top_logprobs_k=1 时返回 teacher_log_probs (B, L)；
        top_logprobs_k>1 时额外返回 teacher_top_token_ids (B, L, K) 和 teacher_top_log_probs (B, L, K)。
        """
        input_ids = data.batch['input_ids'] # (B, 12288)
        responses = data.batch['responses'] # (B, 4096)
        attention_mask = data.batch['attention_mask'] # (B, 12288)

        batch_size = input_ids.shape[0]
        response_length = responses.shape[1] # 4096
        prompt_length = input_ids.shape[1] - response_length # 8192

        input_ids_list = [seq[mask==1].tolist() for seq, mask in zip(input_ids, attention_mask)]

        results = self.client.get_batch_answers(input_ids_list)

        # top_k=1 原有逻辑：构造采样 token 的标量 log_prob 并直接返回
        if self.top_logprobs_k <= 1:
            teacher_log_probs_res = torch.zeros((batch_size, response_length), dtype=torch.float32, device=input_ids.device)
            for i, res in enumerate(results):
                all_valid_lps = res.get('prompt_logprobs', [])
                res_mask = attention_mask[i, prompt_length:]
                actual_res_token_num = int(res_mask.sum().item())
                if actual_res_token_num > 0:
                    res_lps = all_valid_lps[-actual_res_token_num:]
                    lps_tensor = torch.tensor(res_lps, dtype=torch.float32, device=input_ids.device)
                    res_valid_indices = torch.nonzero(res_mask, as_tuple=True)[0]
                    teacher_log_probs_res[i, res_valid_indices] = lps_tensor
            return DataProto.from_dict(tensors={'teacher_log_probs': teacher_log_probs_res}).to('cpu')

        # top_k > 1：解析每个位置的 top-k {token_str: logprob} 字典，构造稀疏张量
        K = self.top_logprobs_k
        teacher_top_token_ids  = torch.zeros((batch_size, response_length, K), dtype=torch.int32,   device=input_ids.device)
        teacher_top_log_probs  = torch.full( (batch_size, response_length, K), fill_value=-1e9,     dtype=torch.float32, device=input_ids.device)

        for i, res in enumerate(results):
            all_top_lps = res.get('top_logprobs', [])  # List[Dict[str, float]]，长度=有效 token 数
            res_mask = attention_mask[i, prompt_length:]
            actual_res_token_num = int(res_mask.sum().item())

            if actual_res_token_num > 0 and len(all_top_lps) > 0:
                res_top_lps = all_top_lps[-actual_res_token_num:]  # 取 response 部分
                res_valid_indices = torch.nonzero(res_mask, as_tuple=True)[0]

                for pos_idx, (valid_pos, top_dict) in enumerate(zip(res_valid_indices, res_top_lps)):
                    if not isinstance(top_dict, dict):
                        continue
                    # top_dict: {token_str: logprob}，按 logprob 降序取前 K 个
                    sorted_items = sorted(top_dict.items(), key=lambda x: x[1], reverse=True)[:K]
                    for k_idx, (token_str, lp) in enumerate(sorted_items):
                        # 将 token_str 转回 token_id
                        token_id = self.tokenizer.convert_tokens_to_ids(token_str)
                        if token_id is None:
                            token_id = self.tokenizer.unk_token_id or 0
                        teacher_top_token_ids[i, valid_pos, k_idx] = token_id
                        teacher_top_log_probs[i, valid_pos, k_idx] = lp

        return DataProto.from_dict(tensors={
            'teacher_top_token_ids': teacher_top_token_ids,    # (B, L, K)
            'teacher_top_log_probs': teacher_top_log_probs,    # (B, L, K)
        }).to('cpu')
