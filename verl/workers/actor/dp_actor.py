# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
Single Process Actor
"""

import itertools
import logging
import os
from typing import Tuple

import torch
from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, compute_policy_loss, kl_penalty
from verl.utils.debug import GPUMemoryLogger
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import get_reverse_idx, rearrange_micro_batches
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import gather_outpus_and_unpad, ulysses_pad_and_slice_inputs
from verl.workers.actor import BasePPOActor

__all__ = ["DataParallelPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class DataParallelPPOActor(BasePPOActor):
    def __init__(self, config, actor_module: nn.Module, actor_optimizer: torch.optim.Optimizer = None):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        self.use_remove_padding = self.config.get("use_remove_padding", False)
        print(f"Actor use_remove_padding={self.use_remove_padding}")
        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        self.compute_entropy_from_logits = (
            torch.compile(verl_F.entropy_from_logits, dynamic=True)
            if self.config.get("use_torch_compile", True)  #  use torch compile by default
            else verl_F.entropy_from_logits
        )

    def _forward_micro_batch(self, micro_batch, temperature, calculate_entropy=False, return_logits=False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
            logits (optional): # (bs, response_len, vocab_size), only when return_logits=True
        """
        response_length = micro_batch["responses"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch:
            for key in micro_batch["multi_modal_inputs"][0].keys():
                multi_modal_inputs[key] = torch.cat([inputs[key] for inputs in micro_batch["multi_modal_inputs"]], dim=0)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            entropy = None
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 3, seqlen) -> (3, bsz, seqlen)

            if self.use_remove_padding:
                input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices).transpose(0, 1).unsqueeze(1)  # (3, bsz, seqlen) -> (3, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices).transpose(0, 1)

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(input_ids_rmpad, position_ids_rmpad, sp_size=self.ulysses_sequence_parallel_size)
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(input_ids_rmpad_rolled, None, self.ulysses_sequence_parallel_size)

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                output = self.actor_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                )  # prevent model thinks we are generating
                logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)

                logits_rmpad.div_(temperature)

                # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                inplace_backward = True
                if calculate_entropy or return_logits:
                    inplace_backward = False
                log_probs = logprobs_from_logits(logits=logits_rmpad, labels=input_ids_rmpad_rolled, inplace_backward=inplace_backward)

                # compute entropy
                if calculate_entropy:
                    entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outpus_and_unpad(log_probs, gather_dim=0, unpad_dim=0, padding_size=pad_size)
                    if calculate_entropy:
                        entropy_rmpad = gather_outpus_and_unpad(entropy_rmpad, gather_dim=0, unpad_dim=0, padding_size=pad_size)
                # pad back to (bsz, seqlen)
                if calculate_entropy:
                    full_entropy = pad_input(hidden_states=entropy_rmpad.unsqueeze(-1), indices=indices, batch=batch_size, seqlen=seqlen)
                full_log_probs = pad_input(hidden_states=log_probs.unsqueeze(-1), indices=indices, batch=batch_size, seqlen=seqlen)

                # only return response part:
                if calculate_entropy:
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)

                # return_logits: pad rmpad logits back and slice response part
                if return_logits:
                    full_logits = pad_input(hidden_states=logits_rmpad, indices=indices, batch=batch_size, seqlen=seqlen)
                    # pad_input: (total_nnz, vocab_size) -> (bsz, seqlen, vocab_size)
                    logits = full_logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)

            else:  # not using rmpad and no ulysses sp
                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                )  # prevent model thinks we are generating
                logits = output.logits
                logits.div_(temperature)
                logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
                log_probs = logprobs_from_logits(logits, micro_batch["responses"])
                if calculate_entropy:
                    entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)

            if return_logits:
                return entropy, log_probs, logits
            return entropy, log_probs

    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        elif isinstance(self.actor_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)

        # if grad_norm is not finite, skip the update
        if not torch.isfinite(grad_norm):
            print(f"WARN: rank {torch.distributed.get_rank()} grad_norm is not finite: {grad_norm}")
            self.actor_optimizer.zero_grad()
        else:
            self.actor_optimizer.step()
        return grad_norm

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob(self, data: DataProto, calculate_entropy=False) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]

        # top-k KL reward 模式：需要学生 logits 来计算 top-k KL，替换 teacher_log_probs
        use_topk_kl_reward = (
            not self.config.distill_signal_in_loss
            and "teacher_top_token_ids" in data.batch.keys()
        )

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        if use_topk_kl_reward:
            select_keys += ["teacher_top_token_ids", "teacher_top_log_probs"]
        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()

        if has_multi_modal_inputs:
            num_micro_batches = data.batch.batch_size[0] // micro_batch_size
            non_tensor_select_keys = ["multi_modal_inputs"]
            micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
        elif use_dynamic_bsz:
            # split using dynamic bsz
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, indices = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)
        else:
            micro_batches = batch.split(micro_batch_size)

        log_probs_lst = []
        entropy_lst = []
        topk_kl_lst = []
        for micro_batch in micro_batches:
            if isinstance(micro_batch, DataProto):
                micro_batch = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            with torch.no_grad():
                if use_topk_kl_reward:
                    entropy, log_probs, logits = self._forward_micro_batch(micro_batch, temperature=temperature, calculate_entropy=calculate_entropy, return_logits=True)
                    # 用 logits 在教师 top-k token 位置 gather 学生 log_prob，计算反向 KL
                    teacher_top_ids = micro_batch["teacher_top_token_ids"].long()   # (B, L, K)
                    teacher_top_lp  = micro_batch["teacher_top_log_probs"]          # (B, L, K)
                    student_top_logits = torch.gather(logits, dim=-1, index=teacher_top_ids)                      # (B, L, K)
                    student_top_log_probs = torch.nn.functional.log_softmax(student_top_logits, dim=-1)           # (B, L, K)
                    teacher_top_log_probs_renorm = torch.nn.functional.log_softmax(teacher_top_lp, dim=-1)        # (B, L, K)
                    topk_kl = (student_top_log_probs.exp() * (student_top_log_probs - teacher_top_log_probs_renorm)).sum(dim=-1)  # (B, L)
                    topk_kl_lst.append(topk_kl)
                else:
                    entropy, log_probs = self._forward_micro_batch(micro_batch, temperature=temperature, calculate_entropy=calculate_entropy)
            log_probs_lst.append(log_probs)
            if calculate_entropy:
                entropy_lst.append(entropy)

        log_probs = torch.concat(log_probs_lst, dim=0)
        entropys = None
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)

        if use_dynamic_bsz:
            indices = list(itertools.chain.from_iterable(indices))
            assert len(indices) == log_probs.size(0), f"{len(indices)} vs. {log_probs.size()}"
            revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
            log_probs = log_probs[revert_indices]
            if calculate_entropy:
                entropys = entropys[revert_indices]

        result = (log_probs, entropys)
        if use_topk_kl_reward:
            # 拼接后作为新的 teacher_log_probs 写回，供 compute_advantage 使用
            topk_kl = torch.concat(topk_kl_lst, dim=0)
            if use_dynamic_bsz:
                topk_kl = topk_kl[revert_indices]
            result = (log_probs, entropys, topk_kl)
        return result

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        multi_turn = data.meta_info.get("multi_turn", False)

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids", "old_log_probs", "advantages"]
        if multi_turn:
            select_keys.append("loss_mask")
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")
        if self.config.distill_signal_in_loss:
            if "teacher_top_token_ids" in data.batch.keys():
                select_keys.append("teacher_top_token_ids")
                select_keys.append("teacher_top_log_probs")
            else:
                select_keys.append("teacher_log_probs")
        # SCOPE 双路径自适应加权权重（如果存在则加入）
        if "student_path_weights" in data.batch.keys():
            select_keys.append("student_path_weights")
            select_keys.append("teacher_path_weights")
        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        if has_multi_modal_inputs:
            num_mini_batches = data.batch.batch_size[0] // self.config.ppo_mini_batch_size
            non_tensor_select_keys = ["multi_modal_inputs"]
            dataloader = data.select(select_keys, non_tensor_select_keys).chunk(num_mini_batches)
        else:
            dataloader = batch.split(self.config.ppo_mini_batch_size)

        metrics = {}
        for epoch in range(self.config.ppo_epochs):
            for batch_idx, data in enumerate(dataloader):
                # split batch into micro_batches
                mini_batch = data
                if has_multi_modal_inputs:
                    self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    num_micro_batches = mini_batch.batch.batch_size[0] // self.config.ppo_micro_batch_size_per_gpu
                    micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
                elif self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = rearrange_micro_batches(batch=mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    # split batch into micro_batches
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()

                for data in micro_batches:
                    # Support all hardwares
                    if isinstance(data, DataProto):
                        data = {**data.batch.to(torch.cuda.current_device()), **data.non_tensor_batch}
                    else:
                        data = data.to(torch.cuda.current_device())  # actor device is cpu when using offload
                    responses = data["responses"]
                    response_length = responses.size(1)
                    attention_mask = data["attention_mask"]
                    if multi_turn:
                        response_mask = data["loss_mask"][:, -response_length:]
                    else:
                        response_mask = attention_mask[:, -response_length:]

                    old_log_prob = data["old_log_probs"]
                    advantages = data["advantages"]

                    clip_ratio = self.config.clip_ratio
                    clip_ratio_low = self.config.clip_ratio_low if self.config.clip_ratio_low is not None else clip_ratio
                    clip_ratio_high = self.config.clip_ratio_high if self.config.clip_ratio_high is not None else clip_ratio
                    clip_ratio_c = self.config.get("clip_ratio_c", 3.0)
                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode

                    # all return: (bsz, response_length)
                    calculate_entropy = entropy_coeff != 0

                    # 只调用一次 _forward_micro_batch，避免重复计算和内存浪费
                    need_logits = self.config.distill_signal_in_loss and "teacher_top_token_ids" in data
                    if need_logits:
                        entropy, log_prob, full_logits = self._forward_micro_batch(micro_batch=data, temperature=temperature, calculate_entropy=calculate_entropy, return_logits=True)
                    else:
                        entropy, log_prob = self._forward_micro_batch(micro_batch=data, temperature=temperature, calculate_entropy=calculate_entropy)

                    pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = compute_policy_loss(
                        old_log_prob=old_log_prob,
                        log_prob=log_prob,
                        entropy=entropy,
                        advantages=advantages,
                        response_mask=response_mask,
                        cliprange=clip_ratio,
                        cliprange_low=clip_ratio_low,
                        cliprange_high=clip_ratio_high,
                        clip_ratio_c=clip_ratio_c,
                        loss_agg_mode=loss_agg_mode,
                    )

                    if entropy_coeff != 0:
                        entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        # compute policy loss
                        policy_loss = pg_loss - entropy_loss * entropy_coeff
                    else:
                        policy_loss = pg_loss

                    if self.config.use_kl_loss:
                        ref_log_prob = data["ref_log_prob"]
                        # compute kl loss
                        kld = kl_penalty(logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type)
                        kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=self.config.loss_agg_mode)

                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        metrics["actor/kl_loss"] = kl_loss.detach().item()
                        metrics["actor/kl_coef"] = self.config.kl_loss_coef

                    # SCOPE 双路径自适应加权：IK路径正确 response 的加权 SFT loss
                    if "student_path_weights" in data:
                        cw = data["student_path_weights"]  # (B,) float32
                        sft_loss_mat = -log_prob * cw.unsqueeze(-1)  # (B, L)
                        # 分母只计正确样本的有效 token，避免被错误样本稀释
                        correct_token_mask = (cw > 0).unsqueeze(-1).float() * response_mask.float()  # (B, L)
                        sft_loss = agg_loss(loss_mat=sft_loss_mat, loss_mask=correct_token_mask, loss_agg_mode=self.config.loss_agg_mode)
                        policy_loss = policy_loss + sft_loss
                        append_to_dict(metrics, {"actor/sft_loss": sft_loss.detach().item()})

                    if self.config.distill_signal_in_loss:
                        if "teacher_top_token_ids" in data:
                            # top-k 反向 KL: KL(P_S || P_T) ≈ Σ_{v∈top-k} P_S(v)*[logP_S(v) - logP_T(v)]
                            teacher_top_ids = data["teacher_top_token_ids"].long()   # (B, L, K)
                            teacher_top_lp  = data["teacher_top_log_probs"]          # (B, L, K)

                            # 直接使用 torch.gather 提取 top-k logits
                            student_top_logits = torch.gather(full_logits, dim=-1, index=teacher_top_ids)  # (B, L, K)

                            # 立即清理 full_logits 释放内存
                            del full_logits
                            torch.cuda.empty_cache()

                            # 学生和教师都在 top-k 子词表上重新归一化，保证两侧分布定义在同一子词表上
                            student_top_log_probs = torch.nn.functional.log_softmax(student_top_logits, dim=-1)             # (B, L, K)
                            teacher_top_log_probs_renorm = torch.nn.functional.log_softmax(teacher_top_lp, dim=-1)          # (B, L, K)

                            # 反向 KL: Σ_k P_S(k) * [log P_S(k) - log P_T(k)]
                            distill_kld = (student_top_log_probs.exp() * (student_top_log_probs - teacher_top_log_probs_renorm)).sum(dim=-1)  # (B, L)

                            # SCOPE 双路径自适应加权：DR路径错误 response 的 OPD 权重（D 类 > B 类）
                            if "teacher_path_weights" in data:
                                ww = data["teacher_path_weights"]  # (B,) float32
                                distill_kld = distill_kld * ww.unsqueeze(-1)
                                # 分母只计错误样本的有效 token
                                wrong_token_mask = (ww > 0).unsqueeze(-1).float() * response_mask.float()  # (B, L)
                                distill_loss = agg_loss(loss_mat=distill_kld, loss_mask=wrong_token_mask, loss_agg_mode=self.config.loss_agg_mode)
                            else:
                                distill_loss = agg_loss(loss_mat=distill_kld, loss_mask=response_mask, loss_agg_mode=self.config.loss_agg_mode)
                            policy_loss = policy_loss + distill_loss
                            append_to_dict(metrics, {"actor/distill_loss": distill_loss.detach().item()})
                        else:
                            # 原有 top_k=1 逻辑
                            student_logprob = log_prob
                            teacher_logprob = data["teacher_log_probs"]
                            distill_kl_type = self.config.distill_loss_type
                            distill_kld = kl_penalty(student_logprob, teacher_logprob, kl_penalty=distill_kl_type)

                            # SCOPE 双路径自适应加权：DR路径错误 response 的 OPD 权重（D 类 > B 类）
                            if "teacher_path_weights" in data:
                                ww = data["teacher_path_weights"]  # (B,) float32
                                distill_kld = distill_kld * ww.unsqueeze(-1)
                                # 分母只计错误样本的有效 token
                                wrong_token_mask = (ww > 0).unsqueeze(-1).float() * response_mask.float()  # (B, L)
                                distill_loss = agg_loss(loss_mat=distill_kld, loss_mask=wrong_token_mask, loss_agg_mode=self.config.loss_agg_mode)
                            else:
                                distill_loss = agg_loss(loss_mat=distill_kld, loss_mask=response_mask, loss_agg_mode=self.config.loss_agg_mode)
                            policy_loss = policy_loss + distill_loss
                            append_to_dict(metrics, {"actor/distill_loss": distill_loss.detach().item()})

                    if self.config.use_dynamic_bsz:
                        # relative to the dynamic bsz
                        loss = policy_loss * (len(data) / self.config.ppo_mini_batch_size)
                    else:
                        loss = policy_loss / self.gradient_accumulation
                    loss.backward()

                    # 清理 CUDA 缓存以减少碎片化
                    torch.cuda.empty_cache()

                    data = {
                        "actor/pg_loss": pg_loss.detach().item(),
                        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
                        "actor/ppo_kl": ppo_kl.detach().item(),
                        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
                    }
                    append_to_dict(metrics, data)

                grad_norm = self._optimizer_step()
                data = {"actor/grad_norm": grad_norm.detach().item()}
            append_to_dict(metrics, data)
        self.actor_optimizer.zero_grad()
        return metrics
