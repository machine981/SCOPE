#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import logging
import os
import time
import torch
import random
from transformers import AutoTokenizer, AutoConfig
from vllm import LLM, SamplingParams
# from tensor_parallel import TensorParallelPreTrainedModel
# import vllm
# print(vllm.__version__)

from tqdm import tqdm


def get_logger(name, to_file, level=logging.INFO):
    logger = logging.getLogger(name=name)
    logger.setLevel(level=level)
    formatter = logging.Formatter('%(asctime)s-%(name)s-%(levelname)s-%(message)s')

    handler = logging.FileHandler(filename=to_file)
    handler.setLevel(level=level)
    handler.setFormatter(formatter)

    console = logging.StreamHandler()
    console.setLevel(level=level)
    console.setFormatter(formatter)

    logger.addHandler(handler)
    logger.addHandler(console)
    return logger


def get_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--local_rank", required=False, type=int, help="used by dist launchers")
    parser.add_argument(
        "--json_file",
        type=str,
        default=None,
        help="The json file to be use.",
        required=True
    )
    # parser.add_argument("--json_type", type=str, default='mtc',
    #                     help="json file data type, mtc:muti-turn-conv, megatron:input-target", required=True)

    parser.add_argument("--output_path", type=str, help="generation result path", required=True)
    parser.add_argument("--model_name_or_path", type=str, help="Name path", required=True)
    parser.add_argument("--debug", default=0, type=int, help="if > 0, debug mode")
    parser.add_argument("--rollout", default=1, type=int, help="if > 0, debug mode")
    parser.add_argument("--max_new_tokens", default=2048, type=int, help="max_new_tokens for generation")
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.6,
        help="""Float that controls the randomness of the sampling. Lower
            values make the llm more deterministic, while higher values make
            the llm more random. Zero means greedy sampling."""
    )
    parser.add_argument(
        "--presence_penalty",
        type=float,
        default=0,
        help="""Float that penalizes new tokens based on whether they
            appear in the generated text so far. Values > 0 encourage the llm
            to use new tokens, while values < 0 encourage the llm to repeat
            tokens.""",
    )
    parser.add_argument(
        "--frequency_penalty",
        type=float,
        default=0,
        help="""Float that penalizes new tokens based on their
            frequency in the generated text so far. Values > 0 encourage the
            llm to use new tokens, while values < 0 encourage the llm to
            repeat tokens.""",
    )
    parser.add_argument('--top_k', type=int, default=-1)
    parser.add_argument('--top_p', type=float, default=1)
    parser.add_argument('--max_model_len', type=int, default=8192)
    parser.add_argument('--repetition_penalty', type=float, default=1.0)
    parser.add_argument('--retry_num', type=int, default=0, help='重复思考次数')
    parser.add_argument('--sample_duplicate', type=lambda x: (str(x).lower() == 'true'), default=True, help='关键词是否重复出现')
    return parser.parse_args()

def read_jsonl(file_path):
    data_list = []
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        for line in lines:
            temp_json = json.loads(line)
            data_list.append(temp_json)
    
    return data_list

if __name__ == '__main__':
    args = get_args()
    json_file = args.json_file
    output_path = args.output_path
    model_name_or_path = args.model_name_or_path
    max_new_tokens = args.max_new_tokens
    temperature = args.temperature
    presence_penalty = args.presence_penalty
    frequency_penalty = args.frequency_penalty
    top_p = args.top_p
    top_k = args.top_k
    repetition_penalty = args.repetition_penalty
    debug = args.debug
    rollout = args.rollout
    max_model_len = args.max_model_len

    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    logger = get_logger(name="inference", to_file=os.path.join(output_dir, "infer.log"))
    logger.info("to deal: {}".format(json_file))
    logger.info("======= Output path: {} =======".format(output_dir))

    """
    静态生成：预设的query和参考回复作为上下文
    """
    world_size = torch.cuda.device_count()
    logger.info("*** Using {} gpus".format(world_size))
    
    logger.info("======= Loading llm {} =======".format(model_name_or_path))
    config = AutoConfig.from_pretrained(
        model_name_or_path,
        trust_remote_code=False,
    )

    logger.info("======= llm config: {} =======".format(config))

    # context_length = get_context_length(config)
    model_max_length = max_new_tokens
    max_input_length = model_max_length
    logger.info("*** max_input_length = context_length({}) = {}".format(model_max_length, max_input_length))

    
    device_ids = ["cuda:{}".format(i) for i in range(world_size)]
    llm = LLM(
        model=model_name_or_path,
        trust_remote_code=True,
        tensor_parallel_size=world_size,
        gpu_memory_utilization=0.9,
        max_model_len=max_model_len,
    )
    
    
    tokenizer = llm.get_tokenizer()
    stop_token = tokenizer.eos_token
    inference_params = {
        "n": rollout,
        "temperature": temperature,
        "max_tokens": max_new_tokens,
        "stop": [stop_token],
        "top_k": top_k,
        "top_p": top_p,
        "presence_penalty": presence_penalty,
        "frequency_penalty": frequency_penalty,
        # "repetition_penalty": repetition_penalty,
    }
    logger.info("*** Inference with params: {}".format(inference_params))
    sampling_params = SamplingParams(**inference_params)

    logger.info("======= Loading dataset {} =======".format(json_file))

    inputs = []
    targets = []
    rerank_list = []
    rerank_list = read_jsonl(json_file)
    for data in tqdm(rerank_list):

        # # distill
        # input_text = data['input']
        # target_text = data['output']
        input_text = data['question']
        target_text = data['answer']
        # input_tokens = tokenizer.tokenize(input_text)
        input_text = tokenizer.apply_chat_template([{"role": "user", "content": input_text}], tokenize=False, add_generation_prompt=True)
        input_tokens = tokenizer.tokenize(input_text)
        if len(input_tokens) > max_input_length:
            # 如果输入文本的长度超过了模型的最大序列长度，我们可以选择截断它
            logger.info("*** len(input_token)={}超过了max_input_length={}，进行截断!".format(len(input_tokens), max_input_length))
            input_tokens = input_tokens[-max_input_length:]
        
        # 将截断后的分词序列转换回文本
        input_text = tokenizer.convert_tokens_to_string(input_tokens)
        inputs.append(input_text)
        targets.append(target_text)

    if debug > 0:
        inputs = inputs[:debug]
        targets = targets[:debug]

    logger.info("*** Load {} samples from {}".format(len(inputs), json_file))
    logger.info("*** input example: {} ".format(inputs[0]))
    logger.info("*** target example: {} ".format(targets[0]))


    logger.info("======= Start generate =======")
    t_generate_start = time.time()
    outputs = llm.generate(inputs, sampling_params)
    time_cost = time.time() - t_generate_start
    logger.info("""
        *** Performance stats:
        Num of Inputs: {}
        Num of Outputs:{}
        Time cost: {:.3f} secs
        """.format(len(inputs), len(outputs), time_cost))

    logger.info("======= Start write outputs into file =======")
    logger.info("文件保存至：{}".format(output_path))
    with open(output_path, "w", encoding='utf-8') as f:
        for index, output in tqdm(enumerate(outputs)):
            prompt = output.prompt
            # print(output.outputs)
            generated = output.outputs[0].text
            out = rerank_list[index]
            out['generated'] = generated
            candidate_generated = []
            if rollout > 1:
                for i in range(rollout):
                    candidate_generated.append(output.outputs[i].text)
            out['candidate_generated'] = candidate_generated
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
