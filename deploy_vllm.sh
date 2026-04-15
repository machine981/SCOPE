# https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html#
# https://www.cnblogs.com/badwood316/p/18187990
export PATH="./condas/conda0430/bin:$PATH"

export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

which python

model_name_or_path=./Models/Skywork-OR1-7B
served_model_name=Skywork-OR1-7B

NNODES=`python parse_env.py nnodes`
MASTER_ADDR=`python parse_env.py master_addr`
GPUS_PER_NODE=`python parse_env.py nproc_per_node`
NODE_RANK=`python parse_env.py node_rank`
WORLD_SIZE=$(($NNODES*$GPUS_PER_NODE))


tensor_parallel_size=$GPUS_PER_NODE

echo $model_name_or_path
echo $served_model_name

python3 ./verl-distillation-ori/verl/utils/low_gpu_utilization_vllm.py &

python3 -m vllm.entrypoints.openai.api_server \
    --model ${model_name_or_path} \
    --api-key xxx \
    --tensor-parallel-size ${tensor_parallel_size} \
    --served-model-name ${served_model_name} \
    --enable-chunked-prefill \
    --gpu-memory-utilization 0.6 \
    --max_num_batched_tokens 4096 \
    --max-model-len 34000 \
    --max-logprobs 30
    
sleep 10

# # 启动服务
echo "服务已部署，访问地址为：http://$(hostname -I | awk '{print $1}'):8000"