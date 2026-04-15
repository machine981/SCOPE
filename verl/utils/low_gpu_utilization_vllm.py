import time
import torch
import torch.multiprocessing as mp
import requests
import argparse
import sys
import os

# 设置启动方法为 spawn，这是 PyTorch 多进程的最佳实践
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass

def get_vllm_status(url):
    """
    检查 vLLM 是否忙碌
    返回 True 表示忙碌 (有请求在跑或在排队)，False 表示空闲
    """
    try:
        response = requests.get(url, timeout=0.5)
        if response.status_code != 200:
            return True # 无法连接视为忙碌，避免干扰
        
        lines = response.text.split('\n')
        running = 0.0
        waiting = 0.0
        
        for line in lines:
            if line.startswith("#"): continue
            if "vllm:num_requests_running" in line:
                parts = line.split(' ')
                if len(parts) >= 2: running = float(parts[-1])
            if "vllm:num_requests_waiting" in line:
                parts = line.split(' ')
                if len(parts) >= 2: waiting = float(parts[-1])
        
        # 只要有请求在处理或等待，就视为忙碌
        return (running + waiting) > 0
    except Exception:
        return True # 出错默认视为忙碌

def gpu_worker(gpu_id, vllm_busy_event, matrix_size):
    """
    工作进程：负责在指定 GPU 上制造负载
    """
    # 绑定当前进程到指定 GPU
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = torch.device(f"cuda:{gpu_id}")
    
    print(f"[Worker-{gpu_id}] 初始化完成，准备就绪。")
    
    try:
        # 预分配显存
        a = torch.randn(matrix_size, matrix_size, device=device)
        b = torch.randn(matrix_size, matrix_size, device=device)
    except torch.cuda.OutOfMemoryError:
        print(f"[Worker-{gpu_id}] 错误: 显存不足！请调整 vllm 的 gpu-memory-utilization。")
        return

    while True:
        # 检查共享事件：如果 vllm_busy_event 被设置，说明 vLLM 忙碌
        if vllm_busy_event.is_set():
            # vLLM 忙碌中，我们休眠避让
            time.sleep(0.1)
        else:
            # vLLM 空闲，我们全速运算填充利用率
            torch.matmul(a, b)
            torch.cuda.synchronize()
            # 极短休眠，防止死锁并允许响应中断
            time.sleep(0.001)

def main():
    parser = argparse.ArgumentParser(description="vLLM 多卡利用率填充脚本")
    parser.add_argument("--url", type=str, default="http://localhost:8000/metrics", help="vLLM metrics 接口地址")
    parser.add_argument("--size", type=int, default=4096, help="矩阵大小 (越大负载越高)")
    parser.add_argument("--gpus", type=str, default="all", help="指定GPU ID，例如 '0,1,2,3' 或 'all'")
    args = parser.parse_args()

    # 1. 确定要使用的 GPU 列表
    if args.gpus == "all":
        gpu_count = torch.cuda.device_count()
        gpu_ids = list(range(gpu_count))
    else:
        gpu_ids = [int(x) for x in args.gpus.split(',')]
    
    print(f"[Main] 检测到 GPU 列表: {gpu_ids}")
    print(f"[Main] 监控 vLLM 地址: {args.url}")

    # 2. 创建跨进程共享的 Event
    # set() -> 表示忙碌 (停止填充)
    # clear() -> 表示空闲 (开始填充)
    vllm_busy_event = mp.Event()
    
    # 默认先设为忙碌，防止启动瞬间抢占资源
    vllm_busy_event.set()

    # 3. 启动子进程
    processes = []
    for gpu_id in gpu_ids:
        p = mp.Process(target=gpu_worker, args=(gpu_id, vllm_busy_event, args.size))
        p.start()
        processes.append(p)

    print("[Main] 所有 Worker 已启动，开始监控循环...")

    while True:
        # 获取 vLLM 状态
        is_busy = get_vllm_status(args.url)
        
        if is_busy:
            # 如果 vLLM 忙，设置 Event，通知子进程停止
            if not vllm_busy_event.is_set():
                print("[Status] vLLM 忙碌 -> 停止填充")
                vllm_busy_event.set()
        else:
            # 如果 vLLM 闲，清除 Event，通知子进程开始
            if vllm_busy_event.is_set():
                print("[Status] vLLM 空闲 -> 开始填充 (全卡)")
                vllm_busy_event.clear()
        
        # 监控频率
        time.sleep(0.5)



if __name__ == "__main__":
    main()