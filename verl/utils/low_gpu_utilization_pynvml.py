import time
import torch
import pynvml
import argparse
import sys
import os

def get_gpu_utilization(handle):
    """使用 NVML 获取瞬时利用率，速度极快"""
    try:
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        return util.gpu
    except pynvml.NVMLError:
        return 0

def adaptive_keeper(gpu_index, threshold=70, check_interval=0.05):
    """
    自适应保活主循环
    :param gpu_index: 物理 GPU ID
    :param threshold: 触发保活的利用率阈值（低于此值开始造假数据）
    :param check_interval: 检查间隔（秒）
    """
    # 1. 初始化 NVML
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
    
    # 2. 初始化 Torch 资源
    # 注意：显存要给的极小，避免挤爆 vLLM 的 KV Cache
    device = torch.device(f'cuda:{gpu_index}')
    # 使用极小的矩阵，只为了触发 GPU 核心调度，不占带宽和显存
    # 512x512 足够让 Tensor Core 动起来，但显存占用忽略不计
    tensor_a = torch.randn(512, 512, device=device, dtype=torch.float16)
    tensor_b = torch.randn(512, 512, device=device, dtype=torch.float16)
    
    print(f"[Keeper-{gpu_index}] Started. Threshold: {threshold}%")
    
    try:
        while True:
            # --- 阶段 1: 监控 ---
            current_util = get_gpu_utilization(handle)
            
            if current_util < threshold:
                # --- 阶段 2: 介入 (GPU 闲置中) ---
                # 只要利用率低，就持续提交小任务
                # 我们使用一个小循环，避免频繁调用 NVML
                start_burst = time.time()
                while time.time() - start_burst < 0.5: # 每次介入最多持续 0.5s
                    torch.mm(tensor_a, tensor_b)
                    
                    # 关键：每提交几次任务就检查一下，如果 vLLM 回来了，立刻撤退
                    # 这里不加 synchronize，利用 CUDA 异步队列特性
                    # 如果 vLLM 提交了任务，利用率会瞬间上去
                    if (time.time() - start_burst) > 0.1:
                        if get_gpu_utilization(handle) > threshold:
                            break 
                
                # 强制同步一次，确保利用率被统计到
                torch.cuda.synchronize()
                
            else:
                # --- 阶段 3: 避让 (GPU 忙碌中) ---
                time.sleep(check_interval)
                
    except KeyboardInterrupt:
        print(f"[Keeper-{gpu_index}] Stopped.")
    finally:
        pynvml.nvmlShutdown()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_number", type=int, default=8)
    args = parser.parse_args()

    # 使用多进程并行监控每一张卡
    from multiprocessing import Process

    procs = []
    for i in range(args.gpu_number):
        p = Process(target=adaptive_keeper, args=(i,))
        p.daemon = True
        p.start()
        procs.append(p)

    # 主进程挂起
    for p in procs:
        p.join()