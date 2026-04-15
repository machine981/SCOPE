import torch
import time
import argparse
import os
import sys
import threading

def run_burst_check(device_idx, tensor_a, tensor_b, flag_path, duration=0.15):
    """
    执行计算负载，但在循环内部增加了对标记文件的实时检查。
    如果主进程删除了标记文件，立即停止保活，把资源还给主进程。
    """
    device = torch.device(f'cuda:{device_idx}')
    
    # 显式指定设备上下文
    with torch.cuda.device(device):
        start_time = time.time()
        
        # 将大的 duration 拆解为微小的循环，以便随时响应
        while time.time() - start_time < duration:
            # 【优化点】: 每次计算前再次检查信号
            # 如果文件被主进程删除了，说明主进程回来了，立即停止骚扰
            if not os.path.exists(flag_path):
                return 

            # 提交计算任务
            torch.mm(tensor_a, tensor_b)
        
        # 强制同步，确保负载真实发生
        torch.cuda.synchronize()

def gpu_monitor_worker(gpu_id, tensor_a, tensor_b):
    """
    单张 GPU 的独立监控线程函数
    """
    flag_path = f"/dev/shm/verl_gpu_idle_{gpu_id}"
    print(f"[Thread-{gpu_id}] Started monitoring {flag_path}")
    
    while True:
        try:
            if os.path.exists(flag_path):
                # 发现信号 -> 执行保活计算
                # 这里只会阻塞当前线程，不会影响其他 GPU 的线程
                run_burst_check(gpu_id, tensor_a, tensor_b, flag_path, duration=0.15)
                
                # 短暂休眠，避免 CPU 空转过高，同时保持高负载
                time.sleep(0.01)
            else:
                # 无信号 -> 忙碌避让 -> 较长休眠节省 CPU
                time.sleep(0.2)
                
        except Exception as e:
            print(f"[Thread-{gpu_id}] Error: {e}")
            time.sleep(1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_number", required=False, default=8, type=int, help="Number of GPUs")
    args = parser.parse_args()

    print(f"Initializing Concurrent GPU Keeper on {args.gpu_number} GPUs...")

    # 1. 资源预初始化 (主线程完成)
    # 预先分配显存，避免在线程中反复申请
    gpu_resources = []
    try:
        for i in range(args.gpu_number):
            device = torch.device(f'cuda:{i}')
            # 保持 2000x2000 的矩阵以产生足够的负载防止被 Kill
            t1 = torch.rand(2000, 2000, dtype=torch.float, device=device)
            t2 = torch.rand(2000, 2000, dtype=torch.float, device=device)
            gpu_resources.append((t1, t2))
            print(f"GPU {i} resource initialized.")
    except Exception as e:
        print(f"Initialization Failed: {e}")
        sys.exit(1)

    print("Starting monitor threads...")

    # 2. 启动多线程监控
    threads = []
    for i in range(args.gpu_number):
        t1, t2 = gpu_resources[i]
        # 创建线程，target 指向监控函数
        t = threading.Thread(target=gpu_monitor_worker, args=(i, t1, t2), daemon=True)
        t.start()
        threads.append(t)

    print("All threads started. Press Ctrl+C to stop.")

    # 3. 主线程挂起
    # 因为子线程是 daemon 线程，主线程必须活着，子线程才会运行
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping keeper...")