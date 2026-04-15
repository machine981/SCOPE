import os
import contextlib

class GPUIdleGuard:
    """
    上下文管理器：用于标记一段代码为 'GPU 闲时'。
    进入时：创建信号文件，唤醒后台 Hacking 脚本。
    退出时：删除信号文件，让 Hacking 脚本休眠。
    """
    def __init__(self):
        # 自动获取当前进程的本地 Rank
        # 兼容 Ray 和 PyTorch Distributed
        self.rank = int(os.environ.get('LOCAL_RANK', os.environ.get('RANK', 0)))
        
        # 使用 /dev/shm (内存文件系统)
        self.flag_path = f"/dev/shm/verl_gpu_idle_{self.rank}"

    def __enter__(self):
        try:
            with open(self.flag_path, 'w') as f:
                f.write('1')
        except OSError:
            pass
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            if os.path.exists(self.flag_path):
                os.remove(self.flag_path)
        except OSError:
            pass