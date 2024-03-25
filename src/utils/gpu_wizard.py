import subprocess
import GPUtil
from filelock import FileLock
import time
import os
import threading

def find_idle_gpu(memory_required):
    while True:
        GPUs = GPUtil.getGPUs()
        for i, gpu in enumerate(GPUs):
            if gpu.load == 0 and gpu.memoryFree > memory_required:
                print(f"GPU {i} is idle")
                lock = FileLock(f"/var/lock/gpu_lock_{i}")
                try:
                    lock.acquire(timeout=1)
                    return i, lock
                except:
                    print(f"Failed to lock GPU {i}")
                    pass
        time.sleep(1)  # Wait before checking again

def run_task_on_gpu(cmd, memory_required):
    gpu_id, lock = find_idle_gpu(memory_required)

    def task():
        try:
            cmd_full = f"CUDA_VISIBLE_DEVICES={gpu_id} {cmd}"
            process = subprocess.Popen(cmd_full, shell=True)
            print(f"Task started: {cmd_full}")
            process.wait()
        finally:
            lock.release()
            os.remove(lock.lock_file)
            print(f"Task completed: {cmd_full}")

    # Start the task in a separate thread
    thread = threading.Thread(target=task)
    thread.start()

    return thread
