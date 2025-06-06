import subprocess
import time
import webbrowser


def launch_tensorboard(logdir):
    # 启动 TensorBoard 进程
    tensorboard_process = subprocess.Popen(
        # ['/home/d203_3090ti/micromamba/envs/SEHF/bin/tensorboard', '--logdir', logdir],
        ['/home/naran/micromamba/envs/SEHF/bin/tensorboard', '--logdir', logdir],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    # 等待 TensorBoard 启动
    time.sleep(5)  # 等待 5 秒，确保 TensorBoard 完全启动
    
    # 打开浏览器
    webbrowser.open('http://localhost:6006/')


    # 在终端按ctrl+c时，终止 TensorBoard 进程
    
    return tensorboard_process

if __name__ == '__main__':
    # logdir = '~/projects/SEHF/out_dir/2025_5_31_b2_lr0.0001'
    logdir = '~/projects/result/2025_5_31_b2_lr0.0001'
    tensorboard_process = launch_tensorboard(logdir)
    try:
        # 保持主线程运行，直到用户手动终止
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping TensorBoard...")
        tensorboard_process.terminate()
        tensorboard_process.wait()
        print("TensorBoard stopped.")