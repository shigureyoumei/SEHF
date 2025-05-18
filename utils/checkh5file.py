import h5py
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

def print_h5_contents(file_path):
    def print_attrs(name, obj):
        print(f"{name}:")
        for key, val in obj.attrs.items():
            print(f"    {key}: {val}")
        if isinstance(obj, h5py.Group):
            print("    (Group)")
        elif isinstance(obj, h5py.Dataset):
            print(f"    (Dataset) - shape: {obj.shape}, dtype: {obj.dtype}")

    with h5py.File(file_path, 'r') as f:
        f.visititems(print_attrs)

# 使用示例
# file_path = '/mnt/e/Program/PROJECT/dataset/DATASETS/try6/try6.h5'
# file_path = '/mnt/d/Storage/ball1/1/1.h5'
file_path = '/mnt/d/ball_data_4_ver2/ball1_1_10.h5'
# print_h5_contents(file_path)
# file_list = []
# max_length = 0
# min_length = 100000000
# max_path  = ''
# min_path = ''

# for root, dirs, files in os.walk(file_path):
#     for file in files:
#         if file.endswith('.h5'):
#             file_list.append(os.path.join(root, file))


# for path in tqdm(file_list, desc="Processing files", total=len(file_list)):
#     with h5py.File(path, 'r') as f:
#         t = f['t']
#         for t_ in t:
#             if len(t_) > max_length:
#                 max_length = len(t_)
#                 max_path = path
#             if len(t_) < min_length:
#                 min_length = len(t_)
#                 min_path = path
# print(f"min_length: {min_length}, min_path: {min_path}")
# print(f"max_length: {max_length}, max_path: {max_path}")

with h5py.File(file_path, 'r') as f:
    # 读取数据集
    event_frames = f['event_frames'][:]
    import numpy as np
    frame_on = np.array(event_frames[0][0]).transpose(1, 0)
    frame_off = np.array(event_frames[0][1]).transpose(1, 0)

    # 显示frame_on灰度图
    plt.imshow(frame_on, cmap='gray')
    plt.title('Frame On')
    plt.colorbar()
    plt.show()

    # 保存frame_on灰度图到本地
    plt.imsave('out_dir/frame_on.png', frame_on, cmap='gray')

    # 显示frame_off灰度图
    plt.imshow(frame_off, cmap='gray')
    plt.title('Frame Off')
    plt.colorbar()
    plt.show()

    # 保存frame_off灰度图到本地
    plt.imsave('out_dir/frame_off.png', frame_off, cmap='gray')
