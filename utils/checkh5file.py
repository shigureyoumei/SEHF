import h5py
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np


def stack_data(t, x, y, p, w, h):
    map_on = np.zeros((w, h), dtype=np.float32)
    map_off = np.zeros((w, h), dtype=np.float32)
    assert t.shape == x.shape == y.shape == p.shape
    len = t.shape[0]
    for i in range(len):
        if p[i] > 0:
            map_on[int(x[i]), int(y[i])] += p[i]
        else:
            map_off[int(x[i]), int(y[i])] += 1

    # Normalize map_on and map_off to the range 0-255
    map_on = (map_on - np.min(map_on)) / (np.max(map_on) - np.min(map_on)) * 255
    map_off = (map_off - np.min(map_off)) / (np.max(map_off) - np.min(map_off)) * 255
    map_on = np.expand_dims(map_on, axis=0)
    map_off = np.expand_dims(map_off, axis=0)

    map = np.concatenate((map_on, map_off), axis=0)
    map = np.transpose(map, (2, 1, 0))  # 转换为 (H, W, C)
    map = np.fliplr(map)

    return map

def fetch_trigger(t, x, y, p, trigger):
    triggered_t = []
    triggered_x = []
    triggered_y = []
    triggered_p = []

    total = len(trigger)
    t_start = 0
    t_end = 1

    idx_start = 0
    idx_end = idx_start

    while t_end < total:
        """
        due to the trigger is not exactly interger 2015, the interval between two trigger is either 2015 or 2016
        when the interval is 2016, we mannually decrease the end trigger by 1
        """
        if trigger[t_end] - trigger[t_start] >= 2016:  
            trigger[t_end] = trigger[t_start] + 2015
        while t[idx_start] < trigger[t_start]:
            idx_start += 1
        idx_end = idx_start
        while t[idx_end+1] < trigger[t_end]:
            idx_end += 1
            if idx_end+1 == len(t):
                break
        triggered_t.append(np.array(t[idx_start:idx_end], dtype='uint32'))
        triggered_x.append(np.array(x[idx_start:idx_end], dtype='uint16'))
        triggered_y.append(np.array(y[idx_start:idx_end], dtype='uint16'))
        triggered_p.append(np.array(p[idx_start:idx_end], dtype='int8'))
        t_start += 2
        t_end += 2

    return triggered_t, triggered_x, triggered_y, triggered_p


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
file_path = '/mnt/d/Storage/ball1/1/ball1_1.h5'
# file_path = '/mnt/d/ball_data_4_ver2/ball1_1_10.h5'


print_h5_contents(file_path)



with h5py.File(file_path, 'r') as f:
    rgb = f['rgb/rgb_aligned'][:]
    p = f['event/original/p'][:]
    t = f['event/original/t'][:]
    x = f['event/original/x'][:]
    y = f['event/original/y'][:]

    trigger = f['event/trigger'][:]

    rgb_aligned = f['rgb/rgb_aligned'][:]

    t_t, t_x, t_y, t_p = fetch_trigger(t, x, y, p, trigger)

    assert len(t_t) == len(t_x) == len(t_y) == len(t_p)
    assert len(t_t) == 100

    start_idx = 0
    end_idx = start_idx + 25

    idx = 0
    for i in range(25):
                
        assert end_idx <= len(t_t) 
        idx += 1
        if idx == 26:
            idx = 1
        event_frames = []
        t_save = [np.array(t) for t in t_t[start_idx:end_idx]]
        x_save = [np.array(x) for x in t_x[start_idx:end_idx]]
        y_save = [np.array(y) for y in t_y[start_idx:end_idx]]
        p_save = [np.array(p) for p in t_p[start_idx:end_idx]]
        rgb_frame = rgb_aligned[start_idx:end_idx]

        for t_, x_, y_, p_ in zip(t_save, x_save, y_save, p_save):
            event_frames.append(stack_data(t_, x_, y_, p_, 448, 280))
        event_frames = np.stack(event_frames, axis=0)

        # rgb_save = []
        # event_save = []
        # for i in range(slice):
        #     rgb_save.append(downsample_img(rgb_frame[i], target_h=140, target_w=224))
        #     event_save.append(downsample_img(event_frames[i], target_h=140, target_w=224))
        # rgb_save = np.stack(rgb_save, axis=0)
        # event_frames = np.stack(event_save, axis=0)

        # show img
        

        fig, axs = plt.subplots(1, 5, figsize=(20, 4))
        for i in range(5):
            axs[i].imshow(rgb_frame[i])
            axs[i].set_title(f'Frame {i}')
            axs[i].axis('off')
        plt.tight_layout()
        plt.show()




        # count = 0
        # for i in range(25):
        #     # img = rgb_save[i]
        #     img = rgb_frame[i]
        #     event_i = event_frames[i]
        #     event_img = np.zeros((280, 448), dtype=np.int8)
        #     event_img = event_img + event_i[:, :, 0] - event_i[:, :, 1]
        #     mask_on = event_img > 0
        #     mask_off = event_img < 0
        #     img_masked = img.copy()
        #     img_masked[mask_on] = [255, 0, 0]   # 红色
        #     img_masked[mask_off] = [0, 0, 255]  # 蓝色
            
        #     fig, axs = plt.subplots(1, 3, figsize=(15, 5))

        #     axs[0].imshow(img)
        #     axs[0].set_title('Original img')
        #     axs[0].axis('off')

        #     axs[1].imshow(event_img, cmap='gray')
        #     axs[1].set_title('Event img')
        #     axs[1].axis('off')

        #     axs[2].imshow(img_masked)
        #     axs[2].set_title('Masked img')
        #     axs[2].axis('off')

        #     plt.tight_layout()
        #     plt.show()

        #     count += 1
        # start_idx += slice
        # end_idx += slice










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




# with h5py.File(file_path, 'r') as f:
#     # 读取数据集
#     event_frames = f['event_frames'][:]
#     import numpy as np
#     frame_on = np.array(event_frames[0][0]).transpose(1, 0)
#     frame_off = np.array(event_frames[0][1]).transpose(1, 0)

#     # 显示frame_on灰度图
#     plt.imshow(frame_on, cmap='gray')
#     plt.title('Frame On')
#     plt.colorbar()
#     plt.show()

#     # 保存frame_on灰度图到本地
#     plt.imsave('out_dir/frame_on.png', frame_on, cmap='gray')

#     # 显示frame_off灰度图
#     plt.imshow(frame_off, cmap='gray')
#     plt.title('Frame Off')
#     plt.colorbar()
#     plt.show()

#     # 保存frame_off灰度图到本地
#     plt.imsave('out_dir/frame_off.png', frame_off, cmap='gray')
