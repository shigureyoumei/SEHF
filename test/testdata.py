import h5py
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import cv2

def render(x: np.ndarray, y: np.ndarray, pol: np.ndarray, H: int, W: int) -> np.ndarray:
    assert x.size == y.size == pol.size
    assert H > 0
    assert W > 0
    img = np.full((H, W, 3), fill_value=255, dtype='uint8')
    mask = np.zeros((H, W), dtype='uint8')
    pol = pol.astype('int')
    pol[pol == 0] = -1
    mask1 = (x >= 0) & (y >= 0) & (W > x) & (H > y)
    mask[y[mask1], x[mask1]] = pol[mask1]
    img[mask == 0] = [255, 255, 255]
    img[mask == -1] = [255, 0, 0]
    img[mask == 1] = [0, 0, 255]
    img = cv2.flip(img, 1)  # Flip the image horizontally
    return img

def downsample_img(img, target_h=140, target_w=224):
    # 支持 (C, H, W) 或 (H, W, C)
    if img.ndim == 3 and img.shape[0] <= 4:  # (C, H, W)
        c, h, w = img.shape
        img_down = np.stack([
            cv2.resize(img[i], (target_w, target_h), interpolation=cv2.INTER_AREA)
            for i in range(c)
        ], axis=0)
        img_down = np.transpose(img_down, (1, 2, 0))  # (H, W, C)
    elif img.ndim == 3 and img.shape[2] <= 4:  # (H, W, C)
        img_down = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)
    else:
        raise ValueError(f"Unsupported img shape: {img.shape}")
    return img_down

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


def newtrigger(trigger):
    assert len(trigger) == 200
    new_trigger = []
    for i in range(0, len(trigger), 2):
        t0 = trigger[i]
        t1 = trigger[i+1]
        new_trigger.append(t0)
        new_trigger.append(t1)
        for _ in range(3):
            t0 = t1 + 7978
            t1 = t0 + 2015
            new_trigger.append(t0)
            new_trigger.append(t1)
    return np.array(new_trigger, dtype='uint32')

def fetch_trigger(t, x, y, p, trigger):
    triggered_t = []
    triggered_x = []
    triggered_y = []
    triggered_p = []

    total = len(trigger)
    t_start = 0
    t_end = 1

    idx_start = 0

    while t_end < total:
        # 处理 trigger 区间
        end_trigger = trigger[t_end]
        if end_trigger - trigger[t_start] >= 2016:
            end_trigger = trigger[t_start] + 2015

        # 找到起始索引
        while idx_start < len(t) and t[idx_start] < trigger[t_start]:
            idx_start += 1
        idx_end = idx_start
        # 找到结束索引
        while idx_end + 1 < len(t) and t[idx_end + 1] < end_trigger:
            idx_end += 1

        # 切片
        triggered_t.append(np.array(t[idx_start:idx_end+1], dtype='uint32'))
        triggered_x.append(np.array(x[idx_start:idx_end+1], dtype='uint16'))
        triggered_y.append(np.array(y[idx_start:idx_end+1], dtype='uint16'))
        triggered_p.append(np.array(p[idx_start:idx_end+1], dtype='int8'))

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
file_path = '~/projects/test/testdata/try_3.h5'
file_path = os.path.expanduser(file_path)
# file_path = '/mnt/d/ball_data_4_ver2/ball1_1_10.h5'


print_h5_contents(file_path)

save_path = '~/projects/test/testtry3'
save_path = os.path.expanduser(save_path)
if not os.path.exists(save_path):
    os.makedirs(save_path)




with h5py.File(file_path, 'r') as f:
    rgb = f['rgb/rgb_aligned'][:]
    p = f['event/original/p'][:]
    t = f['event/original/t'][:]
    x = f['event/original/x'][:]
    y = f['event/original/y'][:]

    trigger = f['event/trigger'][:]


    # fore = 1
    # end = 2
    # inter = 37960
    # while end < len(trigger):
    #     if trigger[end] - trigger[fore] != inter:
    #         print(f"Trigger mismatch at indices {fore} and {end}: {trigger[end]} - {trigger[fore]} = {trigger[end] - trigger[fore]}")
    #     fore += 2
    #     end += 2


    rgb_aligned = f['rgb/rgb_aligned'][:]



new_trigger = newtrigger(trigger)

# idx = 1
# while idx < len(new_trigger):
#     if new_trigger[idx+1]-new_trigger[idx] != 7978:
#         print(f"Trigger mismatch at indices {idx} and {idx+1}: {new_trigger[idx+1]} - {new_trigger[idx]} = {new_trigger[idx+1] - new_trigger[idx]}")
#     idx += 2

t_t, t_x, t_y, t_p = fetch_trigger(t, x, y, p, new_trigger)







assert len(t_t) == len(t_x) == len(t_y) == len(t_p)
assert len(t_t) == 400

num_groups = 25
group_size = 4
event_group_size = 16


file_idx = 1



for i in tqdm(range(num_groups), desc='Processing groups', total=num_groups):
    file_path = os.path.join(save_path, f'{file_idx}.h5')

    start_idx = i * group_size
    end_idx = start_idx + group_size
    e_end_idx = i * event_group_size
    e_next_idx = e_end_idx + event_group_size

    # 防止越界
    if end_idx > rgb_aligned.shape[0] or e_next_idx > len(t_t):
        break

    rgb_frame = rgb_aligned[start_idx:end_idx]
    t_save = [np.array(t) for t in t_t[e_end_idx:e_next_idx]]
    x_save = [np.array(x) for x in t_x[e_end_idx:e_next_idx]]
    y_save = [np.array(y) for y in t_y[e_end_idx:e_next_idx]]
    p_save = [np.array(p) for p in t_p[e_end_idx:e_next_idx]]

    event_frames = []
    event_frames_show = []
    for t_, x_, y_, p_ in zip(t_save, x_save, y_save, p_save):
        event_frames.append(stack_data(t_, x_, y_, p_, 448, 280))
        event_frames_show.append(render(x_, y_, p_, 280, 480))
    # event_frames_show = np.stack(event_frames_show, axis=0)
    event_frames = np.stack(event_frames, axis=0)
    event_frames_show = np.stack(event_frames_show, axis=0)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(event_frames_show[0])
    plt.title('Event Frames Show0')
    plt.subplot(1, 2, 2)
    plt.imshow(event_frames_show[10])
    plt.title('Event Frames Show1')
    plt.show()

    # 后续处理...
    rgb_save = []
    event_save = []
    event_frames_save = []
    for i in range(4):
        rgb_save.append(downsample_img(rgb_frame[i], target_h=140, target_w=224))
        for j in range(4):
            # print(f"current: {i*4+j}")
            event_save.append(downsample_img(event_frames[i*4+j], target_h=140, target_w=224))
            event_frames_save.append(downsample_img(event_frames_show[i*4+j], target_h=140, target_w=224))
    rgb_save = np.stack(rgb_save, axis=0)
    event_frames = np.stack(event_save, axis=0)
    event_frames_save = np.stack(event_frames_save, axis=0)
    with h5py.File(file_path, 'w') as f:
        f.create_dataset('rgb', data=rgb_save, compression='gzip')
        f.create_dataset('event', data=event_frames, compression='gzip')
        f.create_dataset('event_show', data=event_frames_save, compression='gzip')
    
    file_idx += 1

print('over')
