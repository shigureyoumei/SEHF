import gc
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import h5py
import os
import numpy as np
import torch

class pairDateset(Dataset):
    def __init__(self, path_list, w, h):
        """
        处理数据的两种做法：
            1：All data load into memory(结构化数据)
            2：定义一个列表，把每个sample路径放到一个列表，标签放到另一个列表，避免数据一次性全部加载
        """
        self.path_list = path_list
        self.w = w
        self.h = h
    
    # def trans2ndarray(self, x, type):
    #     out = []
    #     for element in x:
    #         out.append(np.array(element, dtype=type))
    #     return out


    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, idx):
        path = self.path_list[idx]
        with h5py.File(path, 'r') as f:

            # p = f['p'][:]
            # t = f['t'][:]
            # x = f['x'][:]
            # y = f['y'][:]
            rgb = f['rgb'][:]

            # 确保数据是均匀的
            
            # t = self.trans2ndarray(t, 'uint32')
            # x = self.trans2ndarray(x, 'uint16')
            # y = self.trans2ndarray(y, 'uint16')
            # p = self.trans2ndarray(p, 'int8')

            # map = self.stack_data(t, x, y, p, 2015)
            # event = np.stack(map, axis=0)

            # event = np.stack([np.array(t), np.array(x), np.array(y), np.array(p)], axis=0)
            # event = {'t': t, 'x': x, 'y': y, 'p': p}

            event_frames = f['event_frames'][:]

        return event_frames, rgb   #event: { [ tensor[1,x] ] }
    


if __name__ == '__main__':
    path = '/mnt/d/ball_data_4_ver2'
    train_list = []
    test_list = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.h5'):
                no = file.split('.')[0].split('_')[1]
                if no not in ['5', '10', '15', '20']:
                    train_list.append(os.path.join(root, file))
                else:
                    test_list.append(os.path.join(root, file))

    w = 448
    h = 280

    train_dataset = pairDateset(train_list, w, h)
    test_dataset = pairDateset(test_list, w, h)

    print(f'Train dataset: \n {train_dataset} \n')
    print(f'length of train dataset: {len(train_dataset)} \n')
    # print(train_dataset.__getitem__(0))
    # print(train_dataset[0])

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=False, num_workers=0, pin_memory=False)
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, num_workers=0)

    for i, (event_frames, rgb) in enumerate(train_loader):
        print(f'iter: {i}')
        print(f'item: {event_frames.shape, rgb.shape}')
        print('-')
