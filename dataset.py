from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import h5py
import os
import numpy as np

class pairDateset(Dataset.Dataset):
    def __init__(self, path_list, w, h):
        """
        处理数据的两种做法：
            1：All data load into memory(结构化数据)
            2：定义一个列表，把每个sample路径放到一个列表，标签放到另一个列表，避免数据一次性全部加载
        """
        self.path_list = path_list
        self.w = w
        self.h = h

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
            while t[idx_start] < trigger[t_start]:
                idx_start += 1
            idx_end = idx_start
            while t[idx_end] < trigger[t_end]:
                idx_end += 1
                if idx_end == len(t):
                    break
            triggered_t.append(np.array(t[idx_start:idx_end], dtype='uint16'))
            triggered_x.append(np.array(x[idx_start:idx_end], dtype='uint16'))
            triggered_y.append(np.array(y[idx_start:idx_end], dtype='uint16'))
            triggered_p.append(np.array(p[idx_start:idx_end], dtype='uint16'))
            t_start += 2
            t_end += 2

        return triggered_t, triggered_x, triggered_y, triggered_p
    
    def stack_data(self, t, x, y, p, interval):
        total = len(t)
        map = []
        for i in range(total):
            slice = no.zeros((self.w, self.h, interval), dtype='uint16')
            for j in range(len(t[i])):
                slice[x[i][j], y[i][j], t[i][j]] = p[i][j]
            map.append(slice)
        return map


    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, idx):
        path = self.path_list[idx]
        with h5py.File(path, 'r') as f:
            p = f['event/original/p'][:]
            t = f['event/original/t'][:]
            x = f['event/original/x'][:]
            y = f['event/original/y'][:]

            trigger = f['event/trigger'][:]

            interval = trigger[1] - trigger[0]

            rgb_aligned = f['rgb/1_rgb_aligned'][:]

            t_t, t_x, t_y, t_p = self.fetch_trigger(t, x, y, p, trigger)

            map = self.stack_data(t_t, t_x, t_y, t_p, interval)

        return map, rgb_aligned
    


if __name__ == '__main__':
    path = '/mnt/d/data'
    train_list = []
    test_list = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.h5'):
                no = file.split('.')[0].split('_')[-1]
                if no not in ['5', '10', '15', '20']:
                    train_list.append(os.path.join(root, file))
                else:
                    test_list.append(os.path.join(root, file))

    w = 448
    h = 280

    train_dataset = pairDateset(train_list, w, h)
    test_dataset = pairDateset(test_list, w, h)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)
