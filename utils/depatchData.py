import os
import h5py
from tqdm import tqdm
import numpy as np


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
        if trigger[t_end] - trigger[t_start] == 2016:  
            trigger[t_end] -= 1
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
        triggered_p.append(np.array(p[idx_start:idx_end], dtype='uint8'))
        t_start += 2
        t_end += 2

    return triggered_t, triggered_x, triggered_y, triggered_p


if __name__ == '__main__':
    # 读取数据
    root = '/mnt/d/Storage'
    h5_list = []
    for root, dirs, files in os.walk(root):
        for file in files:
            if file.endswith('.h5'):
                h5_list.append(os.path.join(root, file))
    print(f'Total h5 files: {len(h5_list)}')


    idx = 1
    save_dir = '/mnt/d/ball_data'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    
    for h5_file in tqdm(h5_list, desc='Processing h5 files', total=len(h5_list)):

        slice = 25
        start_idx = 0
        end_idx = start_idx + slice

        with h5py.File(h5_file, 'r') as f:
            # 读取事件相机数据
            p = f['event/original/p'][:]
            t = f['event/original/t'][:]
            x = f['event/original/x'][:]
            y = f['event/original/y'][:]

            trigger = f['event/trigger'][:]

            rgb_aligned = f['rgb/rgb_aligned'][:]

            t_t, t_x, t_y, t_p = fetch_trigger(t, x, y, p, trigger)

            assert len(t_t) == len(t_x) == len(t_y) == len(t_p)
            assert len(t_t) == 100


            base_name = h5_file.split('/')[-1].split('.')[0]

            for i in range(4):
                
                assert end_idx <= len(t_t)
                
                file_name = base_name + f'_{idx}.h5'
                idx += 1
                if idx == 5:
                    idx = 1
                save_path = os.path.join(save_dir, file_name)
                t_save = [np.array(t) for t in t_t[start_idx:end_idx]]
                x_save = [np.array(x) for x in t_x[start_idx:end_idx]]
                y_save = [np.array(y) for y in t_y[start_idx:end_idx]]
                p_save = [np.array(p) for p in t_p[start_idx:end_idx]]
                rgb_save = rgb_aligned[start_idx:end_idx]

                start_idx += slice
                end_idx += slice

                with h5py.File(save_path, 'w') as f:
                    dt = h5py.special_dtype(vlen=np.dtype('uint32'))
                    f.create_dataset('t', data=t_save, dtype=dt)
                    dt = h5py.special_dtype(vlen=np.dtype('uint16'))
                    f.create_dataset('x', data=x_save, dtype=dt)
                    f.create_dataset('y', data=y_save, dtype=dt)
                    dt = h5py.special_dtype(vlen=np.dtype('uint8'))
                    f.create_dataset('p', data=p_save, dtype=dt)
                    f.create_dataset('rgb', data=rgb_save)

                print(f'Save h5 file at {save_path}')
    print(f'finished save {idx} files')

            

            
           