from tqdm import tqdm
from edp import *
import os
import argparse
""" Warning: Before using this code, please install the MetaVision SDK! """
from metavision_core.event_io.raw_reader import RawReader
from metavision_core.event_io import EventsIterator
import numpy as np


def save_frame_every_trigger(frame_path, h, w, x, y, p, t, trigger):

    print('**********************************************')
    print('Save frames according to every trigger')
    print('**********************************************')

    if not os.path.exists(frame_path):
        os.makedirs(frame_path)
    frame_id = 0
    total = len(trigger)
    start = 0
    end = 1

    idx_start = 0
    idx_end = 0
    while end < total:
        while t[idx_start] < trigger[start][1]:
            idx_start += 1
        while t[idx_end+1] < trigger[end][1]:
            idx_end += 1
        x_temp = np.array(x[idx_start:idx_end], dtype='uint16')
        y_temp = np.array(y[idx_start:idx_end], dtype='uint16')
        p_temp = np.array(p[idx_start:idx_end], dtype='uint16')

        # x_temp = x[idx_start:idx_end]
        # y_temp = y[idx_start:idx_end]
        # p_temp = p[idx_start:idx_end]
        
        img = render(x_temp, y_temp, p_temp, h, w)
        frame_name = os.path.join(frame_path, str(frame_id).zfill(6) + '.png')
        cv2.imwrite(frame_name, img)
        frame_id += 1
        start += 2
        end += 2
        
    print('------------------------------------')
    print("Over! total frame: " + str(frame_id))
    print('------------------------------------')
    del x_temp, y_temp, p_temp


def save_frame_total_trigger(frame_path, h, w, x, y, p, t, trigger, dt):
    print('**********************************************')
    print('Save frames between start and end trigger')
    print('**********************************************')

    if not os.path.exists(frame_path):
        os.makedirs(frame_path)
    frame_id = 0

    assert len(trigger) > 0 
    t_end = trigger[-1][1]
    t_start = trigger[0][1]

    print('**********************************************')
    print(f'The start timestamp of trigger is {t_start}')
    print(f'The end timestamp of trigger is {t_end}')
    print(f'The total time span is {(t_end - t_start)/1e6} s')
    print('**********************************************')

    idx_1_start = 0
    idx_1_end = 1

    while t[idx_1_start] < t_start:
        idx_1_start += 1
    idx_1_end = idx_1_start+1
    while t[idx_1_end+1] < t_end:
        idx_1_end += 1

    x_temp = x[idx_1_start:idx_1_end]
    y_temp = y[idx_1_start:idx_1_end]
    p_temp = p[idx_1_start:idx_1_end]
    t_temp = t[idx_1_start:idx_1_end]

    idx_2_start = idx_1_start
    idx_2_end = idx_2_start + 1

    # while idx_2_end < len(t_temp):
    #     while t[idx_2_end] - t[idx_2_start] < dt:
    #         if idx_2_end == len(t_temp)-1:
    #             break
    #         idx_2_end += 1
    #     x_temp = np.array(x[idx_2_start:idx_2_end], dtype='uint16')
    #     y_temp = np.array(y[idx_2_start:idx_2_end], dtype='uint16')
    #     p_temp = np.array(p[idx_2_start:idx_2_end], dtype='uint16')


    #     img = render(x_temp, y_temp, p_temp, h, w)
    #     frame_name = os.path.join(frame_path, str(frame_id).zfill(6) + '.png')
    #     cv2.imwrite(frame_name, img)
    #     frame_id += 1
    #     idx_2_start = idx_2_end
    #     idx_2_end += 1


    total_frames = (t_temp[-1] - t_temp[0]) // dt
    with tqdm(total=total_frames, desc="Processing frames") as pbar:
        while idx_2_end < len(t_temp):
            while t[idx_2_end] - t[idx_2_start] < dt:
                if idx_2_end == len(t_temp) - 1:
                    break
                idx_2_end += 1
            x_temp = np.array(x[idx_2_start:idx_2_end], dtype='uint16')
            y_temp = np.array(y[idx_2_start:idx_2_end], dtype='uint16')
            p_temp = np.array(p[idx_2_start:idx_2_end], dtype='uint16')

            img = render(x_temp, y_temp, p_temp, h, w)
            frame_name = os.path.join(frame_path, str(frame_id).zfill(6) + '.png')
            cv2.imwrite(frame_name, img)
            frame_id += 1
            idx_2_start = idx_2_end
            idx_2_end += 1

            # 更新进度条
            pbar.update(idx_2_end - idx_2_start)


    print('------------------------------------')
    print("Creating video from images in the folder")
    print('------------------------------------')

    fps = 1000000 // dt
    create_videos_from_images(frame_path, fps)

    print('------------------------------------')
    print("Over! total frame: " + str(frame_id))
    print('------------------------------------')
    del x_temp, y_temp, p_temp, t_temp
                
    

if __name__ == '__main__':

    arg = argparse.ArgumentParser(description='save frame according to external trigger events')
    arg.add_argument('--folder_path', type=str, help='raw files folder path')
    arg.add_argument('--mode1', action='store_true', help='save frames according to every trigger')
    arg.add_argument('--mode2', action='store_true', help='save frames between start and end trigger')
    arg.add_argument('--dt', type=int, default=10000, help='time interval')

    args = arg.parse_args()
    folder_path = args.folder_path
    dt = args.dt

    for root, dirname, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.raw'):
                raw_path = os.path.join(root, file)
                record_raw = RawReader(raw_path)
                h, w = record_raw.get_size()
                mv_iterator = EventsIterator(input_path=raw_path, delta_t=dt, mode='delta_t')

                x = []
                y = []
                p = []
                t = []

                trigger_total = []
                for evs in mv_iterator:
                    if evs.size != 0:
                        triggers = mv_iterator.reader.get_ext_trigger_events()
                        x.extend(evs['x'].tolist())
                        y.extend(evs['y'].tolist())
                        p.extend(evs['p'].tolist())
                        t.extend(evs['t'].tolist())
                        if len(triggers) > 0:
                            print("there are " + str(len(triggers)) + " external trigger events!)")
                            for trigger in triggers:
                                print(trigger)
                                save = trigger.copy()
                                trigger_total.append(save)
                    mv_iterator.reader.clear_ext_trigger_events()
                print("-----------------------------------------------")
                print("Total number of external trigger events: " + str(len(trigger_total)))
                if args.mode1:
                    frame_path = os.path.join(os.path.dirname(raw_path), 'event')
                    save_frame_every_trigger(frame_path, h, w, x, y, p, t, trigger_total)
                if args.mode2:
                    frame_path = os.path.join(os.path.dirname(raw_path), 'event_frames_dt')
                    save_frame_total_trigger(frame_path, h, w, x, y, p, t, trigger_total, dt)

    # trigger_total = []
    # dt = 10000 # 10ms
    # raw_path = '/mnt/e/Program/PROJECT/dataset/DATASETS/ball2/raw/ball2.raw'
    # record_raw = RawReader(raw_path)
    # h, w = record_raw.get_size()
    # mv_iterator = EventsIterator(input_path=raw_path, delta_t=dt, mode='delta_t')
    # #mv_iterator1 = EventsIterator(input_path=raw_path, delta_t=dt, mode='delta_t')
    # x_t = []
    # y_t = []
    # p_t = []
    # t_t = []

    # x = []
    # y = []
    # p = []
    # t = []

    # for evs in mv_iterator:
    #     if evs.size != 0:
    #         triggers = mv_iterator.reader.get_ext_trigger_events()
    #         x_t.extend(evs['x'].tolist())
    #         y_t.extend(evs['y'].tolist())
    #         p_t.extend(evs['p'].tolist())
    #         t_t.extend(evs['t'].tolist())
    #         if len(triggers) > 0:
    #             print("there are " + str(len(triggers)) + " external trigger events!)")
    #             for trigger in triggers:
    #                 print(trigger)
    #                 save = trigger.copy()
    #                 trigger_total.append(save)
    #     mv_iterator.reader.clear_ext_trigger_events()

    # print("-----------------------------------------------")
    # print("Total number of external trigger events: " + str(len(trigger_total)))

    # x = np.array(x, dtype='uint16')
    # y = np.array(y, dtype='uint16')
    # p = np.array(p, dtype='uint16')
    # t = np.array(t, dtype='uint64')
    # frame_save_folder = os.path.join(os.path.dirname(raw_path), 'frame')
    # save_frame(frame_save_folder, h, w, x, y, p, t, trigger_total)