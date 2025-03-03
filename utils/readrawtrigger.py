from tqdm import tqdm
from edp import *
import os
import argparse
""" Warning: Before using this code, please install the MetaVision SDK! """
from metavision_core.event_io.raw_reader import RawReader
from metavision_core.event_io import EventsIterator
import numpy as np
from my_utils import *


def save_frame_every_trigger(frame_path, h, w, x, y, p, t, trigger):

    print('**********************************************')
    print('Save frames according to every trigger')
    print('**********************************************')

    if not os.path.exists(frame_path):
        os.makedirs(frame_path)
    frame_id = 1

    t_, x_, y_, p_ = fetch_trigger(t, x, y, p, trigger)


    for t_temp, x_temp, y_temp, p_temp in zip(t_, x_, y_, p_):
        img = render(x_temp, y_temp, p_temp, h, w)
        frame_name = os.path.join(frame_path, str(frame_id).zfill(6) + '.png')
        cv2.imwrite(frame_name, img)
        frame_id += 1

    # total = len(trigger)
    # start = 0
    # end = 1

    # idx_start = 0
    # while end < total:
    #     while t[idx_start] < trigger[start]:
    #         idx_start += 1
    #     idx_end = idx_start + 1
    #     while t[idx_end] < trigger[end]:
    #         idx_end += 1
    #         if idx_end == len(t) - 1:
    #             break
    #     x_temp = np.array(x[idx_start:idx_end], dtype='uint16')
    #     y_temp = np.array(y[idx_start:idx_end], dtype='uint16')
    #     p_temp = np.array(p[idx_start:idx_end], dtype='uint16')

    #     # x_temp = x[idx_start:idx_end]
    #     # y_temp = y[idx_start:idx_end]
    #     # p_temp = p[idx_start:idx_end]
        
    #     img = render(x_temp, y_temp, p_temp, h, w)
    #     frame_name = os.path.join(frame_path, str(frame_id).zfill(6) + '.png')
    #     cv2.imwrite(frame_name, img)
    #     frame_id += 1
    #     start += 2
    #     end += 2
        
    print('------------------------------------')
    print("Over! total frame: " + str(frame_id-1))
    print('------------------------------------')
    del x_temp, y_temp, p_temp


def save_frame_total_trigger(frame_path, h, w, x, y, p, t, trigger, dt):
    print('**********************************************')
    print('Save frames between start and end trigger')
    print('**********************************************')

    if not os.path.exists(frame_path):
        os.makedirs(frame_path)
    frame_id = 1

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
    print("Over! total frame: " + str(frame_id-1))
    print('------------------------------------')
    del x_temp, y_temp, p_temp, t_temp
                
    

if __name__ == '__main__':

    arg = argparse.ArgumentParser(description='save frame according to external trigger events')
    arg.add_argument('--folder_path', type=str, help='raw files folder path')
    arg.add_argument('--ets', action='store_true', help='event trail suppression process')
    arg.add_argument('--mode1', action='store_true', help='save frames according to every trigger')
    arg.add_argument('--mode2', action='store_true', help='save frames between start and end trigger')
    arg.add_argument('--dt', type=int, default=10000, help='time interval')
    arg.add_argument('--video', action='store_true', help='create video from images')
    arg.add_argument('--fps', type=int, default=25, help='frames per second')
    arg.add_argument('--registration', action='store_true', help='register the folder')
    arg.add_argument('--saveh5', action='store_true', help='save triggerred events to h5 file')
    
    args = arg.parse_args()
    folder_path = args.folder_path
    dt = args.dt
    fps = args.fps


    raw_files_list = []

    for root, dirname, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.raw'):
                raw_file_path = os.path.join(root, file)
                raw_files_list.append(raw_file_path)

    print('**********************************************')
    print('Total raw files number: ' + str(len(raw_files_list)))
    print('**********************************************')

    for raw_path in tqdm(raw_files_list, desc='Processing raw files', total=len(raw_files_list)):
        print('**********************************************')
        print('Processing file: ' + raw_path)
        print('**********************************************')
        print()
        
        root = os.path.dirname(os.path.dirname(raw_path))

        # Wether this raw file has been processed
        processed_flag = False
        for root_, dir_, file_ in os.walk(root):
            if 'event_frames_trigger' in dir_:
                processed_flag = True
                break
            if 'event_frames_dt' in dir_:
                processed_flag = True
                break
            if 'event_aligned' in dir_:
                processed_flag = True
                break
            if 'egb_aligned' in dir_:
                processed_flag = True
                break 
                
        if processed_flag:
            print()
            print('-------The raw file has been processed, skip it...-------')
            print()
            continue


        section = os.path.basename(root) # 1, 2, 3, ..., 20
        ball_No = os.path.basename(os.path.dirname(root)) # ball1, ball2, ball3, ..., ball9
        h5_file_name = ball_No + '_' + section + '.h5'
        h5_file_name = os.path.join(root, h5_file_name)
        record_raw = RawReader(raw_path)
        eh, ew = record_raw.get_size()
        mv_iterator = EventsIterator(input_path=raw_path, delta_t=dt, mode='delta_t')

        file_name = raw_path.split('.')[0]

        x_ = []
        y_ = []
        p_ = []
        t_ = []

        trigger_total = []
        for evs in mv_iterator:
            if evs.size != 0:
                triggers = mv_iterator.reader.get_ext_trigger_events()
                x_.extend(evs['x'].tolist())
                y_.extend(evs['y'].tolist())
                p_.extend(evs['p'].tolist())
                t_.extend(evs['t'].tolist())
                if len(triggers) > 0:
                    # print("there are " + str(len(triggers)) + " external trigger events!)")
                    for trigger in triggers:
                        # print(trigger)
                        save = trigger.copy()
                        trigger_total.append(save[1])
            mv_iterator.reader.clear_ext_trigger_events()
        print("-----------------------------------------------")
        print("Total number of external trigger events: " + str(len(trigger_total)))


        # save events between first and last triggers
        mask = np.where((t_ >= trigger_total[0]) & (t_ <= trigger_total[-1]))[0]
        x_ = np.array(x_)[mask].tolist()
        y_ = np.array(y_)[mask].tolist()
        p_ = np.array(p_)[mask].tolist()
        t_ = np.array(t_)[mask].tolist()

        # save original events
        ox = x_
        oy = y_
        op = p_
        ot = t_

        if args.ets:
            s_h = 480
            s_w = 640
            t_on = 1e6
            t_off = 1e6
            soft_t = 0

            print('')
            print('**********************************************')
            print('Event Trail Suppression Process')
            print('**********************************************')
            t_, x_, y_, p_ = ets(t_, x_, y_, p_, s_w, s_h, t_on, t_off, soft_t)

        

        if args.mode1:
            frame_path = os.path.join(os.path.dirname(raw_path), 'event_frames_trigger')
            save_frame_every_trigger(frame_path, eh, ew, x_, y_, p_, t_, trigger_total)
            if args.video:
                print('------------------------------------')
                print("Creating video from images in the folder")
                print('------------------------------------')
                create_videos_from_images(frame_path, args.fps)
        if args.mode2:
            frame_path = os.path.join(os.path.dirname(raw_path), 'event_frames_dt')
            save_frame_total_trigger(frame_path, eh, ew, x_, y_, p_, t_, trigger_total, dt)
            if args.video:
                print('------------------------------------')
                print("Creating video from images in the folder")
                print('------------------------------------')
                create_videos_from_images(frame_path, 1000000//dt)

        if args.registration:
            align_imgs_and_create_videos(root, fps)
            print('Creating RGB event video...')
            create_rgb_event_video(root, fps)


        if args.saveh5:
            # save_h5(root, h5_file_name, eh, ew, x_, y_, p_, t_, ox, oy, op, ot, trigger_total)
            print('**********************************************')
            print('Save events to h5 file')
            print('**********************************************')
            save_h5(root, h5_file_name, eh, ew, x_, y_, p_, t_, trigger_total)
            print('Done!')
            print()
        
