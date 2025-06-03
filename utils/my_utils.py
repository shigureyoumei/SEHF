import time
import h5py
import os
import cv2
from utils.ets_utils import *
from metavision_core.event_io.raw_reader import RawReader
from metavision_core.event_io import EventsIterator
import numpy as np

# check h5 file's contents
# example: print_h5_contents('Event_Trail_Suppression/data/test_20.0ms/h5/test.h5')
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

# create images from h5 files
# example: create_images_from_h5('Event_Trail_Suppression/data/test_20.0ms/h5', 480, 640)
def create_img_from_h5(root, H, W):
    h5_files_list = []
    for root, dir, files in os.walk(root):
        for file in files:
            if file.endswith(".h5"):
                path = os.path.join(root, file)
                h5_files_list.append(path)
    if len(h5_files_list) == 0:
        print("No h5 files found")
        exit(0)
    else:
        h5_files_list = sorted(h5_files_list, key=lambda x: os.path.basename(x))
        print(f"Found {len(h5_files_list)} h5 files")
        image_folder = os.path.dirname(os.path.dirname(h5_files_list[0])) + "/images"
        if not os.path.exists(image_folder):
            os.makedirs(image_folder)
        for idx, h5_file in enumerate(h5_files_list):
            print(f"Processing {h5_file}")
            with h5py.File(h5_file, 'r') as h5_file:
                t, x, y, p = h5_file['t_denoised'][:], h5_file['x_denoised'][:], h5_file['y_denoised'][:], h5_file['p_denoised'][:]
            img = render(x, y, p, H, W)
            img_name = str(idx).zfill(6) + ".png"
            cv2.imwrite(os.path.join(image_folder, img_name), img)

# create video from images in a folder
# example: create_video_from_images('Event_Trail_Suppression/data/test_20.0ms/images', 30)
def create_videos_from_images(root, fps):
    def is_image_file(filename):
        return filename.lower().endswith(('.png', '.jpg', '.jpeg'))

    def create_video_from_images(image_folder, fps, video_save_path):
        images = [img for img in os.listdir(image_folder) if is_image_file(img)]
        images.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))  # Ensure the images are in the correct order
        # images.sort(key=lambda x:int(x.split('.')[0]))
        print(f"Creating video from {len(images)} images in {image_folder}")
        # print('images after sorted:')
        # print(images)
        if not images:
            return

        frame = cv2.imread(os.path.join(image_folder, images[0]))
        height, width, layers = frame.shape

        video = cv2.VideoWriter(video_save_path, cv2.VideoWriter_fourcc(*'DIVX'), fps, (width, height))

        for image in tqdm(images, desc="Creating video", total=len(images)):
            video.write(cv2.imread(os.path.join(image_folder, image)))

        video.release()
        time.sleep(1)
        print(f"Video saved at {video_save_path}")

    video_folder = root + "/video"
    if not os.path.exists(video_folder):
        os.makedirs(video_folder)
    video_save_path = video_folder + f"/{fps}fps_videos.avi"
    for subdir, dirs, files in os.walk(root):
        if all(is_image_file(file) for file in files):
            create_video_from_images(subdir, fps, video_save_path)


# fetch all triggered event
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

    


# save triggered event frames
def save_frame_trigger(frame_path, h, w, x, y, p, t, trigger):
    if not os.path.exists(frame_path):
        os.makedirs(frame_path)
    frame_id = 0

    t_, x_, y_, p_ = fetch_trigger(t, x, y, p, trigger)

    for _t, _x, _y, _p in zip(t_, x_, y_, p_):
        img = render(_x, _y, _p, h, w)
        frame_name = os.path.join(frame_path, str(frame_id).zfill(6) + '.png')
        cv2.imwrite(frame_name, img)
        frame_id += 1
    print("Over! total frame: " + str(frame_id))

    # total = len(trigger)
    # start = 0
    # end = 1

    # idx_start = 0
    # idx_end = 0
    # while end < total:
    #     while t[idx_start] < trigger[start][1]:
    #         idx_start += 1
    #     while t[idx_end+1] < trigger[end][1]:
    #         idx_end += 1
    #     x_temp = x[idx_start:idx_end]
    #     y_temp = y[idx_start:idx_end]
    #     p_temp = p[idx_start:idx_end]
    #     t_temp = t[idx_start:idx_end]
    #     img = render(x_temp, y_temp, p_temp, h, w)
    #     frame_name = os.path.join(frame_path, str(frame_id).zfill(6) + '.png')
    #     cv2.imwrite(frame_name, img)
    #     frame_id += 1
    #     start += 2
    #     end += 2
    # print("Over! total frame: " + str(frame_id))

# read all raw files from given root path and save triggered event frames
# example: readrawtrigger(path)
def readrawtrigger(path):
    dt = 10000
    for root, dir, files in os.walk(path):
        for file in files:
            if file.endswith(".raw"):
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
                        assert len(x) == len(y) == len(p) == len(t)
                        assert len(triggers) > 0
                        if len(triggers) > 0:
                            print("there are " + str(len(triggers)) + " external trigger events!)")
                            for trigger in triggers:
                                print(trigger)
                                save = trigger.copy()
                                trigger_total.append(save)
                    mv_iterator.reader.clear_ext_trigger_events()

                print("-----------------------------------------------")
                print("Total number of external trigger events: " + str(len(trigger_total)))

                x = np.array(x, dtype='uint16')
                y = np.array(y, dtype='uint16')
                p = np.array(p, dtype='uint16')
                t = np.array(t, dtype='uint64')
                frame_save_folder = os.path.join(os.path.dirname(raw_path), 'frame')
                save_frame_trigger(frame_save_folder, h, w, x, y, p, t, trigger_total)

                # Release memory
                del x, y, p, t, trigger_total
                
# align images from RGB and event cameras
# example: align_images(path)
def align_images(path):
    rgb_points = np.float32([[10, 13], [1905, 15], [1914, 1198], [4, 1199]])
    event_points = np.float32([[101, 115], [536, 120], [535, 396], [96, 390]])
    matrix = cv2.getPerspectiveTransform(rgb_points, event_points)
    #mode = -1   # 0: RGB, 1: event ; current file's style
    for root, dirs, files in os.walk(path):
        for dir_name in dir:
            if dir_name == 'rgb':
                subfolder_path = os.path.join(root, dir_name)
                rgb_aligned_folder = os.path.join(root, 'rgb_aligned')
                if not os.path.exists(rgb_aligned_folder):
                    os.makedirs(rgb_aligned_folder)
                for file in os.listdir(subfolder_path):
                    file_path = os.path.join(subfolder_path, file)
                    rgb_image = cv2.imread(file_path)
                    aligned_rgb_image = cv2.warpPerspective(rgb_image, matrix, (640, 480))
                    rgb_correct = aligned_rgb_image[115:396, 101:536]
                    output_rgb_image_path = os.path.join(rgb_aligned_folder, file)
                    cv2.imwrite(output_rgb_image_path, rgb_correct)
            elif dir_name == 'event':
                subfolder_path = os.path.join(root, dir_name)
                event_aligned_folder = os.path.join(root, 'event_aligned')
                if not os.path.exists(event_aligned_folder):
                    os.makedirs(event_aligned_folder)
                for file in os.listdir(subfolder_path):
                    file_path = os.path.join(subfolder_path, file)
                    event_image = cv2.imread(file_path)
                    event_cut = event_image[115:396, 101:536] # 435, 280
                    event_rect = np.float32([[0,0], [435, 0], [435, 281], [0, 281]])
                    matrix = cv2.getPerspectiveTransform(event_points, event_rect)
                    event_correct = cv2.warpPerspective(event_cut, matrix, (435, 281))
                    output_event_image_path = os.path.join(event_aligned_folder, file)
                    cv2.imwrite(output_event_image_path, event_correct)

# align images from RGB and event cameras and create videos
# example: align_images_and_create_videos(path, fps)
def align_imgs_and_create_videos(path, fps):
    # rgb_points = np.float32([[11, 15], [1904, 16], [1913, 1196], [5, 1198]])  #version1
    # rgb_points = np.float32([[9, 3], [1905, 6], [1914, 1206], [3, 1208]])   #version2
    # rgb_points = np.float32([[8, 2], [1907, 4], [1917, 1208], [2, 1211]])   #version3
    # event_points = np.float32([[5, 0], [440, 5], [439, 281], [0, 275]])  #101,115  536,120  535,396  96,390
    # matrix = cv2.getPerspectiveTransform(rgb_points, event_points)
    for root, dirs, files in os.walk(path):
        for dir_name in dirs:
            if dir_name == 'rgb':
                subfolder_path = os.path.join(root, dir_name)
                rgb_aligned_folder = os.path.join(root, 'rgb_aligned')
                if not os.path.exists(rgb_aligned_folder):
                    os.makedirs(rgb_aligned_folder)
                for file in os.listdir(subfolder_path):
                    file_path = os.path.join(subfolder_path, file)
                    rgb_image = cv2.imread(file_path)
                    # aligned_rgb_image = cv2.warpPerspective(rgb_image, matrix, (441, 282))
                    # aligned_rgb_image = cv2.resize(rgb_image, (442, 276))  #baseline version
                    # aligned_rgb_image = cv2.resize(rgb_image, (440, 275))   #缩小版
                    # aligned_rgb_image = cv2.resize(rgb_image, (445, 278))   #放大版1
                    aligned_rgb_image = cv2.resize(rgb_image, (448, 280))   #放大版2

                    #img_name = file.split('_')[1]   #rgb files' names contain '_'
                    img_name = file
                    output_rgb_image_path = os.path.join(rgb_aligned_folder, img_name)
                    cv2.imwrite(output_rgb_image_path, aligned_rgb_image)
                
                create_videos_from_images(rgb_aligned_folder, fps)
            elif dir_name == 'event_frames_trigger':
                subfolder_path = os.path.join(root, dir_name)
                event_aligned_folder = os.path.join(os.path.dirname(root), 'event_aligned')
                if not os.path.exists(event_aligned_folder):
                    os.makedirs(event_aligned_folder)
                for file in os.listdir(subfolder_path):
                    file_path = os.path.join(subfolder_path, file)
                    event_image = cv2.imread(file_path)
                    # event_correct = event_image[117:393,97:539].copy()  #baseline version
                    # event_correct = event_image[117:392,98:538].copy()  #缩小版
                    # event_correct = event_image[116:394,96:541].copy()  #放大版1
                    event_correct = event_image[115:395,96:544].copy()  #放大版2 平移
                    # event_correct = event_image[115:395,94:542].copy()  #放大版2 旋转
                    # event_correct = event_image[115:395,95:543].copy()  #放大版3 

                    output_event_image_path = os.path.join(event_aligned_folder, file)
                    cv2.imwrite(output_event_image_path, event_correct)
                create_videos_from_images(event_aligned_folder, fps)


# create video of paired rgb and event images
# example: create_rgb_event_video(folder, fps)
def create_rgb_event_video(folder, fps):
    for root, dirs, files in os.walk(folder):
        if 'rgb_aligned' in dirs and 'event_aligned' in dirs:
            rgb_folder = os.path.join(root, 'rgb_aligned')
            event_folder = os.path.join(root, 'event_aligned')
            rgb_files = [f for f in os.listdir(rgb_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
            event_files = [f for f in os.listdir(event_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
            assert len(rgb_files) == len(event_files)
            rgb_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
            event_files.sort()
            paired_folder = os.path.join(root, 'paired')
            if not os.path.exists(paired_folder):
                os.makedirs(paired_folder)
            img_id = 0
            for rgb_file, event_file in zip(rgb_files, event_files):
                rgb_img = cv2.imread(os.path.join(rgb_folder, rgb_file))
                event_img = cv2.imread(os.path.join(event_folder, event_file))
                assert rgb_img.shape == event_img.shape
                # 创建一个布尔掩码，标记 event_img 中不为 (255, 255, 255) 的像素
                mask = np.any(event_img != [255, 255, 255], axis=-1)

                # 使用掩码选择对应的像素值
                paired_img = np.where(mask[..., None], event_img, rgb_img)
                img_name = str(img_id).zfill(6) + '.jpg'
                img_id += 1
                img_path = os.path.join(paired_folder, img_name)
                # 保存合并后的图像
                cv2.imwrite(img_path, paired_img)
                #paired_img = np.concatenate((rgb_img, event_img), axis=1)


            create_videos_from_images(paired_folder, fps)
            print(f"Videos created at {paired_folder}")


# ETS process
def ets(t, x, y, p, s_w, s_h, threshold_t_on, threshold_t_off, soft_thr):
    # ----------------------------Grid the events according to coordinates, with each pixel containing a sequence of timestamp values.----------------------------
    # Create two empty lists with a shape of [H, W].
    ts_map = [[[] for _ in range(s_w)] for _ in range(s_h)]
    p_map = [[[] for _ in range(s_w)] for _ in range(s_h)]

    # Traverse the array events and append t(i) to the list at the corresponding position in X.
    for t_, x_, y_, p_ in zip(t, x, y, p):
        ts_, xs_, ys_, ps_ = t_, x_, y_, p_
        ts_map[ys_][xs_].append(ts_)
        p_map[ys_][xs_].append(ps_)
    ts_map = np.array(ts_map)
    p_map = np.array(p_map)

    # Each element t_array in ts_map represents the timestamps of all events triggered at a pixel point (xx, yy). Convert the two-dimensional matrix into a one-dimensional array.
    ts_map = np.concatenate([np.array(row) for row in ts_map if len(row) > 0])
    p_map = np.concatenate([np.array(row) for row in p_map if len(row) > 0])

    # ----------------------------------------ETS processing----------------------------------------
    ets_events = np.ones((len(t), 4)) * -1
    n_evs = 0

    for ii, t_array in tqdm(enumerate(ts_map), desc="ETS processing", total=len(ts_map)):
        # Skip elements that are empty lists.
        if not t_array:
            continue
        xx = ii % s_w
        yy = int((ii - xx) / s_w)
        t_array = np.array(t_array)
        if len(np.atleast_1d(t_array)) == 1:
            p_array = np.array(p_map[ii])
            ets_events[n_evs] = np.array([t_array, xx, yy, p_array])
            n_evs += 1
        else:
            sort_id = np.argsort(t_array)
            t_array = t_array[sort_id]
            p_array = np.array(p_map[ii])[sort_id]

            for nn in range(len(t_array)):
                if nn == 0:
                    num = 0
                    previous_p = p_array[nn]
                    previous_t = t_array[nn]
                    start_t = previous_t
                    time_interval = 0
                else:
                    if p_array[nn] == 1:
                        threshold_t = threshold_t_on
                    else:
                        threshold_t = threshold_t_off
                    # Events triggered within the same polarity, where the time interval since the last event is greater than the previous interval but less than the threshold value threshold_t.
                    if p_array[nn] == previous_p and t_array[nn] - previous_t > time_interval and t_array[nn] - previous_t < threshold_t:
                        # For events that meet the tailing condition, modify their triggering timestamps to be the time of the previous event triggered at that pixel plus 1 microsecond.
                        # Update iteration parameters.
                        num += 1
                        time_interval = t_array[nn] - previous_t - soft_thr
                        previous_t = t_array[nn]
                        t_array[nn] = start_t + num  # Correct timestamps.
                        # start_t = previous_t
                        previous_p = p_array[nn]
                    else:
                        # If the condition is not met, initialize parameters and start the next iteration
                        num = 0
                        previous_p = p_array[nn]
                        previous_t = t_array[nn]
                        start_t = previous_t
                        time_interval = 0

                ets_events[n_evs] = np.array([t_array[nn], xx, yy, p_array[nn]])
                n_evs += 1

    ets_events = ets_events.reshape(-1, 4)
    ets_events[:, 0] = ets_events[:, 0]
    # Reorder the events processed by ETS based on their timestamps
    idex = np.lexsort([ets_events[:, 0]])
    ets_events = ets_events[idex, :]

    # Release memory
    del ts_map, p_map

    # Return the processed events
    t = ets_events[:, 0]
    x = ets_events[:, 1]
    y = ets_events[:, 2]
    p = ets_events[:, 3]

    return t, x, y, p



# mask events in the given region
def mask_events(_t, _x, _y, _p):
    total = len(_t)
    x = np.array(_x, dtype='uint16')
    y = np.array(_y, dtype='uint16')
    p = np.array(_p, dtype='uint8')
    t = np.array(_t, dtype='uint64')
    mask = (x[0:total]>=96) & (x[0:total]<544) & (y[0:total]>=115) & (y[0:total]<395)
    indices = np.where(mask)[0]
    filter_t = t[indices]
    filter_x = x[indices]
    filter_y = y[indices]
    filter_p = p[indices]

    filter_x = filter_x - 96
    filter_y = filter_y - 115

    return filter_t, filter_x, filter_y, filter_p



# save h5 files
def save_h5(root, path, eh, ew, ox, oy, op, ot, trigger):
# def save_h5(root, path, eh, ew, t, x, p, y, ox, oy, op, ot, trigger): # ets

    save_t, save_x, save_y, save_p = mask_events(ot, ox, oy, op)

    with h5py.File(path, 'w') as f:
        # 保存 h 和 w 为属性
        f.attrs['event_height'] = eh
        f.attrs['event_width'] = ew
        f.attrs['rgb_height'] = 1216
        f.attrs['rgb_width'] = 1936
        f.attrs['paired_height'] = 280
        f.attrs['paired_width'] = 448
        
        
        # create two groups: event and rgb
        event = f.create_group('event')
        rgb = f.create_group('rgb')

        # create two groups under event: original and triggered_event
        original = event.create_group('original')
        
        # ets = event.create_group('ets')

        # 保存 t, x, y, p 为数据集
        original.create_dataset('t', data=save_t)
        original.create_dataset('x', data=save_x)
        original.create_dataset('y', data=save_y)
        original.create_dataset('p', data=save_p)
        
        event.create_dataset('trigger', data=trigger)
       
        for root, dir, files in os.walk(root):
                for dirnames in dir:
                    if dirnames == 'rgb':
                        imgs = []
                        rgb_folder = os.path.join(root, dirnames)
                        rgb_files = [f for f in os.listdir(rgb_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
                        rgb_files.sort()
                        for f in rgb_files:
                            img = cv2.imread(os.path.join(rgb_folder, f))
                            imgs.append(img)
                        imgs_array = np.stack(imgs, axis=0)
                        rgb.create_dataset('rgb_original', data=imgs_array, compression='gzip', compression_opts=9)
                    # elif dirnames == 'event_frames_trigger':
                    #     imgs = []
                    #     data_name = os.path.basename(root)
                    #     event_folder = os.path.join(root, dirnames)
                    #     event_files = [f for f in os.listdir(event_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
                    #     event_files.sort()
                    #     for f in event_files:
                    #         img = cv2.imread(os.path.join(event_folder, f))
                    #         imgs.append(img)
                    #     imgs_array = np.stack(imgs, axis=0)
                    #     event.create_dataset(data_name+'_eventframe', data=imgs_array, compression='gzip', compression_opts=9)
                    # elif dirnames == 'event_aligned':
                    #     imgs = []
                    #     data_name = os.path.basename(root)
                    #     event_folder = os.path.join(root, dirnames)
                    #     event_files = [f for f in os.listdir(event_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
                    #     event_files.sort()
                    #     for f in event_files:
                    #         img = cv2.imread(os.path.join(event_folder, f))
                    #         imgs.append(img)
                    #     imgs_array = np.stack(imgs, axis=0)
                    #     event.create_dataset(data_name+'_event_aligned', data=imgs_array, compression='gzip', compression_opts=9)
                    elif dirnames == 'rgb_aligned':
                        imgs = []
                        rgb_folder = os.path.join(root, dirnames)
                        rgb_files = [f for f in os.listdir(rgb_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
                        rgb_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))  # Sort by the number in the filename

                        print('--------------------------------------')
                        print('after sorted:')
                        print(rgb_files)
                        print('--------------------------------------')
                        for f in rgb_files:
                            img = cv2.imread(os.path.join(rgb_folder, f))
                            imgs.append(img)
                        imgs_array = np.stack(imgs, axis=0)
                        rgb.create_dataset('rgb_aligned', data=imgs_array, compression='gzip', compression_opts=9)


    print()
    print('-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=')
    print(f"Saved h5 file at {path}")
    print('-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=')
    print()