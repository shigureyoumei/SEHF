from tqdm import tqdm
import os
from metavision_core.event_io import EventsIterator
import numpy as np
import cv2
from concurrent.futures import ThreadPoolExecutor



#transform a raw file into a list of events(slice by time)
def read_raw_files(path, delta_t:int):
    result_t = []
    result_x = []
    result_y = []
    result_p = []

    eventIter = EventsIterator(input_path=path, mode="delta_t", delta_t=delta_t)
    for events in eventIter:
        result_t.append(events['t'])
        result_x.append(events['x'])
        result_y.append(events['y'])
        result_p.append(events['p'])

    return result_t, result_x, result_y, result_p

#transform a set of events into an image
def render(x: np.ndarray, y: np.ndarray, pol: np.ndarray, H: int, W: int) -> np.ndarray:
    assert x.size == y.size == pol.size
    assert H > 0
    assert W > 0
    img = np.full((H, W, 3), fill_value=255, dtype='int32')
    mask = np.zeros((H, W), dtype='int32')
    pol = pol.astype('int')
    pol[pol == 0] = -1
    mask1 = (x >= 0) & (y >= 0) & (W > x) & (H > y)
    mask[y[mask1], x[mask1]] = pol[mask1]
    img[mask == 0] = [255, 255, 255]
    img[mask == -1] = [255, 0, 0]
    img[mask == 1] = [0, 0, 255]
    img = cv2.flip(img, 1)  # Flip the image horizontally
    return img



#create a video from a set of images and save it in the same folder
#this funcion walks through the root folder and creates a video for each folder containing images
def create_videos_from_images(root, fps):
    def is_image_file(filename):
        return filename.lower().endswith(('.png', '.jpg', '.jpeg'))

    def create_video_from_images(image_folder, fps):
        images = [img for img in os.listdir(image_folder) if is_image_file(img)]
        # images.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))  # Ensure the images are in the correct order
        if '_' in images[0]:
            images.sort(key = lambda x: int(x.split('_')[1].split('.')[0]))
        else:
            images.sort(key = lambda x: int(x.split('.')[0]))
        print(f"Creating video from {len(images)} images in {image_folder}")
        print('images after sorted:')
        print(images)
        if not images:
            return

        frame = cv2.imread(os.path.join(image_folder, images[0]))
        height, width, layers = frame.shape

        video_folder = os.path.join(image_folder, 'video')
        if not os.path.exists(video_folder):
            os.makedirs(video_folder)
        video_save_path = os.path.join(video_folder, f"{fps}fps_video.mp4")

        video = cv2.VideoWriter(video_save_path, cv2.VideoWriter_fourcc(*'DIVX'), fps, (width, height))

        for image in images:
            video.write(cv2.imread(os.path.join(image_folder, image)))

        video.release()

        print(f"Video saved at {video_save_path}")

    for subdir, dirs, files in os.walk(root):
        if all(is_image_file(file) for file in files):
            create_video_from_images(subdir, fps)



#denoise a set of events
def event_denoising(t, x, y, p, H, W):   
    t_ = []
    x_ = []
    y_ = []
    p_ = []

    idx = 0
    idx_start = 0

    all_False = [[False for _ in range(W)] for _ in range(H)]
    wether_checked = all_False
    buffer_ahead = all_False
    buffer_for_next = all_False
    buffer = [[[] for _ in range(W)] for _ in range(H)]

       
    for idx, _ in tqdm(enumerate(range(len(t))), desc="Denoising events", total=len(t)):
        if t[idx] - t[idx_start] > 5000 or idx == len(t) - 1:    #处理当前100ms内的events/最后一部分events
            for y_c in range(H):
                for x_c in range(W):
                    if wether_checked[y_c][x_c]:   #如果该坐标已经被检查过
                        continue
                    save = False

                    for dx in [-1, 0, 1]:  # x方向的偏移
                        for dy in [-1, 0, 1]:  # y方向的偏移
                            nx, ny = x_c + dx, y_c + dy
                            # 确保相邻坐标在范围内
                            if  0 <= ny < H and 0 <= nx < W:
                                # 进行值更改的条件（可以根据需求修改）
                                if (ny != y_c and nx != x_c) and (buffer_ahead[ny][nx] or len(buffer[ny][nx])>0):  #如果相邻坐标未被检查过且有events
                                    if not save:
                                        save = True
                                    if not wether_checked[ny][nx]:
                                        for idx_ in buffer[ny][nx]:
                                            t_.append(t[idx_])
                                            x_.append(nx)
                                            y_.append(ny)
                                            p_.append(p[idx_])
                                        
                                        wether_checked[ny][nx] = True
                    if save:
                        for idx_ in buffer[y_c][x_c]:
                            t_.append(t[idx])
                            x_.append(x_c)
                            y_.append(y_c)
                            p_.append(p[idx_])
                        wether_checked[y_c][x_c] = True
            buffer_ahead = buffer_for_next
            buffer_for_next = all_False
            idx_start = idx
            continue 

        _x, _y = x[idx], y[idx]
        if t[idx] - t[idx_start] > 3000 and t[idx] - t[idx_start] <= 5000:   #将前50ms的events放入buffer_ahead
            if not buffer_for_next[_y][_x]:
                buffer_for_next[_y][_x] = True
        buffer[_y][_x].append(idx)
        
        idx += 1

    # 将四个列表组合在一起
    combined = list(zip(t_, x_, y_, p_))
    
    # 根据 t_ 中的元素从小到大进行排序
    sorted_combined = sorted(combined, key=lambda x: x[0])
    
    # 将排序后的结果解压回四个列表
    t_, x_, y_, p_ = zip(*sorted_combined)
    
    # 将结果转换回列表
    t_ = list(t_)
    x_ = list(x_)
    y_ = list(y_)
    p_ = list(p_)

    del buffer, buffer_ahead, buffer_for_next, wether_checked

    return t_, x_, y_, p_ 



# ets process
def ets_process(events, t0, s_w, s_h, threshold_t_on, threshold_t_off, soft_thr):
    # ----------------------------Grid the events according to coordinates, with each pixel containing a sequence of timestamp values.----------------------------
    # Create two empty lists with a shape of [H, W].
    ts_map = [[[] for _ in range(s_w)] for _ in range(s_h)]
    p_map = [[[] for _ in range(s_w)] for _ in range(s_h)]

    # Traverse the array events and append t(i) to the list at the corresponding position in X.
    for ev in tqdm(events):
        ts_, xs_, ys_, ps_ = ev[0], ev[1], ev[2], ev[3]
        ts_map[ys_][xs_].append(ts_)
        p_map[ys_][xs_].append(ps_)
    ts_map = np.array(ts_map)
    p_map = np.array(p_map)

    # Each element t_array in ts_map represents the timestamps of all events triggered at a pixel point (xx, yy). Convert the two-dimensional matrix into a one-dimensional array.
    ts_map = np.concatenate([np.array(row) for row in ts_map if len(row) > 0])
    p_map = np.concatenate([np.array(row) for row in p_map if len(row) > 0])

    # ----------------------------------------ETS processing----------------------------------------
    ets_events = np.ones((len(events), 4)) * -1
    n_evs = 0

    for ii, t_array in tqdm(enumerate(ts_map)):
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
    ets_events[:, 0] = ets_events[:, 0] + t0
    # Reorder the events processed by ETS based on their timestamps
    idex = np.lexsort([ets_events[:, 0]])
    ets_events = ets_events[idex, :]

    # Release memory
    del ts_map, p_map

    return ets_events
        

