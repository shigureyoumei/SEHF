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
    img = np.full((H, W, 3), fill_value=255, dtype='uint8')
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

    def create_video_from_images(image_folder, fps, video_save_path):
        images = [img for img in os.listdir(image_folder) if is_image_file(img)]
        # images.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))  # Ensure the images are in the correct order
        print(f"Creating video from {len(images)} images in {image_folder}")
        print('images after sorted:')
        print(images)
        if not images:
            return

        frame = cv2.imread(os.path.join(image_folder, images[0]))
        height, width, layers = frame.shape

        video = cv2.VideoWriter(video_save_path, cv2.VideoWriter_fourcc(*'DIVX'), fps, (width, height))

        for image in images:
            video.write(cv2.imread(os.path.join(image_folder, image)))

        video.release()

        print(f"Video saved at {video_save_path}")

    video_save_path = os.path.dirname(root) + f"/{fps}fps_videos.avi"
    for subdir, dirs, files in tqdm(os.walk(root), desc="Processing folders"):
        if all(is_image_file(file) for file in files):
            create_video_from_images(subdir, fps, video_save_path)

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

#---check if adjacent events exist
# def check_adjacent(buffer_ahead, buffer, save_flag_process, event):
#     for y in range(len(buffer)):
#         for x in range(len(buffer[y])):
#             if buffer_ahead[y][x]:
#                 for idx in buffer[y][x]:
                    

#     return 

#---transform t,x,y,p into a dictionary
# return a dictionary event {t:[[x],[y],[p],[flag]]} and a list of time [t1, t2, t3, ...]
# def event2dict(t, x, y, p):
#     event={}
#     for idx, t_ in enumerate(t):
#         if t_ not in event:
#             event[t_] = [[],[],[]]
#         event[t_][0].append(x[idx])
#         event[t_][1].append(y[idx])
#         event[t_][2].append(p[idx])
       
        
#     return event
        

