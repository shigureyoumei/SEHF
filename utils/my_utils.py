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
def createe_img_from_h5(root, H, W):
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
        #images.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))  # Ensure the images are in the correct order
        images.sort()
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

    video_folder = root + "/video"
    if not os.path.exists(video_folder):
        os.makedirs(video_folder)
    video_save_path = video_folder + f"/{fps}fps_videos.avi"
    for subdir, dirs, files in tqdm(os.walk(root), desc="Processing folders"):
        if all(is_image_file(file) for file in files):
            create_video_from_images(subdir, fps, video_save_path)

    
# save triggered event frames
def save_frame_trigger(frame_path, h, w, x, y, p, t, trigger):
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
        x_temp = x[idx_start:idx_end]
        y_temp = y[idx_start:idx_end]
        p_temp = p[idx_start:idx_end]
        t_temp = t[idx_start:idx_end]
        # t_temp = np.array(t[mask(t)], dtype='int32')
        # p_temp = np.array(p[mask(t)], dtype='int8')
        # x_temp = np.array(x[mask(t)], dtype='int8')
        # y_temp = np.array(y[mask(t)], dtype='int8')
        img = render(x_temp, y_temp, p_temp, h, w)
        frame_name = os.path.join(frame_path, str(frame_id).zfill(6) + '.png')
        cv2.imwrite(frame_name, img)
        frame_id += 1
        start += 2
        end += 2
    print("Over! total frame: " + str(frame_id))

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
                    event_correct = event_image[115:396, 101:536]
                    output_event_image_path = os.path.join(event_aligned_folder, file)
                    cv2.imwrite(output_event_image_path, event_correct)

# align images from RGB and event cameras and create videos
# example: align_images_and_create_videos(path, fps)
def align_imgs_and_create_videos(path, fps):
    rgb_points = np.float32([[10, 13], [1905, 15], [1914, 1198], [4, 1199]])
    event_points = np.float32([[101, 115], [536, 120], [535, 396], [96, 390]])
    matrix = cv2.getPerspectiveTransform(rgb_points, event_points)
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
                    aligned_rgb_image = cv2.warpPerspective(rgb_image, matrix, (640, 480))
                    rgb_correct = aligned_rgb_image[115:396, 101:536]
                    output_rgb_image_path = os.path.join(rgb_aligned_folder, file)
                    cv2.imwrite(output_rgb_image_path, rgb_correct)
                create_videos_from_images(rgb_aligned_folder, fps)
            elif dir_name == 'event':
                subfolder_path = os.path.join(root, dir_name)
                event_aligned_folder = os.path.join(root, 'event_aligned')
                if not os.path.exists(event_aligned_folder):
                    os.makedirs(event_aligned_folder)
                for file in os.listdir(subfolder_path):
                    file_path = os.path.join(subfolder_path, file)
                    event_image = cv2.imread(file_path)
                    event_correct = event_image[115:396, 101:536]
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
            rgb_files.sort()
            event_files.sort()
            paired_folder = os.path.join(root, 'paired')

            for rgb_file, event_file in zip(rgb_files, event_files):
                rgb_img = cv2.imread(os.path.join(rgb_folder, rgb_file))
                event_img = cv2.imread(os.path.join(event_folder, event_file))
                assert rgb_img.shape == event_img.shape
                # 创建一个布尔掩码，标记 event_img 中不为 (255, 255, 255) 的像素
                mask = np.any(event_img != [255, 255, 255], axis=-1)

                # 使用掩码选择对应的像素值
                paired_img = np.where(mask[..., None], event_img, rgb_img)

                # 保存合并后的图像
                cv2.imwrite(os.path.join(paired_folder, rgb_file), paired_img)
                #paired_img = np.concatenate((rgb_img, event_img), axis=1)

            create_videos_from_images(paired_folder, fps)
            print(f"Videos created at {paired_folder}")