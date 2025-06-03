import time
from tqdm import tqdm
import os
import cv2



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

if __name__=="__main__":

    root = '~/projects/dataset/try/result'
    root = os.path.expanduser(root)
    fps = 100

    create_videos_from_images(root, fps)