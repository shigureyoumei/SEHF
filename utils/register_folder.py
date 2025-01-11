import argparse
from my_utils import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=str, help='original folder')
    parser.add_argument('--fps', type=int, help='Frames per second', default=25)
    args = parser.parse_args()

    folder = args.folder
    fps = args.fps

    align_imgs_and_create_videos(folder, fps)

    print('Creating RGB event video...')
    create_rgb_event_video(folder, fps)
