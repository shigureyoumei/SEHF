from tqdm import tqdm
import os
import cv2
import argparse
from utils import edp


if __name__=="__main__":
    argparser = argparse.ArgumentParser(description='create video from images in a folder')
    argparser.add_argument('--root', type=str, default='data', help='The path of the directory')
    argparser.add_argument('--fps', type=int, default=30, help='The frames per second')

    args = argparser.parse_args()
    root = args.root
    fps = args.fps

    edp.create_videos_from_images(root, fps)