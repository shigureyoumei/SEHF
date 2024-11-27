import os
from tqdm import tqdm
import cv2
import argparse
from utils.ets_utils import *
from utils.edp import *
import numpy as np

if __name__=="__main__":

    argparser = argparse.ArgumentParser(description='create video images from h5 files')
    argparser.add_argument('--root', type=str, default='data', help='The path of the directory')
    argparser.add_argument('--h', type=int, default=480, help='image height')
    argparser.add_argument('--w', type=int, default=640, help='image width')
    
    args = argparser.parse_args()
    root = args.root
    H = args.h
    W = args.w

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