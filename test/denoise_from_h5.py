from utils.edp import *
from metavision_core.event_io.raw_reader import RawReader
import argparse
import h5py
from utils import ets_utils

def print_h5_structure(file_path):
    def print_attrs(name, obj):
        print(f"{name}: {obj}")
        for key, val in obj.attrs.items():
            print(f"    {key}: {val}")

    with h5py.File(file_path, 'r') as f:
        f.visititems(print_attrs)


def read_and_append_events(file_path, interval=10000):
    with h5py.File(file_path, 'r') as f:
        ts = f['/events/ts'][:]
        xs = f['/events/xs'][:]
        ys = f['/events/ys'][:]
        ps = f['/events/ps'][:]

    t = []
    x = []
    y = []
    p = []

    start_idx = 0
    while start_idx < len(ts):
        end_idx = start_idx
        while end_idx < len(ts) and ts[end_idx] - ts[start_idx] < interval:
            end_idx += 1

        t.append(ts[start_idx:end_idx])
        x.append(xs[start_idx:end_idx])
        y.append(ys[start_idx:end_idx])
        p.append(ps[start_idx:end_idx])

        start_idx = end_idx

    return t, x, y, p


if __name__=="__main__":
    argparser = argparse.ArgumentParser(description='denoise from h5 files')
    argparser.add_argument('--root', type=str, default='data', help='The path of the directory')
    argparser.add_argument('--save_video', action='store_true', help='Save the video, if true, set fps parameter')
    argparser.add_argument('--fps', type=int, default=30, help='The frames per second')
    argparser.add_argument('--w', type=int, default=640, help='Sensor width')
    argparser.add_argument('--h', type=int, default=480, help='Sensor Height')

    args = argparser.parse_args()

    H = args.h
    W = args.w

    files_list = []
    for root, dir, files in os.walk(args.root):
        for file in files:
            if file.endswith(".h5"):
                path = os.path.join(root, file)
                files_list.append(path)
    
    for file in files_list:
        print(f"Processing {file}")
        print_h5_structure(file)
        t_list, x_list, y_list, p_list = read_and_append_events(file)

        t_denoised = []
        x_denoised = []
        y_denoised = []
        p_denoised = []
        for t, x, y, p in tqdm(zip(t_list, x_list, y_list, p_list), desc=f"Processing {file}", total=len(t_list)):
            #event_dict = event2dict(t, x, y, p)
            t_, x_, y_, p_ = event_denoising(t, x, y, p, H, W)
            t_denoised.append(t_)
            x_denoised.append(x_)
            y_denoised.append(y_)
            p_denoised.append(p_)
        
        
        h5_idx = 0
        h5_floder = os.path.join(os.path.dirname(file), "denoised_h5")
        if not os.path.exists(h5_floder):
            os.makedirs(h5_floder)
        # 保存为 h5 文件
        for t, x, y, p in zip(t_denoised, x_denoised, y_denoised, p_denoised):
            t_denoised_h5 = np.array(t)
            x_denoised_h5 = np.array(x)
            y_denoised_h5 = np.array(y)
            p_denoised_h5 = np.array(p)

            h5_name = str(h5_idx).zfill(6)
            h5_filename = os.path.join(h5_floder, f"{h5_name}.h5")
            with h5py.File(h5_filename, 'w') as h5f:
                h5f.create_dataset('t_denoised', data=t_denoised_h5)
                h5f.create_dataset('x_denoised', data=x_denoised_h5)
                h5f.create_dataset('y_denoised', data=y_denoised_h5)
                h5f.create_dataset('p_denoised', data=p_denoised_h5)
            h5_idx += 1

        # save image and create video
        
        image_folder = os.path.join(os.path.dirname(file), "images_denoised")
        if not os.path.exists(image_folder):
            os.makedirs(image_folder)
        img_idx = 0
        for t, x, y, p in zip(t_denoised, x_denoised, y_denoised, p_denoised):
            t_nd = np.array(t).astype(np.int64)
            x_nd = np.array(x).astype(np.uint16)
            y_nd = np.array(y).astype(np.uint16)
            p_nd = np.array(p).astype(np.uint8)

            img = render(x_nd, y_nd, p_nd, H, W)
            img_name = str(img_idx).zfill(6)
            cv2.imwrite(os.path.join(image_folder, f"{img_name}.png"), img)
            img_idx += 1

        create_videos_from_images(image_folder, args.fps)
