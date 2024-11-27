from utils.edp import *
from metavision_core.event_io.raw_reader import RawReader
import argparse
import h5py



if __name__=="__main__":
    argparser = argparse.ArgumentParser(description='Read a raw file')
    argparser.add_argument('root_path', type=str, default='data', help='The path of the directory')
    argparser.add_argument('--delta_t', type=int, default=10000, help='The time interval')
    argparser.add_argument('--save_video', action='store_true', help='Save the video, if true, set fps parameter')
    argparser.add_argument('--fps', type=int, default=30, help='The frames per second')

    args = argparser.parse_args()

    root = args.root_path
    delta_t = args.delta_t

    event = {}


    for root, dirs, filenames in os.walk(root):
        for file in filenames:
            if file.endswith(".raw"):
                path = os.path.join(root, file)
                reader = RawReader(path)
                H, W = reader.get_size()
                t, x, y, p = read_raw_files(path, delta_t) #[ndarray, ndarray, ndarray, ndarray]
                savepath = os.path.dirname(root)    #savepath is the parent directory of the raw file
                event[file] = (t, x, y, p, H, W, savepath)    #{filename: tuple(t(list), x(list), y(list), p(list), H, W)}

    for key, value in event.items():
        print(f"Key: {key}, Value: {type(value)}")
        t_list, x_list, y_list, p_list, H, W ,savepath = value #t_list is ndarray*1212 

        t_denoised = []
        x_denoised = []
        y_denoised = []
        p_denoised = []
        for t, x, y, p in tqdm(zip(t_list, x_list, y_list, p_list), desc=f"Processing {key}", total=len(t_list)):
            #event_dict = event2dict(t, x, y, p)
            t_, x_, y_, p_ = event_denoising(t, x, y, p, H, W)
            t_denoised.append(t_)
            x_denoised.append(x_)
            y_denoised.append(y_)
            p_denoised.append(p_)
        
        
        h5_idx = 0
        h5_floder = os.path.join(savepath, f"{key}_denoised_h5")
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

        print(f"{key} denoised, {len(t_denoised)} events remained")

        # save image and create video
        if args.save_video:
            image_folder = os.path.join(savepath, "images_denoised")
            if not os.path.exists(image_folder):
                os.makedirs(image_folder)
            img_idx = 0
            for t, x, y, p in zip(t_denoised, x_denoised, y_denoised, p_denoised):
                img = render(x, y, p, H, W)
                img_name = str(img_idx).zfill(6)
                cv2.imwrite(os.path.join(image_folder, f"{img_name}.png"), img)
                img_idx += 1

            create_videos_from_images(image_folder, args.fps)
            
            
                