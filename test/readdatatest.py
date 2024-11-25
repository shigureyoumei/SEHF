from utils.edp import *
from metavision_core.event_io.raw_reader import RawReader
import argparse




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
                event[file] = (t, x, y, p, H, W)    #{filename: tuple(t(list), x(list), y(list), p(list), H, W)}

    for key, value in event.items():
        print(f"Key: {key}, Value: {type(value)}")
        t_list, x_list, y_list, p_list, H, W = value #t_list is ndarray*1212 

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
        
        print(f"{key} denoised, {len(t_denoised)} events remained")
    
                