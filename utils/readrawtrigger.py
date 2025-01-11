from tqdm import tqdm
from edp import *
import os

""" Warning: Before using this code, please install the MetaVision SDK! """
from metavision_core.event_io.raw_reader import RawReader
from metavision_core.event_io import EventsIterator
import numpy as np


def save_frame(frame_path, h, w, x, y, p, t, trigger):
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
                
    

if __name__ == '__main__':

    trigger_total = []
    dt = 10000 # 10ms
    raw_path = '/mnt/e/Program/PROJECT/dataset/DATASETS/ypeople/event/people.raw'
    record_raw = RawReader(raw_path)
    h, w = record_raw.get_size()
    mv_iterator = EventsIterator(input_path=raw_path, delta_t=dt, mode='delta_t')
    mv_iterator1 = EventsIterator(input_path=raw_path, delta_t=dt, mode='delta_t')
    x = []
    y = []
    p = []
    t = []

    for evs in mv_iterator:
        if evs.size != 0:
            triggers = mv_iterator.reader.get_ext_trigger_events()
            x.extend(evs['x'].tolist())
            y.extend(evs['y'].tolist())
            p.extend(evs['p'].tolist())
            t.extend(evs['t'].tolist())
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
    save_frame(frame_save_folder, h, w, x, y, p, t, trigger_total)