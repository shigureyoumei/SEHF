import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.activation_based import functional
from torch.utils.tensorboard import SummaryWriter
import os
import argparse
import time
from tqdm import tqdm

from dataset import *
from torch.utils.data import DataLoader

import model
import model.SEHF

from loss import HybridLoss

import subprocess
import webbrowser

import datetime as dt
from PIL import Image

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torchmetrics.utilities.prints")


def stack_data(t, x, y, p, interval, w, h):
    slice = interval // 5
    stack_on = []
    stack_off = []
    start = 0
    end = slice
    t = t - t[0]
    for i in range(5):
        stack_on.append(torch.zeros((1, slice, w, h), dtype=torch.float16))
        stack_off.append(torch.zeros((1, slice, w, h), dtype=torch.float16))

        mask = (t >= start) & (t < end)
        t_slice = t[mask]
        p_slice = p[mask]

        t_slice = t_slice - t_slice[0]
        
        p_on = torch.where(p_slice == 1)
        p_off = torch.where(p_slice == 0)
        for idx in p_on[0]:
            stack_on[i][0, int(t_slice[idx].item()), int(x[idx].item()), int(y[idx].item())] = 1.0
        for idx in p_off[0]:
            stack_off[i][0, int(t_slice[idx].item()), int(x[idx].item()), int(y[idx].item())] = -1.0
        start += slice
        end += slice

    return stack_on, stack_off


def launch_tensorboard(logdir):
    # 启动 TensorBoard 进程
    tensorboard_process = subprocess.Popen(
        ['tensorboard', '--logdir', logdir],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    # 等待 TensorBoard 启动
    time.sleep(5)  # 等待 5 秒，确保 TensorBoard 完全启动
    
    # 打开浏览器
    webbrowser.open('http://localhost:6006/')


    # 在终端按ctrl+c时，终止 TensorBoard 进程
    
    return tensorboard_process



def main():
    

    parser = argparse.ArgumentParser(description='input project name here')
    # parser.add_argument('--arg', type=int, default=10, help='input arg here')
    # parser.add_argument('--root', type=str, default='~/projects/', help='project root')
    parser.add_argument('--data', type=str, default='data', help='data path')
    parser.add_argument('--out_dir', type=str, default='out_dir', help='input out_dir here')
    parser.add_argument('--amp', action='store_true', help='wether train with amp')
    parser.add_argument('--bs', type=int, default=128, help='input batchsize here')
    parser.add_argument('--epochs', type=int, default=100, help='input epochs here')
    parser.add_argument('--lr', type=float, default=0.01, help='input lr here')
    parser.add_argument('--w', type=int, default=448, help='width of frame')
    parser.add_argument('--h', type=int, default=280, help='height of frame')
    parser.add_argument('--resume', type=str, default='', help='resume training')

    args = parser.parse_args()
    data_dir = args.data
    bs = args.bs
    epochs = args.epochs
    lr = args.lr
    amp = args.amp
    out_dir = args.out_dir
    w = args.w
    h = args.h
    resume = args.resume


    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f'Using device {device}')

    REClipper = model.SEHF.REClipper(output_channels=7)
    print(REClipper)
    print()

    SEHF = model.SEHF.SEHF()
    print(SEHF)
    print()

    LSTM = model.SEHF.LSTM(num_layers=3)
    print(LSTM)
    print()

    REClipper.to(device)
    SEHF.to(device)
    LSTM.to(device)




    # load data
    train_list = []
    test_list = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.h5'):
                no = file.split('.')[0].split('_')[1]
                if no not in ['5', '10', '15', '20']:
                    train_list.append(os.path.join(root, file))
                else:
                    test_list.append(os.path.join(root, file))

    train_dataset = pairDateset(train_list, w, h)
    test_dataset = pairDateset(test_list, w, h)

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)

    scaler = None
    if amp:
        scaler = torch.GradScaler("cuda:0")

    start_epoch = 0
    lowest_test_loss = 1e10
    best_epoch = -1
    optimizer = torch.optim.Adam(list(SEHF.parameters())+list(REClipper.parameters())+list(LSTM.parameters()), lr=lr)
    hybrid_loss = HybridLoss(lambda_mse=1.0, lambda_ssim=0.5, lambda_lpips=0.2)

    # whether to resume training
    if len(resume) > 0:
        checkpoint = torch.load(resume, map_location='cpu') # 先加载模型到cpu上，等需要的时候再加载到gpu
        REClipper.load_state_dict(checkpoint['REClipper'])
        SEHF.load_state_dict(checkpoint['SEHF'])
        LSTM.load_state_dict(checkpoint['LSTM'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        lowest_test_loss = checkpoint['max_test_acc']
        best_epoch = checkpoint['best_epoch']

        

    now = dt.datetime.now()
    year = now.year
    month = now.month
    day = now.day
    out_dir = os.path.join(out_dir, f'{year}_{month}_{day}_b{bs}_lr{lr}')
    if amp:
        out_dir += '_amp'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    writer = SummaryWriter(out_dir, purge_step=start_epoch)
    with open(os.path.join(out_dir, 'args.txt'), 'w', encoding='utf-8') as args_txt:
        args_txt.write(str(args))


    sample_check = os.path.join(out_dir, 'sample_check')
    if not os.path.exists(sample_check):
        os.makedirs(sample_check)
    

    # training part
    for epoch in tqdm(range(start_epoch, epochs), desc='Epoch', total=(epochs-start_epoch)):
        starttime = time.time()
        REClipper.train()
        SEHF.train()
        LSTM.train()
        total_train_loss = 0
        train_pic_idx = 0
        train_patch_iter = 0

        for event_total, rgb_total in tqdm(train_loader, desc='Train dataLoader', total=len(train_loader)):

            train_sample = 0
            train_loss = 0
            avg_train_loss = 0

            rgb_1 = None
            stack_on_1 = None
            stack_off_1 = None
            

            for i in tqdm(range(rgb_total.shape[1]), desc='training frame', total=rgb_total.shape[1]):
                rgb = rgb_total[:, i, :, :, :] # torch[1,448,280,3]
                t = event_total['t'][i][0].to(torch.float32) # torch[5863]
                x = event_total['x'][i][0].to(torch.float32)
                y = event_total['y'][i][0].to(torch.float32)
                p = event_total['p'][i][0].to(torch.float32)
                stack_on, stack_off = stack_data(t, x, y, p, 2015, w, h) # torch[1,448,280,2015]
                rgb = rgb.to(torch.float16)    #1*280*448*3
                rgb = rgb.permute(0, 3, 2, 1)   #1*3*448*280
                stack_on = [s.transpose(0, 1) for s in stack_on] #403*1*448*280
                stack_off = [s.transpose(0, 1) for s in stack_off] #403*1*448*280
                # stack_on = stack_on.to(device)  
                # stack_off = stack_off.to(device)  

                if i == 0:
                    rgb_1 = rgb
                    stack_on_1 = stack_on
                    stack_off_1 = stack_off
                    rgb_1 = rgb_1.to(device)
                    stack_on_1 = [s.to(device) for s in stack_on_1]
                    stack_off_1 = [s.to(device) for s in stack_off_1]
                    lstm_out = None
                    hidden = None
                    continue
                else:
                    optimizer.zero_grad()
                    rgb = rgb.to(device)
                    stack_on = [s.to(device) for s in stack_on]
                    stack_off = [s.to(device) for s in stack_off]

                    if scaler is not None:
                        with torch.autocast("cuda:0"):
                            clipper = REClipper(stack_on_1, stack_off_1, rgb_1, device)  #1*7*448*280
                            out = SEHF(stack_on, stack_off, lstm_out, clipper, device)
                            lstm_out, hidden = LSTM(out, hidden)
                            loss = hybrid_loss(out, rgb)
                        lstm_out = lstm_out.detach()
                        if hidden is not None:
                            hidden = tuple(h.detach() for h in hidden)
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                        train_sample += 1
                        train_loss += loss.item()
                        if i == rgb_total.shape[1] - 1:
                            save_result = out.detach().cpu().squeeze(0).numpy().astype(np.uint8)
                            save_result = save_result.transpose(2, 1, 0)
                            save_result = save_result[:, :, [2, 1, 0]]  # 将 BGR 转换为 RGB
                            sample = Image.fromarray(save_result)
                            sample.save(os.path.join(sample_check, f'{epoch}_{train_pic_idx}_train.png'))
                            train_pic_idx += 1
                        functional.reset_net(REClipper)
                        functional.reset_net(SEHF)
                    else:
                        clipper = REClipper(stack_on_1, stack_off_1, rgb_1)
                        out = SEHF(stack_on, stack_off, lstm_out, clipper)
                        lstm_out, hidden = LSTM(out, hidden)
                        loss = hybrid_loss(out, rgb)
                        lstm_out = lstm_out.detach()
                        if hidden is not None:
                            hidden = tuple(h.detach() for h in hidden)
                        train_sample += 1
                        loss.backward()
                        optimizer.step()
                        train_loss += loss.item()
                        if i == rgb_total.shape[1] - 1:
                            save_result = out.detach().cpu().squeeze(0).numpy().astype(np.uint8)
                            save_result = save_result.transpose(2, 1, 0)
                            save_result = save_result[:, :, [2, 1, 0]]  # 将 BGR 转换为 RGB
                            sample = Image.fromarray(save_result)
                            sample.save(os.path.join(sample_check, f'{epoch}_{train_pic_idx}_train.png'))
                            train_pic_idx += 1
                        functional.reset_net(REClipper)
                        functional.reset_net(SEHF)

                    # del rgb, stack_on, stack_off
                    # torch.cuda.empty_cache()



            train_time = time.time()
            avg_train_loss = train_loss / train_sample
            train_patch_iter += 1
            total_train_loss += avg_train_loss
            # writer.add_scalar('train_loss', avg_train_loss, epoch)

        
        ulti_train_loss = total_train_loss / train_patch_iter
        writer.add_scalar('train_loss', ulti_train_loss, epoch)



        REClipper.eval()
        SEHF.eval()
        LSTM.eval()
        total_test_loss = 0
        test_pic_idx = 0
        test_patch_iter = 0
        with torch.no_grad():
            for rgb_total, event_total in test_loader:
                test_sample = 0
                test_loss = 0
                avg_test_loss = 0

                rgb_1 = None
                stack_on_1 = None
                stack_off_1 = None

                for i in range(rgb_total.shape[1]):
                    rgb = rgb_total[:, i, :, :, :] # torch[1,448,280,3]
                    t = event_total['t'][i][0].to(torch.float32) # torch[5863]
                    x = event_total['x'][i][0].to(torch.float32)
                    y = event_total['y'][i][0].to(torch.float32)
                    p = event_total['p'][i][0].to(torch.float32)
                    stack_on, stack_off = stack_data(t, x, y, p, 2015, w, h) # torch[1,448,280,2015]
                    rgb = rgb.to(torch.float16)    #1*280*448*3
                    rgb = rgb.permute(0, 3, 2, 1)   #1*3*448*280
                    stack_on = [s.transpose(0, 1) for s in stack_on] #403*1*448*280
                    stack_off = [s.transpose(0, 1) for s in stack_off] #403*1*448*280
                    # stack_on = stack_on.to(device)  
                    # stack_off = stack_off.to(device)  
                if i == 0:
                    rgb_1 = rgb
                    stack_on_1 = stack_on
                    stack_off_1 = stack_off
                    rgb_1 = rgb_1.to(device)
                    stack_on_1 = [s.to(device) for s in stack_on_1]
                    stack_off_1 = [s.to(device) for s in stack_off_1]
                    lstm_out = None
                    hidden = None
                    continue
                else:
                    rgb = rgb.to(device)
                    stack_on = [s.to(device) for s in stack_on]
                    stack_off = [s.to(device) for s in stack_off]
                    clipper = REClipper(stack_on, stack_off, rgb)
                    out = SEHF(stack_on, stack_off, lstm_out, clipper)
                    lstm_out, hidden = LSTM(out, hidden)
                    loss = hybrid_loss(out, rgb)
                    test_loss += loss.item()
                    if i == rgb_total.shape[1] - 1:
                            save_result = out.detach().cpu().squeeze(0).numpy().astype(np.uint8)
                            save_result = save_result.transpose(2, 1, 0)
                            save_result = save_result[:, :, [2, 1, 0]]  # 将 BGR 转换为 RGB
                            sample = Image.fromarray(save_result)
                            sample.save(os.path.join(sample_check, f'{epoch}_{test_pic_idx}_test.png'))
                            test_pic_idx += 1
                    functional.reset_net(REClipper)
                    functional.reset_net(SEHF)

                    # del rgb, stack_on, stack_off
                    # torch.cuda.empty_cache()
                test_patch_iter += 1
                avg_test_loss = test_loss / test_sample
                total_test_loss += avg_test_loss


        test_time = time.time()

        ulti_test_loss = total_test_loss / test_patch_iter
        writer.add_scalar('test_loss', ulti_test_loss, epoch)

        save_best = False
        if test_loss < lowest_test_loss:
            lowest_test_loss = test_loss
            best_epoch = epoch
            save_best = True

        checkpoint = {
            'REClipper': REClipper.state_dict(),
            'SEHF': SEHF.state_dict(),
            'LSTM': LSTM.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'lowest_test_loss': test_loss,
            'best_epoch': best_epoch
        }

        if save_best:
            torch.save(checkpoint, os.path.join(out_dir, 'best.pth'))

        torch.save(checkpoint, os.path.join(out_dir, 'last.pth'))

        print(f'epoch = {epoch}, train_loss ={avg_train_loss: .4f}, test_loss ={test_loss: .4f}, epoch_span ={(test_time-starttime)//60}min{(test_time-starttime)%60}s, train_time ={(train_time-starttime)//60}min{(train_time-starttime)%60}s, lowest_test_loss ={lowest_test_loss: .4f}_epoch({best_epoch})')
        print(f'escape time = {(dt.datetime.now() + dt.timedelta(seconds=(time.time() - starttime) * (epochs - epoch))).strftime("%Y-%m-%d %H:%M:%S")}\n')

    # show result in tensorboard
    tensorboard_process = launch_tensorboard(out_dir)
    # 等待用户手动关闭 TensorBoard
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping TensorBoard...")
        tensorboard_process.terminate()
        
if __name__ == '__main__':
    main()


