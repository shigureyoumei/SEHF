import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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

alpha = 1e-1

def stack_data(t, x, y, p, w, h):
    map_on = torch.zeros((w, h), dtype=torch.float32)
    map_off = torch.zeros((w, h), dtype=torch.float32)
    assert t.shape == x.shape == y.shape == p.shape
    len = t.shape[0]
    for i in range(len):
        if p[i] > 0:
            map_on[int(x[i]), int(y[i])] += p[i]
        else:
            map_off[int(x[i]), int(y[i])] += p[i]*(-1.0)
    map_on = map_on.unsqueeze(0)
    map_off = map_off.unsqueeze(0)

    

    map = torch.cat((map_on, map_off), dim=0)

    return map


def launch_tensorboard(logdir):
    # 启动 TensorBoard 进程
    tensorboard_process = subprocess.Popen(
        ['/home/d203/micromamba/envs/SEHF/bin/tensorboard', '--logdir', logdir],
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

    

    SEHF = model.SEHF.SEHF()
    print(SEHF)
    print()

    SEHF.to(device)



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

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0, pin_memory=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=0, pin_memory=False)

    scaler = None
    if amp:
        scaler = torch.GradScaler("cuda:0")

    start_epoch = 0
    lowest_test_loss = 1e10
    best_epoch = -1
    optimizer = torch.optim.Adam(SEHF.parameters(), lr=lr)
    hybrid_loss = HybridLoss(lambda_mse=1.0, lambda_ssim=0.5, lambda_lpips=0.2)

    # whether to resume training
    if len(resume) > 0:
        checkpoint = torch.load(resume, map_location='cpu') # 先加载模型到cpu上，等需要的时候再加载到gpu
        SEHF.load_state_dict(checkpoint['SEHF'])
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
        SEHF.train()
        train_loss = 0
        patch_iter = 0

        for event_, rgb_ in tqdm(train_loader, desc='Train dataLoader', total=len(train_loader)):
            patch_iter += 1
            optimizer.zero_grad()
            rgb_total = rgb_.permute(0, 4, 1, 2, 3).to(device) # 1*3*25*280*448
            rgb_total = rgb_total / 255.0
            event_total = []

            for i in range(rgb_total.shape[2]):
                t = event_['t'][i][0].to(torch.float32)
                x = event_['x'][i][0].to(torch.float32)
                y = event_['y'][i][0].to(torch.float32)
                p = event_['p'][i][0].to(torch.float32)
                event_total.append(stack_data(t, x, y, p, w, h))
            event_total = torch.stack(event_total, dim=0).unsqueeze(0).permute(0, 2, 1, 4, 3).to(device) 
            # 1*1*25*280*448
            rgb_first = rgb_total[:, :, 0, :, :].to(torch.float32)   # 3*280*448
            event_first = event_total[:, :, 0, :, :]   # 2*280*448    

            rgb_gt = rgb_total[:, :, 1:, :, :].to(torch.float32) # 1*3*24*280*448
            event_input = event_total[:, :, 1:, :, :]  # 1*2*24*280*448

            rgb_first.to(device)
            event_first.to(device)
            rgb_gt.to(device)
            event_input.to(device)

            if scaler is not None:
                with torch.autocast("cuda:0"):
                    output = SEHF(event_first, rgb_first, event_input)
                    # loss = F.mse_loss(output, rgb_gt)
                    loss = hybrid_loss(output, rgb_gt)
                print()
                print(f'current patch: {patch_iter}, loss: {loss.item()}')
                print()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)  # 取消缩放以便裁剪
                torch.nn.utils.clip_grad_value_(SEHF.parameters(), clip_value=5.0)
                scaler.step(optimizer)
                scaler.update()
                   
            else:
                output = SEHF(event_first, rgb_first, event_input)
                loss = hybrid_loss(output, rgb_gt)
                # loss = F.mse_loss(output, rgb_gt)
                print()
                print(f'current patch: {patch_iter}, loss: {loss.item()}')
                print()
                loss.backward()
                torch.nn.utils.clip_grad_value_(SEHF.parameters(), clip_value=5.0)

                optimizer.step()
            train_loss += loss.item()

            if loss < 1.0 or patch_iter % 200 == 0:
                output = output.squeeze(0).permute(1, 2, 3, 0).detach().cpu().numpy().astype(np.uint8)
                for i in range(output.shape[0]):
                    img = output[i][:, :, [2, 1, 0]]
                    img = Image.fromarray(img)
                    img.save(os.path.join(sample_check, f'epoch_{epoch}_train_{patch_iter}_frame_{i}_loss_{loss}.png'))
                    


        train_time = time.time()
        avg_train_loss = train_loss / patch_iter
        writer.add_scalar('train_loss', avg_train_loss, epoch)



        SEHF.eval()
        test_loss = 0
        test_patch_iter = 0
        with torch.no_grad():
            for event_total, rgb_total in tqdm(test_loader, desc='test dataLoader', total=len(test_loader)):
                test_patch_iter += 1
                rgb_total = rgb_.permute(0, 4, 1, 2, 3).to(device) # 1*3*25*448*280
                event_total = []

                for i in range(rgb_total.shape[2]):
                    t = event_['t'][i][0].to(torch.float32)
                    x = event_['x'][i][0].to(torch.float32)
                    y = event_['y'][i][0].to(torch.float32)
                    p = event_['p'][i][0].to(torch.float32)
                    event_total.append(stack_data(t, x, y, p, w, h))
                event_total = torch.stack(event_total, dim=0).unsqueeze(0).permute(0, 2, 1, 4, 3).to(device) 

                rgb_first = rgb_total[:, :, 0, :, :].to(torch.float16)   # 3*448*280
                event_first = event_total[:, :, 0, :, :]   # 2*448*280    

                rgb_gt = rgb_total[:, :, 1:, :, :].to(torch.float16)  # 1*3*24*448*280
                event_input = event_total[:, :, 1:, :, :]  # 2*24*448*280

                output = SEHF(event_first, rgb_first, event_input)
                # loss = F.mse_loss(output, rgb_gt)
                loss = hybrid_loss(output, rgb_gt)
                test_loss += loss.item()

                if test_patch_iter % 100 == 0:
                    output = output.squeeze(0).permute(1, 2, 3, 0).detach().cpu().numpy().astype(np.uint8)
                    for i in range(output.shape[0]):
                        img = output[i][:, :, [2, 1, 0]]
                        img = Image.fromarray(img)
                        img.save(os.path.join(sample_check, f'epoch_{epoch}_test_{patch_iter}_frame_{i}.png'))


        test_time = time.time()
        avg_test_loss = test_loss / test_patch_iter
        writer.add_scalar('test_loss',avg_test_loss, epoch)

        save_best = False
        if avg_test_loss < lowest_test_loss:
            lowest_test_loss = avg_test_loss
            best_epoch = epoch
            save_best = True

        checkpoint = {
            'SEHF': SEHF.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'lowest_test_loss': test_loss,
            'best_epoch': best_epoch
        }

        if save_best:
            torch.save(checkpoint, os.path.join(out_dir, 'best.pth'))

        torch.save(checkpoint, os.path.join(out_dir, 'last.pth'))

        print(f'epoch = {epoch}, train_loss ={train_loss: .4f}, test_loss ={test_loss: .4f}, epoch_span ={(test_time-starttime)//60}min{(test_time-starttime)%60}s, train_time ={(train_time-starttime)//60}min{(train_time-starttime)%60}s, lowest_test_loss ={lowest_test_loss: .4f}_epoch({best_epoch})')
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


