import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.tensorboard import SummaryWriter
import os
import argparse
import time
import datetime
from tqdm import tqdm

from dataset import *
from torch.utils.data import DataLoader

import model

import subprocess
import webbrowser

import model.SEHF



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

    REClipper = model.SEHF.REClipper(output_channels=9)
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

    train_dataset = pairDateset.MyDateset('train_path')
    test_dataset = pairDateset.MyDateset('test_path')

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)

    scaler = None
    if amp:
        scaler = torch.GradScaler("cuda:0")

    start_epoch = 0
    max_test_acc = -1

    optimizer = None
    if len(opt) > 0:
        if opt == 'adam':
            optimizer = torch.optim.Adam(net.parameters(), lr=lr)
        elif opt == 'sgd':
            optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=momentum)
        else:
            raise ValueError('Invalid optimizer')
    
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    # Learning Rate Scheduler可以根据epoch的数量来调整学习率，这里是余弦退火调度器，学习率随着epoch而进行余弦变化


    # whether to resume training
    if len(resume) > 0:
        checkpoint = torch.load(resume, map_location='cpu') # 先加载模型到cpu上，等需要的时候再加载到gpu
        net.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        max_test_acc = checkpoint['max_test_acc']
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        

    out_dir = os.path.join(out_dir, f'T{T}_b{bs}_{opt}_lr{lr}_c{channels}')
    if amp:
        out_dir += '_amp'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    writer = SummaryWriter(out_dir, purge_step=start_epoch)
    with open(os.path.join(out_dir, 'args.txt'), 'w', encoding='utf-8') as args_txt:
        args_txt.write(str(args))
    

    # training part
    for epoch in tqdm(range(start_epoch, epochs), desc='Epoch', total=(epochs-start_epoch)):
        starttime = time.time()
        net.train()
        train_loss = 0
        train_acc = 0
        train_samples = 0
        for img, label in tqdm(train_dataloader, desc='Train dataLoader', total=len(train_dataloader)):
            optimizer.zero_grad()
            img = img.to(device)
            label = label.to(device)
            label_one_hot = F.one_hot(label, 10).float()

            if scaler is not None:
                with torch.autocast("cuda:0"):
                    out = net(img)
                    loss = F.mse_loss(out, label_one_hot)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                out = net(img)
                loss = F.mse_loss(out, label_one_hot)
                loss.backward()
                optimizer.step()
            
            train_samples += label.numel()
            train_loss += loss.item() * label.numel()
            train_acc += (out.argmax(dim=1) == label).sum().item()


        train_time = time.time()
        train_speed = train_samples / (train_time - starttime)
        train_loss /= train_samples
        train_acc /= train_samples

        writer.add_scalar('train_loss', train_loss, epoch)
        writer.add_scalar('train_acc', train_acc, epoch)

        lr_scheduler.step()

        net.eval()
        test_loss = 0
        test_acc = 0
        test_samples = 0
        with torch.no_grad():
            for img, label in test_dataloader:
                img = img.to(device)
                label = label.to(device)
                label_one_hot = F.one_hot(label, 10).float()
                out = net(img)
                loss = F.mse_loss(out, label_one_hot)
                test_samples += label.numel()
                test_loss += loss.item() * label.numel()
                test_acc += (out.argmax(dim=1) == label).sum().item()
                functional.reset_net(net)
        test_time = time.time()
        test_speed = test_samples / (test_time - train_time)
        test_loss /= test_samples
        test_acc /= test_samples

        writer.add_scalar('test_loss', test_loss, epoch)
        writer.add_scalar('test_acc', test_acc, epoch)

        save_max = False
        if test_acc > max_test_acc:
            max_test_acc = test_acc
            save_max = True

        checkpoint = {
            'net': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'max_test_acc': max_test_acc,
            'lr_scheduler': lr_scheduler.state_dict()
        }

        if save_max:
            torch.save(checkpoint, os.path.join(out_dir, 'max.pth'))

        torch.save(checkpoint, os.path.join(out_dir, 'last.pth'))

        print(f'epoch = {epoch}, train_loss ={train_loss: .4f}, train_acc ={train_acc: .4f}, test_loss ={test_loss: .4f}, test_acc ={test_acc: .4f}, max_test_acc ={max_test_acc: .4f}')
        print(f'escape time = {(datetime.datetime.now() + datetime.timedelta(seconds=(time.time() - starttime) * (epochs - epoch))).strftime("%Y-%m-%d %H:%M:%S")}\n')

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


