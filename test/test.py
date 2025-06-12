import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import h5py
# import os
import cv2
from tqdm import tqdm
from model import framework
import torch
# from utils import checkh5file

if __name__ == "__main__":
    root = '~/projects/test/testtry3'
    pth = '~/projects/result/6_6_lr/2025_6_4_b2_lr0.0001/best.pth'
    root = os.path.expanduser(root)
    pth = os.path.expanduser(pth)

    file_list = []

    for root, dirs, files in os.walk(root):
        for file in files:
            if file.endswith('.h5'):
                file_list.append(os.path.join(root, file))
    print(f"Total files found: {len(file_list)}")

    file_list.sort(key=lambda x: int(x.split('/')[-1].split('.')[0].split('_')[-1]))
    print(f"Files sorted: {file_list}")

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f'Using device {device}')

    SEHF = framework.get_pose_net()
    SEHF.to(device)

    checkpoint = torch.load(pth, map_location=device)
    SEHF.load_state_dict(checkpoint['SEHF'])
    SEHF.eval()

    epoch = checkpoint['epoch']
    print(f"Model loaded from {pth}, epoch: {epoch}")

    # Create output directory if it doesn't exist
    output_dir = '~/projects/test/testtry3_output'
    output_dir = os.path.expanduser(output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    rgb_output_dir = os.path.join(output_dir, 'rgb')
    event_output_dir = os.path.join(output_dir, 'event')
    if not os.path.exists(rgb_output_dir):
        os.makedirs(rgb_output_dir)
    if not os.path.exists(event_output_dir):
        os.makedirs(event_output_dir)


    idx = 1
    file_iter = 1
    for path in tqdm(file_list, desc="Processing files", total=len(file_list)):
        
        # checkh5file.print_h5_contents(path)
        with h5py.File(path, 'r') as f:
            rgb = f['rgb'][:]   # (4, 140, 224, 3)
            event = f['event'][:]   # (16, 140, 224, 2)
            event_show = f['event_show'][:]   # (16, 140, 224, 2)
            rgb = np.astype(rgb, np.float32)
            event = np.astype(event, np.float32)

            # Convert to PyTorch tensors
            rgb_tensor = torch.tensor(rgb).to(device)
            event_tensor = torch.tensor(event).to(device)

            rgb_tensor = rgb_tensor.unsqueeze(0).permute(0, 1, 4, 2, 3) # (1, 4, 140, 224, 3)
            event_tensor = event_tensor.unsqueeze(0).permute(0, 1, 4, 2, 3) # (1, 16, 140, 224, 2)

            

            e_group = 4
            for i in range(rgb_tensor.shape[1]):
                rgb_input = rgb_tensor[:, i, :, :, :]
                event_input = event_tensor[:, i*e_group:i*e_group+e_group, :, :, :]
                
                # Forward pass through the model
                with torch.no_grad():
                    output = SEHF(event_input, rgb_input)
                    output = output.squeeze(0)
                    output = output * 255.0
                    
                # Convert output to numpy and save
                output_np = output.permute(0, 2, 3, 1).cpu().numpy()
                output_np = output_np[:,:,:,::-1]

                for i in range(output_np.shape[0]):
                    output_img = output_np[i].astype(np.uint8)
                    rgb_output_path = os.path.join(rgb_output_dir, f"{file_iter}_{idx:03d}.png")
                    event_output_path = os.path.join(event_output_dir, f"{file_iter}_{idx:03d}.png")
                    idx += 1
                    cv2.imwrite(rgb_output_path, output_img)
                    cv2.imwrite(event_output_path, event_show[i].astype(np.uint8))
        file_iter += 1


