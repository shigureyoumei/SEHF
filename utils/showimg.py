import numpy as np
import matplotlib.pyplot as plt
import os



class show_img:
    def __init__(self, path):
        self.idx = 0
        self.save_dir = os.path.join(path, 'feature')

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def showimg(self, img, title=None):
        
        n = img.size()[1]
        if n > 8:
            n = 8
        cols = 4
        rows = (n + cols - 1) // cols  # 向上取整
        plt.figure(figsize=(12, 8)) 
        for i in range(n):
            plt.subplot(rows, cols, i + 1)
            plt.imshow(img[0, i, :, :].cpu().detach().numpy())
            plt.axis('off')
        # 补空白子图
        for j in range(n, rows * cols):
            plt.subplot(rows, cols, j + 1)
            plt.axis('off')
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.05)
        plt.suptitle(f'After {title}')
        save_path = os.path.join(self.save_dir, f"{self.idx}.jpg")
        plt.savefig(save_path, format='jpg')
        plt.close()
        self.idx += 1
