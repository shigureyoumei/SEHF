import cv2

if __name__ == '__main__':
    # 读取 RGB 图像
    event = cv2.imread('/mnt/e/projects/pair/pair/event_flip.jpg')  
    
    event_resize = event[115:397, 96:537]  # 重设大小
    
    # 保存映射后的图像
    save_path = '/mnt/e/projects/pair/pair/event_resize.jpg'
    cv2.imwrite(save_path, event_resize)