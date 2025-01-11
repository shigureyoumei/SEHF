import cv2
import numpy as np

def align_images(rgb_image_path, event_image_path, output_rgb_image_path, output_event_image_path):
    # 读取 RGB 图像和事件图像
    rgb_image = cv2.imread(rgb_image_path)
    event_image = cv2.imread(event_image_path)

    # 定义 RGB 相机和事件相机的对应点
    rgb_points = np.float32([[10, 13], [1905, 15], [1914, 1198], [4, 1199]])
    event_points = np.float32([[101, 115], [536, 120], [535, 396], [96, 390]])

    # 计算透视变换矩阵
    matrix = cv2.getPerspectiveTransform(rgb_points, event_points)

    # 对 RGB 图像进行透视变换
    aligned_rgb_image = cv2.warpPerspective(rgb_image, matrix, (640, 480))

    # 裁剪事件相机图像的四个对应点范围，并拉正为长方形
    # event_rect_points = np.float32([[101, 115], [536, 115], [536, 396], [101, 396]])
    # event_correct_matrix = cv2.getPerspectiveTransform(event_points, event_rect_points)
    # event_correct = cv2.warpPerspective(event_image, event_correct_matrix, (435, 281))

    # 裁剪对齐后的 RGB 图像中相对应的区域
    rgb_correct = aligned_rgb_image[115:396, 101:536]
    event_correct = event_image[115:396, 101:536]

    # 保存对齐后的图像和裁剪后的图像
    cv2.imwrite(output_rgb_image_path, rgb_correct)
    cv2.imwrite(output_event_image_path, event_correct)

# 示例用法
rgb_image_path = '/mnt/e/projects/pair/pair/rgb.jpeg'
event_image_path = '/mnt/e/projects/pair/pair/event.jpg'
output_rgb_image_path = '/mnt/e/projects/pair/pair/registered_rgb_correct.jpeg'
output_event_image_path = '/mnt/e/projects/pair/pair/event_correct.jpeg'
align_images(rgb_image_path, event_image_path, output_rgb_image_path, output_event_image_path)