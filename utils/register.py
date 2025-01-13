import cv2
import numpy as np

def align_images(rgb_image_path, event_image_path, output_rgb_image_path, output_event_image_path):
    # 读取 RGB 图像和事件图像
    rgb_image = cv2.imread(rgb_image_path)
    event_image = cv2.imread(event_image_path)

    # 定义 RGB 相机和事件相机的对应点
    # rgb_points = np.float32([[11, 15], [1904, 16], [1913, 1196], [5, 1198]])  #version1
    # rgb_points = np.float32([[9, 3], [1905, 6], [1914, 1206], [3, 1208]])   #version2
    rgb_points = np.float32([[8, 2], [1907, 4], [1917, 1208], [2, 1211]])   #version3
    event_points = np.float32([[5, 0], [440, 5], [439, 281], [0, 275]])  #101,115  536,120  535,396  96,390

    # 计算透视变换矩阵
    matrix = cv2.getPerspectiveTransform(rgb_points, event_points)

    # 对 RGB 图像进行透视变换
    aligned_rgb_image = cv2.warpPerspective(rgb_image, matrix, (441, 282))

    # 裁剪事件相机图像的四个对应点范围，并拉正为长方形
    # event_rect_points = np.float32([[0, 0], [435, 0], [435, 281], [0, 281]])
    # event_correct_matrix = cv2.getPerspectiveTransform(event_points, event_rect_points)
    # event_correct = cv2.warpPerspective(event_image, event_correct_matrix, (435, 281))

    # 裁剪对齐后的 RGB 图像中相对应的区域
    # rgb_correct = aligned_rgb_image[115:397, 96:537]
    event_correct = event_image[115:397, 96:537]

    # 保存对齐后的图像和裁剪后的图像
    cv2.imwrite(output_rgb_image_path, aligned_rgb_image)
    cv2.imwrite(output_event_image_path, event_correct)

# 示例用法
rgb_image_path = '/mnt/e/projects/pair/pair/rgb.jpeg'
# event_image_path = 'data/register/event/000001.png'
event_image_path = '/mnt/e/projects/pair/pair/event.jpg'
# rgb_image_path = 'data/register/rgb/02.jpeg'
output_rgb_image_path = '/mnt/e/projects/pair/pair/registered_rgb_correct2.jpeg'
output_event_image_path = '/mnt/e/projects/pair/pair/event_correct2.jpeg'
align_images(rgb_image_path, event_image_path, output_rgb_image_path, output_event_image_path)