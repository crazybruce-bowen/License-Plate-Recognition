# -*- coding: utf-8 -*-
"""
@author: LBW

DESC：
    本方法寻找车牌位置

TODO list：
    1. 读取图像，转灰度  DONE
    2. 图像剪裁  DONE
    3. 图像降噪  DONE
    4. 形态学处理--开运算  DNOE
    5. 阈值分割  DONE
    6. 边缘检测  DONE
    7. 筛选出车牌边缘  DONE
    8. 摘出车牌  DONE
"""
from PIL import Image, ImageDraw
import cv2
import numpy as np


def my_img_read(path, grey=True, width=480):
    """

    :param path: str 图像路径
    :param grey: bool 是否将图像转为灰度
    :param width: int 转换后的图像宽度
    :return: PIL Image类
    """
    # 读取图像并转灰度
    img = Image.open(path)
    if grey:
        img = img.convert('L')
    # 图像剪裁
    # rat = width / img.size[1]
    # img = img.resize([int(i * rat) for i in img.size])
    return img


def pic_ops(img):
    """
    实现功能：
        1. 图像降噪
        2. 形态学处理--开运算
        3. 阈值分割
        4. 边缘检测
    
    :param img: PIL Image类
    :return  img1, contours
        img1 Image 类 -- 高斯降噪后的图像
        contours  np.array -- 边缘位置
        
    """
    arr_img = np.array(img)
    arr_img1 = cv2.GaussianBlur(arr_img, (5,5), 0, 0, cv2.BORDER_DEFAULT)
    kernel = np.ones((23, 23), np.uint8)
    img_opening = cv2.morphologyEx(arr_img, cv2.MORPH_OPEN, kernel)
    img_opening = cv2.addWeighted(arr_img, 1, img_opening, -1, 0)
    # 找到图像边缘
    ret, img_thresh = cv2.threshold(img_opening, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    img_edge = cv2.Canny(img_thresh, 100, 200)
    # 使用开运算和闭运算让图像边缘成为一个整体
    kernel = np.ones((10, 10), np.uint8)
    img_edge1 = cv2.morphologyEx(img_edge, cv2.MORPH_CLOSE, kernel)
    img_edge2 = cv2.morphologyEx(img_edge1, cv2.MORPH_OPEN, kernel)
    # 查找图像边缘整体形成的矩形区域
    contours, hierarchy = cv2.findContours(img_edge2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    img1 = Image.fromarray(arr_img1)
    return img1, contours    


def find_car_plate(contours, Min_Area = 2000):
    """
    在众多矩阵中用规则筛选出车牌矩阵

    :param contours:
    :param Min_Area:
    :return car_plate: 
    """
    temp_contours = []
    for contour in contours:
        if cv2.contourArea( contour ) > Min_Area:
            temp_contours.append(contour)
    car_plate = []
    for temp_contour in temp_contours:
        rect_tupple = cv2.minAreaRect( temp_contour )
        rect_width, rect_height = rect_tupple[1]
        if rect_width < rect_height:
            rect_width, rect_height = rect_height, rect_width
        aspect_ratio = rect_width / rect_height
        # 车牌正常情况下宽高比在2 - 5.5之间
        if aspect_ratio > 2 and aspect_ratio < 5.5:
            car_plate.append( temp_contour )
            rect_vertices = cv2.boxPoints( rect_tupple )
            rect_vertices = np.int0( rect_vertices )
    return car_plate


def draw_contours(img, contours):
    """
    在图像中画出轮廓线

    :param img:
    :param contours: [3-D np.array]
    :return:
    """
    draw = ImageDraw.Draw(img)
    for contour in contours:
        x_max, y_max = contour.max(0)[0]
        x_min, y_min = contour.min(0)[0]
        draw.rectangle([x_min, y_min, x_max, y_max])
    img.show()
    return True


def filter_plate_pic(img, contour, display=False):
    """
    从图片中截取出车牌范围，返回截取的图片

    :param img:
    :param contour: [3-D np.array]
    :param display: 是否展示
    :return car_plate: PIL Image类，截取的车牌图片
    """
    img = img.copy()
    x_max, y_max = contour.max(0)[0]
    x_min, y_min = contour.min(0)[0]
    box = [x_min, y_min, x_max, y_max]
    car_plate = img.crop(box)
    if display:
        car_plate.show()
    return car_plate


#%% Test
if __name__ == '__main__':
    path = r'D:\Learn\学习入口\大项目\车牌识别\data\1.jpeg'
    img0 = my_img_read(path, False)
    img = my_img_read(path)
    img1, contours = pic_ops(img)
    car_plate_contours = find_car_plate(contours)
    # draw_contours(img0, car_plate_contours)
    img_plate = filter_plate_pic(img0, car_plate_contours[0], display=True)
