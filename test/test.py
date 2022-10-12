# 全流程
# -----------------------------
# import区域
## 基础包
import os
from PIL import Image
import numpy as np
import copy
## 项目方法
path = r'D:\Learn\学习入口\大项目\车牌识别\MyProject'
os.chdir(path)
from fancy_method.detect_plate import detect_main
from utils.common_utils import xywh2xyxy, plot_one_box, plot_images
# -----------------------------
# func 区域

## TODO 创建自己的dataloader
class MyImageLoader:
    def __init__(self, img_path):
        """
            img_path str 图片路径
        """
        self.img = Image.open(img_path)

class MyDataLoader:
    def __init__(self, path_folder):
        self.folder = path_folder
        


#%% 数据读入和图像处理
pass
# 车牌位置识别
path_detect = r'D:\Learn\学习入口\大项目\车牌识别\MyProject\fancy_method'
os.chdir(path_detect)
source = r'D:\Learn\学习入口\大项目\车牌识别\MyProject\fancy_method\data\images\test_BW'
output = r'D:\Learn\学习入口\大项目\车牌识别\MyProject\test\output\license_detect'
weights = r'D:\Learn\学习入口\大项目\车牌识别\MyProject\test\last.pt'
res = detect_main(source=source, output=output, weights=weights, save_txt=True)

#%% 车牌图像截取
res1 = copy.deepcopy(res)
for i in res1:
    tmp = []
    file_path = i['file_path']
    img = Image.open(file_path)
    im = np.array(img)
    h, w = im.shape[:2]
    
    boxs_xywh = np.array(i['detect_box'])
    if boxs_xywh.size>0:
        boxs_xyxy = xywh2xyxy(boxs_xywh).tolist()
        for j in boxs_xyxy:
            j[0] *= w
            j[2] *= w
            j[1] *= h
            j[3] *= h
            tmp.append(j)
        
        i['box_img'] = [img.crop(j) for j in tmp]
        i['box_im'] = [np.array(img.crop(j)) for j in tmp]
    else:
        i['box_img'] = list()
        i['box_im'] = list()
#%%
# 车牌图像文字识别
pass
# 结果导出
pass


#%% 测试区域
list_box = list()
for i in res1:
    if i['box_img']:
        list_box += i['box_img']
        