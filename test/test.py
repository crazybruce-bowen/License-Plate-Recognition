# 全流程
# -----------------------------
# import区域
## 基础包
import os

## 项目方法
path = r'D:\Learn\学习入口\大项目\车牌识别\MyProject'
os.chdir(path)
from fancy_method.detect_plate import detect
# -----------------------------
# func 区域

# 数据读入和图像处理
pass
# 车牌位置识别
path_detect = r'D:\Learn\学习入口\大项目\车牌识别\MyProject\fancy_method'
os.chdir(path_detect)
source = r'D:\Learn\学习入口\大项目\车牌识别\MyProject\test\image'
output = r'D:\Learn\学习入口\大项目\车牌识别\MyProject\test\output\license_detect'
weights = r'D:\Learn\学习入口\大项目\车牌识别\MyProject\test\last.pt'
detect(source=source, output=output, weights=weights, save_txt=True)
# 车牌图像截取
pass
# 车牌图像文字识别
pass
# 结果导出
pass

