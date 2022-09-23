"""
TODO list:
    1. 读取图像

"""
from PIL import Image


class MyImgRead:

    @classmethod
    def read_img(cls, file_path):
        img = Image.open(file_path)
        return img


