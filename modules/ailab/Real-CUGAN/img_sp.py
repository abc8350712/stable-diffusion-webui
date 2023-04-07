'''
Author: yxd3 abc8350712@gmail.com
Date: 2023-03-21 22:54:16
LastEditors: yxd3 abc8350712@gmail.com
LastEditTime: 2023-03-22 00:02:57
FilePath: /AI/super_resolution/ailab/Real-CUGAN/img_sp.py
Description: 

Copyright (c) 2023 by ${git_name_email}, All Rights Reserved. 
'''
import sys
import os
import cv2
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


current_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_path)

sys.path.append(os.path.join(current_dir, "."))
model_path = os.path.join(current_dir, 'model')
img_dir = "/Users/moon/Downloads/tmpjxmby22u.png"
from upcunet_v3 import RealWaifuUpScaler

def img_sp(img):

    img = np.array(img)
    new_img = img[:, :, :3]

    if torch.cuda.is_available():
                upscaler = RealWaifuUpScaler(2, os.path.join(model_path, "pro-denoise3x-up2x.pth"), half=True, device="cuda:0")

    else:
        upscaler = RealWaifuUpScaler(2, os.path.join(model_path, "pro-denoise3x-up2x.pth"), half=False, device="cpu:0")
    result = upscaler(img,tile_mode=5,cache_mode=2,alpha=1)
    result = Image.fromarray(result)
    return result

if __name__ == "__main__":
    #img = cv2.imread(img_dir)
    img = Image.open(img_dir)
    result = img_sp(img)
    plt.imshow(result)
    plt.show()
    #
