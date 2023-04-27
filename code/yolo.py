import torch
from email.mime import image
import glob
import os
from statistics import mean
from sys import stderr
from turtle import forward
import torch
from PIL import Image, ImageOps
import torchvision
from torchvision import transforms
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import math
import random
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import skimage as io
import matplotlib.pyplot as plt
# from dataloader import NyuDepth_test
import torch
import torchvision
from torchvision.io import read_image
from torchvision.utils import draw_bounding_boxes
from mobvit_small import EncoderDecoder
# from test import inverse_depth_norm

def inverse_depth_norm(depth):
    zero_mask = depth == 0.0
    depth = 10 / depth
    depth = torch.clamp(depth, 10 / 100, 10)
    depth[zero_mask] = 0.0
    return depth

def main() :
    to_tensor = transforms.ToTensor()
    # train_dataset = NyuDepth_test()
    # train_load = DataLoader(dataset=train_dataset, batch_size=1, shuffle=True)
    # datatiter = iter(train_load)
    # data = datatiter.next()
    image = Image.open("D:\dimpattas\Python_Code\Robust_vision\RESCUER_ROBUST_VISION\code\GOPR0244.JPG")
    image = to_tensor(image)
    input = torch.autograd.Variable(image.cuda())
    print(input.shape)
    output = torch.autograd.Variable(image.cuda())
    print(output[0].min())
    print(output[0].max())
  

# Model
    # model = torch.hub.load('ultralytics/yolov5', "D:\dimpattas\Python_Code\Robust_vision\RESCUER_ROBUST_VISION\code\yolo.py", pretrained=True)
    transformss = transforms.ToPILImage()
# Images
    imgs = transformss(input[0])

# Inference
    # results = model(imgs)

# # Results
#     results.print()
#     results.show()  # or .show()
#     box_list = []
#     labels = []
#     #print(results.xyxy[0]) # img1 predictions (tensor)
#     print(results.pandas().xyxy[0])  # img1 predictions (pandas)
#     boxes = results.pandas().xyxy[0]

#     for i in range(len(results.pandas().xyxy[0])):
#         xmin= boxes.xmin[i]
#         xmax=boxes.xmax[i]
#         ymin = boxes.ymin[i]
#         ymax = boxes.ymax[i]
#         label = boxes.name[i]
#         c = [xmin,ymin,xmax,ymax]
#         box_list.append(c)
#         labels.append(label)
    

    model = EncoderDecoder(batch_size=1)
    model.cuda()
    PATH = r'./code/weights_depth/mobilenet_small_MSE_l2=0.0001_bsize=8.pth'
    model.load_state_dict(torch.load(PATH))
    model.eval()
    # dataset = NyuDepth_test()
    # dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)
    # total_samples = len(dataloader)
    # datatiter = iter(dataloader)
    # data = datatiter.next()
    image = Image.open("D:\dimpattas\Python_Code\Robust_vision\RESCUER_ROBUST_VISION\code\GOPR0244.JPG")
    image = image.resize((640,480), Image.Resampling.LANCZOS)
    image = to_tensor(image)
    # input = torch.autograd.Variable(image.cuda().unsqueeze(0))
    # output = torch.autograd.Variable(image.cuda())
    output = model.forward(image.cuda().unsqueeze(0))
    output = inverse_depth_norm(output)
    output = output/10
    output = torch.squeeze(output,0)
    output = output*255
    output= output.repeat(3, 1, 1)
    output= torch.tensor(output, dtype=torch.uint8).cpu()

    output = transformss(output)
    output.show()


    depths = []
    box_list = torch.tensor(box_list, dtype=torch.int)
    for box in box_list:
        depths.append(round(torch.mean(output[0,box[0]:box[2],box[1]:box[3]]).item()))


if __name__ == '__main__':
    main()
        
    
        


