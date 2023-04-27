
#PyTorch lib
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torch.nn.functional as F
import torchvision
#Tools lib
import numpy as np
import cv2
import random
import time
import sys
import os


def print_network(net):
	num_params = 0
	for param in net.parameters():
		num_params += param.numel()
	print(net)
	print('Total number of parameters: %d' % num_params)

class DRN(nn.Module):
    def __init__(self, channel=3, inter_iter=7, intra_iter=7, a_S=0.5, use_GPU=True):
        super(DRN, self).__init__()
        self.iteration = inter_iter
        self.intra_iter = intra_iter
        self.a_S = a_S
        self.a_D = 1 - self.a_S
        
        channel_feature = 16

        self.use_GPU = use_GPU

        self.conv0_1 = nn.Sequential(
            nn.Conv2d(channel*2, channel_feature, 3, 1, 1)
            )

        self.relu0_1 = nn.Sequential(
            nn.ReLU()
            )
            
        self.conv0_2 = nn.Sequential(
            nn.Conv2d(channel*2, channel_feature, 3, 1, 1)
            )

        self.relu0_2 = nn.Sequential(
            nn.ReLU()
            )

        self.res_conv1_1 = nn.Sequential(
            nn.Conv2d(channel_feature, channel_feature, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(channel_feature, channel_feature, 3, 1, 1),
            nn.ReLU()
            )

        self.res_conv1_2 = nn.Sequential(
            nn.Conv2d(channel_feature, channel_feature, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(channel_feature, channel_feature, 3, 1, 1),
            nn.ReLU()
            )

        self.res_conv2_1 = nn.Sequential(
            nn.Conv2d(channel_feature, channel_feature, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(channel_feature, channel_feature, 3, 1, 1),
            nn.ReLU()
        )

        self.res_conv2_2 = nn.Sequential(
            nn.Conv2d(channel_feature, channel_feature, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(channel_feature, channel_feature, 3, 1, 1),
            nn.ReLU()
        )

        self.conv_1 = nn.Sequential(
            nn.Conv2d(channel_feature, channel, 3, 1, 1)
            )

        self.conv_2 = nn.Sequential(
            nn.Conv2d(channel_feature, channel, 3, 1, 1)
            )
        
        self.conv_H_1 = nn.Sequential(
            nn.Conv2d(channel, channel_feature, 3, 1, 1),
            nn.ReLU()
            )

        self.conv_H_2 = nn.Sequential(
            nn.Conv2d(channel_feature, channel_feature, 3, 1, 1),
            nn.ReLU()
            )

        self.conv_H_3 = nn.Sequential(
            nn.Conv2d(channel_feature, channel, 3, 1, 1),
            nn.ReLU()
            )


    def cross_stitch_linearcomb(self,aS,aD,xA,xB):
        xA_out = torch.add(aS*xA,aD*xB)
        xB_out = torch.add(aD*xA,aS*xB)
            
        return (xA_out,xB_out)
        
    def forward(self, x, stitchon=0):
        
        x_1_orig = x
        x_2_orig = x
        
        x1 = x
        x2 = x

        x_list_1 = []
        x_list_2 = []
        
        for i in range(self.iteration):
            x_1 = torch.cat((x,x1), 1)
            x_1 = self.conv0_1(x_1)
            x_1 = self.relu0_1(x_1)

            x_2 = torch.cat((x,x2),1)
            x_2 = self.conv0_2(x_2)
            x_2 = self.relu0_2(x_2)

            
            # compute cross-stitch feature maps
            if stitchon == 1:
                x_12_1, x_12_2 = self.cross_stitch_linearcomb(self.a_S, self.a_D, x_1, x_2)

            # swap feature maps
            if stitchon == 1:
                x_1 = x_12_1
                x_2 = x_12_2
            
            resx_1 = x_1
            resx_2 = x_2
            
            for j in range(self.intra_iter):
                # branch A: rain
                x_1 = F.relu(self.res_conv1_1(x_1) + resx_1)
                resx_1 = x_1
                
                x_1 = F.relu(self.res_conv2_1(x_1) + resx_1)
                resx_1 = x_1          
                
                x_2 = F.relu(self.res_conv1_2(x_2) + resx_2)
                resx_2 = x_2
                
                x_2 = F.relu(self.res_conv2_2(x_2) + resx_2)
                resx_2 = x_2

            if stitchon == 1:
                x_12_1, x_12_2 = self.cross_stitch_linearcomb(self.a_S, self.a_D, x_1, x_2)

                x_1 = x_12_1
                x_2 = x_12_2
            
            x_1 = self.conv_1(x_1)
            x_1 = x_1 + x
            
            x_2 = self.conv_2(x_2)
            x_2 = x_2 + x
            
            # bottleneck
            x_list_1.append(x_1)
            x_list_2.append(x_2)
        
        y_add = x_1 + x_2
#        y_add = torch.cat((y_add,x),1)

#        y_add = torch.cat((x_1,x_2),1)
        
        y_add = self.conv_H_1(y_add)
        y_add = self.conv_H_2(y_add)
        y_add = self.conv_H_3(y_add)

        x_stacked = y_add
#        x_stacked = y_add + x

        x_stacked_list = []
        x_stacked_list.append(y_add)
        
        #return x_1, x_list_1, x_2, x_list_2, x_stacked, x_stacked_list
        return x_stacked
        
class DRN_baseline(nn.Module):
    def __init__(self, channel=3, inter_iter=7, intra_iter=7, use_GPU=True):
        super(DRN_baseline, self).__init__()
        self.iteration = inter_iter
        self.intra_iter = intra_iter
        
        channel_feature = 16

        self.use_GPU = use_GPU

        self.conv0_1 = nn.Sequential(
            nn.Conv2d(channel*2, channel_feature, 3, 1, 1)
            )

        self.relu0_1 = nn.Sequential(
            nn.ReLU()
            )
            
        self.conv0_2 = nn.Sequential(
            nn.Conv2d(channel*2, channel_feature, 3, 1, 1)
            )

        self.relu0_2 = nn.Sequential(
            nn.ReLU()
            )

        self.res_conv1_1 = nn.Sequential(
            nn.Conv2d(channel_feature, channel_feature, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(channel_feature, channel_feature, 3, 1, 1),
            nn.ReLU()
            )

        self.res_conv1_2 = nn.Sequential(
            nn.Conv2d(channel_feature, channel_feature, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(channel_feature, channel_feature, 3, 1, 1),
            nn.ReLU()
            )

        self.res_conv2_1 = nn.Sequential(
            nn.Conv2d(channel_feature, channel_feature, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(channel_feature, channel_feature, 3, 1, 1),
            nn.ReLU()
        )

        self.res_conv2_2 = nn.Sequential(
            nn.Conv2d(channel_feature, channel_feature, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(channel_feature, channel_feature, 3, 1, 1),
            nn.ReLU()
        )

        self.conv_1 = nn.Sequential(
            nn.Conv2d(channel_feature, channel, 3, 1, 1)
            )

        self.conv_2 = nn.Sequential(
            nn.Conv2d(channel_feature, channel, 3, 1, 1)
            )
        
        self.conv_H_1 = nn.Sequential(
            nn.Conv2d(channel, channel_feature, 3, 1, 1),
            nn.ReLU()
            )

        self.conv_H_2 = nn.Sequential(
            nn.Conv2d(channel_feature, channel_feature, 3, 1, 1),
            nn.ReLU()
            )

        self.conv_H_3 = nn.Sequential(
            nn.Conv2d(channel_feature, channel, 3, 1, 1),
            nn.ReLU()
            )


    def cross_stitch_linearcomb(self,aS,aD,xA,xB):
        xA_out = torch.add(aS*xA,aD*xB)
        xB_out = torch.add(aD*xA,aS*xB)
            
        return (xA_out,xB_out)
        
    def forward(self, x, stitchon=0):
        
        x_1_orig = x
        x_2_orig = x
        
        x1 = x
        x2 = x

        x_list_1 = []
        x_list_2 = []
        
        for i in range(self.iteration):
            x_1 = torch.cat((x,x1), 1)
            x_1 = self.conv0_1(x_1)
            x_1 = self.relu0_1(x_1)

            x_2 = torch.cat((x,x2),1)
            x_2 = self.conv0_2(x_2)
            x_2 = self.relu0_2(x_2)
            
            resx_1 = x_1
            resx_2 = x_2
            
            for j in range(self.intra_iter):
                # branch A: rain
                x_1 = F.relu(self.res_conv1_1(x_1) + resx_1)
                resx_1 = x_1
                
                x_1 = F.relu(self.res_conv2_1(x_1) + resx_1)
                resx_1 = x_1          
                
                x_2 = F.relu(self.res_conv1_2(x_2) + resx_2)
                resx_2 = x_2
                
                x_2 = F.relu(self.res_conv2_2(x_2) + resx_2)
                resx_2 = x_2

            x_1 = self.conv_1(x_1)
            x_1 = x_1 + x
            
            x_2 = self.conv_2(x_2)
            x_2 = x_2 + x
            
            # bottleneck
            x_list_1.append(x_1)
            x_list_2.append(x_2)
        
        y_add = x_1 + x_2
        
#        y_add = torch.cat((y_add,x),1)
#        y_add = torch.cat((x_1,x_2),1)
        
        y_add = self.conv_H_1(y_add)
        y_add = self.conv_H_2(y_add)
        y_add = self.conv_H_3(y_add)

        x_stacked = y_add
#        x_stacked = y_add + x

        x_stacked_list = []
        x_stacked_list.append(y_add)
        
        return x_1, x_list_1, x_2, x_list_2, x_stacked, x_stacked_list