import torch
import torch.nn as nn
from einops import rearrange
import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable
import glob
import os
from turtle import forward
import torch
from PIL import Image
from torchvision import transforms
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import math
import random
from torch.utils.data import Dataset, DataLoader
#from dataloader import  NHRESIDEITS, NHRESIDEITS_val, RESIDE_SOTS
import numpy as np
import time
from timm import models as tmod
#from torchsummary import summary
#from kornia.contrib.vit_mobile import MobileViT
#from pytorch_msssim import ms_ssim, ssim
import cv2
import pandas as pd


class SigLoss(nn.Module):
    """SigLoss.

    Args:
        valid_mask (bool, optional): Whether filter invalid gt
        loss_weight (float, optional): Weight of the loss. Defaults to 1.0.
    """

    def __init__(self,
                 valid_mask=True,
                 loss_weight=1.0,
                 max_depth=None,
                 warm_up=False,
                 warm_iter=100):
        super(SigLoss, self).__init__()
        self.valid_mask = valid_mask
        self.loss_weight = loss_weight
        self.max_depth = max_depth

        self.eps = 0.1 # avoid grad explode

        # HACK: a hack implement for warmup sigloss
        self.warm_up = warm_up
        self.warm_iter = warm_iter
        self.warm_up_counter = 0
        self.rms = torch.nn.MSELoss()

    def sigloss(self, input, target):
        if self.valid_mask:
            valid_mask = target > 0
            if self.max_depth is not None:
                valid_mask = torch.logical_and(target > 0, target <= self.max_depth)
            input = input[valid_mask]
            target = target[valid_mask]
        
        if self.warm_up:
            if self.warm_up_counter < self.warm_iter:
                g = torch.log(input + self.eps) - torch.log(target + self.eps)
                g = 0.15 * torch.pow(torch.mean(g), 2)
                self.warm_up_counter += 1
                return torch.sqrt(g)

        g = torch.log(input) - torch.log(target)
        Dg = torch.var(g) + 0.15 * torch.pow(torch.mean(g), 2)
        return torch.sqrt(Dg)

    def rmseloss(self, input, target):
        if self.valid_mask:
            valid_mask = target > 0
            if self.max_depth is not None:
                valid_mask = torch.logical_and(target > 0, target <= self.max_depth)
            input = input[valid_mask]
            target = target[valid_mask]

        # return torch.log(torch.sqrt(self.rms(input, target)))
        return torch.sqrt(self.rms(input, target))

    def sqrelloss(self, input, target):
        if self.valid_mask:
            valid_mask = target > 0
            if self.max_depth is not None:
                valid_mask = torch.logical_and(target > 0, target <= self.max_depth)
            input = input[valid_mask]
            target = target[valid_mask]

        return torch.mean(torch.pow(input - target, 2) / target)

    def forward(self,
                depth_pred,
                depth_gt,
                **kwargs):
        """Forward function."""
        
        loss_depth = self.loss_weight * self.sigloss(
            depth_pred,
            depth_gt,
            )

        # loss_depth = self.rmseloss(depth_pred, depth_gt)

        # loss_depth = self.loss_weight * self.sigloss(
        #     depth_pred,
        #     depth_gt,
        #     ) + self.rmseloss(depth_pred, depth_gt)

        # loss_depth = self.sigloss(
        #     depth_pred,
        #     depth_gt,
        #     ) + 2 * self.rmseloss(depth_pred, depth_gt) + 4 * self.sqrelloss(depth_pred, depth_gt)
        return loss_depth


# Load pre-trained VGG model
vgg_model = models.vgg19(weights='IMAGENET1K_V1')
vgg_model = vgg_model.features
vgg_model = nn.Sequential(*list(vgg_model.children())[:35])  # Choose appropriate layers for perceptual loss
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vgg_model = vgg_model.to(device)

class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        self.criterion = nn.L1Loss()  # L1 loss for pixel-level similarity

    def forward(self, input, target):
        input_features = vgg_model(input)
        target_features = vgg_model(target)
        loss = self.criterion(input_features, target_features)
        return loss


class TVNorm(nn.Module):
    def __init__(self):
        super(TVNorm, self).__init__()

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        img = x.view(batch_size, channels, height, width)

        # Compute horizontal differences
        h_diff = img[:, :, :, :-1] - img[:, :, :, 1:]
        h_norm = torch.norm(h_diff, p=2, dim=1)

        # Compute vertical differences
        v_diff = img[:, :, :-1, :] - img[:, :, 1:, :]
        v_norm = torch.norm(v_diff, p=2, dim=1)

        # Compute TV norm
        tv_norm = torch.mean(h_norm) + torch.mean(v_norm)

        return tv_norm


tv_norm = TVNorm()




def depthwise(in_channels, kernel_size):
    padding = (kernel_size-1) // 2
    assert 2*padding == kernel_size-1, "parameters incorrect. kernel={}, padding={}".format(kernel_size, padding)
    return nn.Sequential(
          nn.Conv2d(in_channels,in_channels,kernel_size,stride=1,padding=padding,bias=False,groups=in_channels),
          #nn.BatchNorm2d(in_channels),
          #nn.ReLU(inplace=True),
        )

def pointwise(in_channels, out_channels):
    return nn.Sequential(
          nn.Conv2d(in_channels,out_channels,1,1,0,bias=False),
          #nn.BatchNorm2d(out_channels),
           #nn.ReLU(inplace=True),
        )

def pointwise_out(in_channels, out_channels):
    return nn.Sequential(
          nn.Conv2d(in_channels,out_channels,1,1,0,bias=False),
          #nn.BatchNorm2d(out_channels)
        )



class Decoder(nn.Module):
    def __init__(self):
        kernel_size = 3
        super().__init__()

        self.conv1 = nn.Sequential(
            depthwise(384, kernel_size),
            pointwise(384, 384),
        )

        self.conv2 = nn.Sequential(
            depthwise(464, kernel_size),
            pointwise(464, 260),
        )

        self.conv3 = nn.Sequential(
            depthwise(324, kernel_size),
            pointwise(324, 130),
        )

        self.conv4 = nn.Sequential(
            depthwise(178, kernel_size),
            pointwise(178, 90),
        )

        self.conv5 = nn.Sequential(
            depthwise(122, kernel_size),
            pointwise(122, 64),
        )

        self.conv6 = nn.Sequential(
            pointwise_out(64, 3),
            #nn.BatchNorm2d(3),
            nn.Sigmoid()
        )
 
        self.conv3_19 = nn.Conv2d(384, 384, kernel_size=7, padding=9, groups=384, dilation=3, padding_mode='reflect')
        self.conv3_13 = nn.Conv2d(384, 384, kernel_size=5, padding=6, groups=384, dilation=3, padding_mode='reflect')
        self.conv3_7 = nn.Conv2d(384, 384, kernel_size=3, padding=3, groups=384, dilation=3, padding_mode='reflect')
        self.conv2_3_19 = nn.Conv2d(260, 260, kernel_size=7, padding=9, groups=260, dilation=3, padding_mode='reflect')
        self.conv2_3_13 = nn.Conv2d(260, 260, kernel_size=5, padding=6, groups=260, dilation=3, padding_mode='reflect')
        self.conv2_3_7 = nn.Conv2d(260, 260, kernel_size=3, padding=3, groups=260, dilation=3, padding_mode='reflect')
        self.conv3_3_19 = nn.Conv2d(130, 130, kernel_size=7, padding=9, groups=130, dilation=3, padding_mode='reflect')
        self.conv3_3_13 = nn.Conv2d(130, 130, kernel_size=5, padding=6, groups=130, dilation=3, padding_mode='reflect')
        self.conv3_3_7 = nn.Conv2d(130, 130, kernel_size=3, padding=3, groups=130, dilation=3,  padding_mode='reflect')
        self.conv4_3_19 = nn.Conv2d(90, 90, kernel_size=7, padding=9, groups=90, dilation=3, padding_mode='reflect')
        self.conv4_3_13 = nn.Conv2d(90, 90, kernel_size=5, padding=6, groups=90, dilation=3, padding_mode='reflect')
        self.conv4_3_7 = nn.Conv2d(90, 90, kernel_size=3, padding=3, groups=90, dilation=3,  padding_mode='reflect')
        self.conv5_3_19 = nn.Conv2d(64, 64, kernel_size=7, padding=9, groups=64, dilation=3, padding_mode='reflect')
        self.conv5_3_13 = nn.Conv2d(64, 64, kernel_size=5, padding=6, groups=64, dilation=3, padding_mode='reflect')
        self.conv5_3_7 = nn.Conv2d(64, 64, kernel_size=3, padding=3, groups=64, dilation=3,  padding_mode='reflect')
        self.relu = nn.ReLU(inplace=True)

        #self.batch1 = nn.BatchNorm2d(384)
        #self.batch2 = nn.BatchNorm2d(260)
        #self.batch3 = nn.BatchNorm2d(130)
        #self.batch4 = nn.BatchNorm2d(90)
        #self.batch5 = nn.BatchNorm2d(64)
        
   


    def forward(self,x,dec1,dec2,dec3,dec4):
   
        
   
        x = self.conv1(x)
        x1 = self.conv3_19(x)
        x2 = self.conv3_13(x)
        x3 = self.conv3_7(x)
        x = x1 + x2 + x3 + x
        #x = self.batch1(x)
        x = self.relu(x)
        x= F.interpolate(x,scale_factor=2, mode= 'bilinear')

        #---------------
        x = torch.cat((x,dec4),1)
        x = self.conv2(x)
        x1 = self.conv2_3_19(x)
        x2 = self.conv2_3_13(x)
        x3 = self.conv2_3_7(x)
        x = x1 + x2 + x3 + x
        #x = self.batch2(x)
        x = self.relu(x)
        x= F.interpolate(x,scale_factor=2, mode= 'bilinear')
        #---------------
        x = torch.cat((x,dec3),1)
        x = self.conv3(x)
        x1 = self.conv3_3_19(x)
        x2 = self.conv3_3_13(x)
        x3 = self.conv3_3_7(x)
        x = x1 + x2 + x3 + x
       # x = self.batch3(x)
        x = self.relu(x)
        x= F.interpolate(x,scale_factor=2, mode= 'bilinear')
        #---------------
        x = torch.cat((x,dec2),1)
        x = self.conv4(x)
        x1 = self.conv4_3_19(x)
        x2 = self.conv4_3_13(x)
        x3 = self.conv4_3_7(x)
        x = x1 + x2 + x3 + x
        #x = self.batch4(x)
        x = self.relu(x)
        x= F.interpolate(x,scale_factor=2, mode= 'bilinear')
        #---------------
        x = torch.cat((x,dec1),1)
        x = self.conv5(x) 
        x1 = self.conv5_3_19(x)
        x2 = self.conv5_3_13(x)
        x3 = self.conv5_3_7(x)
        x = x1 + x2 + x3 + x
        #x = self.batch5(x)
        x = self.relu(x)
        x= F.interpolate(x,scale_factor=2, mode= 'bilinear')
        #--------------
        x= self.conv6(x)
        return x

           



        
features = {}
def get_features(name):
    def hook(model, input, output):
        features[name] = output.detach()
    return hook

class EncoderDecoder(nn.Module):

    def __init__(self,batch_size):
        super().__init__()
        
        self.mvit = tmod.create_model('mobilevit_xs', pretrained=False, num_classes=0, global_pool='')   
        self.Decoder = Decoder()
        stage0 = self.mvit.stages[0].register_forward_hook(get_features('stage_0'))
        stage1 = self.mvit.stages[1].register_forward_hook(get_features('stage_1'))
        stage3 = self.mvit.stages[2].register_forward_hook(get_features('stage_2'))
        stage4 = self.mvit.stages[3].register_forward_hook(get_features('stage_3'))
        self.batch_size = batch_size
    

    def forward(self,x):
        out = self.mvit(x)
        dec1 =features['stage_0']
        dec2 =features['stage_1']
        dec3 =features['stage_2']
        dec4 =features['stage_3']
        out = self.Decoder.forward(out,dec1,dec2,dec3,dec4)
        return out
    


    

    
# def main() :
#     log_loss = SigLoss()
#     loss_l1 = nn.L1Loss()
#     perceptual_loss = PerceptualLoss()
#     tv_norm = TVNorm()
#     device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#     transformss = transforms.ToPILImage()
#     model = EncoderDecoder(batch_size=2)
#     num_epochs = 25
#     BatchSize = 2
#     pytorch_total_params = sum(p.numel() for p in model.parameters())
#     print(pytorch_total_params)
#     PATH_SAVE = r'C:\Users\vclab\Desktop\new\venv\lightweight_perloss_bigger_res_nopre'
    
#     #model.load_state_dict(torch.load(PATH_SAVE))
#     optimizer = optim.Adam(model.parameters(), lr= 0.0001)
#     scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
#     train_dataset = NHRESIDEITS()
#     val_dataset = RESIDE_SOTS()
#     train_load = DataLoader(dataset=train_dataset, batch_size=BatchSize, shuffle=True, num_workers=2)
#     val_load = DataLoader(dataset=val_dataset, batch_size=BatchSize, shuffle=True)
#     model.cuda()
#     model.train() 


#     for epoch in range(num_epochs):  
#         model.train()
#         t0 = time.time()
#         running_loss = 0.0
#         running_loss_val = 0.0
#         for param_group in optimizer.param_groups:
#             print(param_group['lr'])
#         for i, data in enumerate(train_load, 0):
#             optimizer.zero_grad()
#             outputs,inputs = data
#             inputs = inputs.to(device)
#             outputs = outputs.to(device)
#             out = model.forward(inputs)
#             out = out.to(device)
#             #tv_out = tv_norm(out)
#             #per_loss = perceptual_loss(out, outputs)
#             #loss_rec = loss_l1(out, outputs)
#             #loss_ssim = 1 - ssim( out, outputs, data_range=1, size_average=True )
#             loss_log = log_loss(out, outputs)
#             loss =  loss_log 
#             loss.backward()
#             running_loss = running_loss + loss.item()
#             #torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)

#             optimizer.step()
        
#         scheduler.step()
#         print(f'[{epoch + 1}] training_loss: {running_loss / (i + 1) :.3f}')
#         print('{} seconds'.format(time.time() - t0))
#         torch.save(model.state_dict(), PATH_SAVE)
#         print('updated training weights')
        
        
       
       

# if __name__ == '__main__':
#     main()
       

  

