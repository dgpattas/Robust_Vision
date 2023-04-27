import torch
import torch.nn as nn
from einops import rearrange
import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable
import glob
import os
# from turtle import forward
import torch
from PIL import Image
from torchvision import transforms
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import math
import random
from torch.utils.data import Dataset, DataLoader
# from dataloader import  NyuDepth_train, NyuDepth_test
import numpy as np
import time
from timm import models as tmod
# from torchsummary import summary
# from kornia.contrib.vit_mobile import MobileViT





def SiLogLoss( pred, target):
    valid_mask = (pred > 0).detach()
    diff_log = torch.log(target[valid_mask]) - torch.log(pred[valid_mask])
    
    loss = torch.sqrt(torch.pow(diff_log, 2).mean() -
                        0.5 * torch.pow(diff_log.mean(), 2))

    return loss

def logloss( depth_est, depth_gt, mask):
        d = torch.log(depth_est[mask]) - torch.log(depth_gt[mask])
        return torch.sqrt((d ** 2).mean() - 0.5 * (d.mean() ** 2)) * 10.0


def depthwise(in_channels, kernel_size):
    padding = (kernel_size-1) // 2
    assert 2*padding == kernel_size-1, "parameters incorrect. kernel={}, padding={}".format(kernel_size, padding)
    return nn.Sequential(
          nn.Conv2d(in_channels,in_channels,kernel_size,stride=1,padding=padding,bias=False,groups=in_channels),
          #nn.BatchNorm2d(in_channels),
          nn.ReLU(inplace=True),
        )

def pointwise(in_channels, out_channels):
    return nn.Sequential(
          nn.Conv2d(in_channels,out_channels,1,1,0,bias=False),
          #nn.BatchNorm2d(out_channels),
          nn.ReLU(inplace=True),
        )





class Decoder(nn.Module):
  def __init__(self):
    super().__init__()
    #self.activation = torch.nn.ReLU()
    self.sigmoid = torch.nn.Sigmoid()
    kernel_size = 3
    self.conv1 = nn.Sequential(
                depthwise(384, kernel_size),
                pointwise(384, 384))
    self.conv2 = nn.Sequential(
                depthwise(464, kernel_size),
                pointwise(464, 260))
    self.conv3 = nn.Sequential(
                depthwise(324, kernel_size),
                pointwise(324, 130))
    self.conv4 = nn.Sequential(
                depthwise(178, kernel_size),
                pointwise(178, 90))
    self.conv5 = nn.Sequential(
                depthwise(122, kernel_size),
                pointwise(122, 64))
    self.conv6 = pointwise(64, 1)


  def forward(self,x,dec1,dec2,dec3,dec4):
   
    x = self.conv1(x)
    x= F.interpolate(x,scale_factor=2, mode= 'bilinear', align_corners= True)
    
    #---------------
    x = torch.cat((x,dec4),1)
    x = self.conv2(x)
    x= F.interpolate(x,scale_factor=2, mode= 'bilinear', align_corners= True)
     #---------------
    x = torch.cat((x,dec3),1)
    x = self.conv3(x)
    x= F.interpolate(x,scale_factor=2, mode= 'bilinear', align_corners= True) 
     #---------------
    x = torch.cat((x,dec2),1)
    x = self.conv4(x)
    x= F.interpolate(x,scale_factor=2, mode= 'bilinear', align_corners= True) 
    #---------------
    x = torch.cat((x,dec1),1)
    x = self.conv5(x)
    x= F.interpolate(x,scale_factor=2, mode= 'bilinear', align_corners= True) 
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
        
        self.mvit = tmod.create_model('mobilevit_xs', pretrained=True, num_classes=0, global_pool='')   
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


def main() :
    

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    transformss = transforms.ToPILImage()
    model = EncoderDecoder(batch_size=8)
    num_epochs = 5
    BatchSize = 8
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(pytorch_total_params)
    PATH_SAVE = r'C:\Users\panagiotis_local\vscode\weights\nyu\mobilenet_small_HuberLoss_l2=0.0001_bsize=8.pth'
    model.load_state_dict(torch.load(PATH_SAVE))
    criterion = nn.HuberLoss()
    
    #criterion = torch.nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=0.000003, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)
    train_dataset = NyuDepth_train()
    train_load = DataLoader(dataset=train_dataset, batch_size=BatchSize, shuffle=True)
    model.cuda()
    model.train() 
    for epoch in range(num_epochs):  # loop over the dataset multiple times   
        t0 = time.time()
        running_loss = 0.0
        val_loss = 0.0
        for param_group in optimizer.param_groups:
            print(param_group['lr'])
        for i, data in enumerate(train_load, 0):
            optimizer.zero_grad()
            inputs = data['image'].to(device)
            outputs = data['depth'].to(device)
        
            out = model.forward(inputs)
            out = out.to(device)
            loss =criterion(out, outputs)
            #l_ssim = torch.clamp((1 - ssim(out, depth_n, val_range = 1000.0 / 10.0)) * 0.5, 0, 1)
            #loss = (1.0 * l_ssim.mean().item()) + (0.1 * l_depth)
            #losses.update(loss.data.item(), image.size(0))
            loss.backward()
            
            #print(loss)
            running_loss = running_loss + loss.item()
            #torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)

            optimizer.step()
        
        a = running_loss / i
        print(f'[{epoch + 1}] training_loss: {running_loss / i :.3f}')
        print('{} seconds'.format(time.time() - t0))
        torch.save(model.state_dict(), PATH_SAVE)
        print('updated training weights')
        scheduler.step()
       

  




if __name__ == '__main__':
    main()