"""
########################################################################
# Copyright (c) 2022 Vicomtech (http://www.vicomtech.org)
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#
#    - Redistributions of source code must retain the above copyright
#     notice, this list of conditions and the following disclaimer.
#
#    - Redistributions in binary form must reproduce the above copyright
#     notice, this list of conditions and the following disclaimer in
#     the documentation and/or other materials provided with the
#     distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
#Description:
#  Plant data gateway
#
# Inital Author:
#      Alberto Di Maro  <adimaro@vicomtech.org>
# Collaborators:
#
########################################################################
"""

import cv2 
import numpy as np
import torch
from torch.autograd import Variable
import skimage 

import cv2
import numpy as np


def white_balance(img):
    # Convert the image to the YCrCb color space, which separates the
    # luminance (Y) channel from the chrominance (Cr and Cb) channels.
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    
    # Split the image into its channels
    channels = cv2.split(ycrcb)
    
    # Compute the mean value of the Cr channel
    mean_cr = np.mean(channels[1])
    
    # Compute the mean value of the Cb channel
    mean_cb = np.mean(channels[2])

    # Correct the mean values of the Cr and Cb channels
    y = np.array(channels[0], dtype = 'float64')
    cb = channels[1] * (128 / mean_cr)
    cr = channels[2] * (128 / mean_cb)
    new_channels = (y, cb, cr)

    # # Merge the corrected channels back into the image
    white_balanced = np.uint8(cv2.merge(new_channels))
    # Convert the image back to the RGB color space
    white_balanced = cv2.cvtColor(white_balanced, cv2.COLOR_YCrCb2BGR)
    
    #return white balanced image as float32
    return white_balanced.astype(np.uint8)

def oexp_process(image):
    ''' The function performs the processing related to the image on which
        the over exposure correction will be applied.
        It just switches the over exposure regions in under exposure ones exploiting the LAB color space.
        The output is an array in the RGBs color space.
    '''
    img_lab = cv2.cvtColor(image,cv2.COLOR_RGB2LAB)
    l = img_lab[:,:,0]
    a = img_lab[:,:,1]
    b = img_lab[:,:,2]
    l_norm = (l - np.min(l))/(np.max(l)-np.min(l))
    l_norm = 1 - l_norm
    l = np.uint8(l_norm*255)
    lab_back = np.stack([l,a,b], axis=2)
    rgb_back = cv2.cvtColor(lab_back,cv2.COLOR_LAB2RGB)
    return rgb_back

def preprocess(gpu_id, image, ue_degree, oe_degree, is_ue = True, wb = True):
    ''' The function preprocess the images in order to prepare them for the model application.
        The first processing is a White balancing followed by a Local Histogram Stretch. The white balancing is performed only if the wb flag is set to True
        Then the processing depends on the type of correction that we need to apply.
        The string parameters ue_degree, oe_degree and the boolean is_ue parameter takes the information about the type of correction and its degree.
        if is_ue is set to True, the correction concerns the under exposion and vice-versa. The variable degree then keeps track of the amount of correction (which is related to the model weights).
        Possible values of ue_degree/oe_degree are 'easy', 'medium', 'difficult', 'maximum', 'null'.
        If the degree is set to null the imahe won't be passed to the correction of the model and its tensor (b,w,h,c) is returned.
        If any degree is specified the image is first processed and then its tensor (b,w,h,c) is obtained.
    '''
    preprocessed_image = image
    if wb:
        preprocessed_image = white_balance(preprocessed_image)
    preprocessed_image = skimage.exposure.equalize_adapthist(np.uint8(preprocessed_image))
    preprocessed_image = np.uint8(preprocessed_image * 255)
    degree = ue_degree if is_ue else oe_degree
    if degree != 'null':     
        if not is_ue:
            preprocessed_image = oexp_process(preprocessed_image)
        preprocessed_image = (preprocessed_image - np.min(preprocessed_image))/(np.max(preprocessed_image) - np.min(preprocessed_image))
    preprocessed_image = np.array([preprocessed_image])
    preprocessed_image = preprocessed_image.transpose([0,3,1,2])
    preprocessed_tensor = torch.FloatTensor(preprocessed_image) if int(gpu_id) != -1 else torch.Tensor(preprocessed_image)
    preprocessed_tensor_ = Variable(preprocessed_tensor).cuda() if int(gpu_id) != -1 else Variable(preprocessed_tensor).to(torch.device('cpu'))
    return preprocessed_tensor_

def post_process(prep_image, ue_degree, oe_degree, is_ue = True):
    ''' The function postprocess the images in order to show them.
        The degree of correction applied is kept by the variable degree, that through the boolean parameter is_ue keeps also track if the correction was for under of over exposure.
        The initial post processing just conver the input tensor in an array of the shape (w,h,c)
        Then the post-processing depends on the type of correction applied.
        If the degree was 'null', the image is directly returned. If it was corrected for under exposion with a certain degree, it's converted in uint8: [0,255].
        If the not null degree refers to over exposure correction, it's post-processed with the same function for the pre-processing.
    '''
    degree = ue_degree if is_ue else oe_degree
    post_proc_image = np.array(prep_image[0].cpu())
    post_proc_image = np.transpose(post_proc_image, (1, 2, 0))
    if degree != 'null':
        post_proc_image = np.uint8(post_proc_image * 255)
        if not is_ue:
            post_proc_image = oexp_process(post_proc_image)
    return np.uint8(post_proc_image)

