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

import skimage.io as io
import sys
sys.path.insert(0, 'SCI')
import os
import cv2
import numpy as np
from processing.processing import preprocess, post_process
from segmentation.segmentation import mask_creation_hsi
from blending.blending_morethan2 import blending_laplacian
from model_load.model_load import *

def hdr_to_8bit(img):
    ''' The function has the purpose of performing a linear mapping to bring the output of the pipeline in the range [0,255]'''
    # Scale the values in the image to the range [0, 1]
    img = img / np.max(img)
    
    # Compute the cumulative distribution function (CDF) of the image
    cdf = np.cumsum(np.histogram(img, bins=256)[0]) / img.size
    img_cdf = np.interp(img, np.linspace(0, 1, 256), cdf)
    
    # Use the CDF to map the values in the image to 8-bit integers
    img_8bit = np.interp(img_cdf, cdf, np.linspace(0, 255, 256).round().astype(np.uint8))
    return img_8bit

def load_model(model_root, gpu_id, ue_degree, oe_degree):
    # Model paths
    model_uexp_path = os.path.join(model_root, ue_degree + '.pt')
    model_oexp_path = os.path.join(model_root, oe_degree + '.pt')
    # Models loading
    model_uexp = model_loading(gpu_id, model_uexp_path, ue_degree, oe_degree, is_ue = True)
    if model_uexp:
        model_uexp.eval()
    model_oexp = model_loading(gpu_id, model_oexp_path, ue_degree, oe_degree, is_ue = False)
    if model_oexp:
        model_oexp.eval()
    return model_uexp, model_oexp

def inference(image, model_uexp, model_oexp, wb = False, gpu_id = 0, ue_degree = 'maximum', oe_degree = 'easy'):
    # Preprocess the image for model application
    uexp_tensor = preprocess(gpu_id, image, ue_degree, oe_degree, is_ue = True, wb = wb)
    oexp_tensor = preprocess(gpu_id, image, ue_degree, oe_degree, is_ue = False, wb = wb)
    # Model application
    uexp_tensor = model_application(model_uexp, uexp_tensor, ue_degree, oe_degree, is_ue = True)
    oexp_tensor = model_application(model_oexp, oexp_tensor, ue_degree, oe_degree, is_ue = False)
    # Post-process the images
    r_uexp = post_process(uexp_tensor, ue_degree, oe_degree, is_ue = True)
    r_oexp = post_process(oexp_tensor, ue_degree, oe_degree, is_ue = False)

    mask_high,mask_low,mask_good = mask_creation_hsi(image)
    output = blending_laplacian(r_uexp, r_oexp, image, mask_low, mask_high, mask_good)
    output = hdr_to_8bit(output)
    output = np.uint8(output)
    return output


def pipeline(image, model_root, wb = False, gpu_id = 0, ue_degree = 'maximum', oe_degree = 'easy'):
    ''' The function performs the pipeline to correct both the over and the under exposure.
        PARAMETERS: 
        image(ndarray): uint8 array representing the input image
        gpu_id(int): the ID of the processing unit. (-1 for CPU)
        model_root(str): the root on which the model is located (./SCI/weights) 
        wb(Boolean): used to specify if the white balancing preprocessing is needed or not by default it's set to False. (This value could affect significatevely the performance of the pipeline)
        ue_degree(str): the amount of correction for the under exposure. Possible values: ['maximum', 'difficult', 'medium', 'easy', 'null']
        oe_degree(str): the amount of correction for the over exposure. Possible values: ['maximum', 'difficult', 'medium', 'easy', 'null']
            
    '''
    # Model paths
    model_uexp_path = os.path.join(model_root, ue_degree + '.pt')
    model_oexp_path = os.path.join(model_root, oe_degree + '.pt')
    # Models loading
    model_uexp = model_loading(gpu_id, model_uexp_path, ue_degree, oe_degree, is_ue = True)
    if model_uexp:
        model_uexp.eval()
    model_oexp = model_loading(gpu_id, model_oexp_path, ue_degree, oe_degree, is_ue = False)
    if model_oexp:
        model_oexp.eval()
    # Preprocess the image for model application
    uexp_tensor = preprocess(gpu_id, image, ue_degree, oe_degree, is_ue = True, wb = wb)
    oexp_tensor = preprocess(gpu_id, image, ue_degree, oe_degree, is_ue = False, wb = wb)
    # Model application
    uexp_tensor = model_application(model_uexp, uexp_tensor, ue_degree, oe_degree, is_ue = True)
    oexp_tensor = model_application(model_oexp, oexp_tensor, ue_degree, oe_degree, is_ue = False)
    # Post-process the images
    r_uexp = post_process(uexp_tensor, ue_degree, oe_degree, is_ue = True)
    r_oexp = post_process(oexp_tensor, ue_degree, oe_degree, is_ue = False)

    mask_high,mask_low,mask_good = mask_creation_hsi(image)
    output = blending_laplacian(r_uexp, r_oexp, image, mask_low, mask_high, mask_good)
    output = hdr_to_8bit(output)
    output = np.uint8(output)
    return output
    


def test():
    imgs_path = './imgs'
    for image_name in os.listdir(imgs_path):
        image = io.imread(os.path.join(imgs_path, image_name))
        model_root = 'SCI/weights'
        gpu_id = 0

        # degree of correction
        # if both are null only Local Scale Histogram Stretch is applied
        ue_degree = 'maximum'
        oe_degree = 'easy'
        output = pipeline(image = image, gpu_id = gpu_id, model_root =  model_root, wb = False, ue_degree = ue_degree, oe_degree = oe_degree)

        save_path = './results'
        os.makedirs(save_path, exist_ok=True)
        # output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
        # cv2.imwrite(os.path.join(save_path, image_name), output)
        io.imsave(os.path.join(save_path, image_name), output)

if __name__=='__main__':
    test()