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

def get_HSI(img):
    ''' The function returns the image in the HSV colorspace '''
    return cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    
def get_gray_scale(img):
    ''' The function returns the image in grayscale  '''
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) 

def laplacian_filter(img, k=3):
    ''' The function returns the Laplacian of the image '''
    gray_img = get_gray_scale(img)
    laplacian = cv2.Laplacian(gray_img, cv2.CV_16S, ksize=k)
    s = cv2.convertScaleAbs(laplacian)
    return s

def get_saturation(hsi_img):
    ''' The function extracts the Saturation channel of an HSV image '''
    return hsi_img[:,:,1]

def get_intensity(hsi_img):
    ''' The function extracts the Value channel of an HSV image '''
    return hsi_img[:,:,2]

def min_max_norm(img):
    ''' The function rescales the pixels between [0,1] '''
    return (img-np.min(img))/(np.max(img)-np.min(img))

def final_mask(img):
    ''' The function returns a tuple of three masks.
    The first one for the detection of the over exposed regions,
    the second for the under exposed ones and the last one for the goodly exposed ones '''
    # -- Getting contrast , intensity and desaturation components --
    img = img
    hsi_img = get_HSI(img)
    hsi_img = min_max_norm(hsi_img)
    contrast_image = min_max_norm(laplacian_filter(img))
    saturation = get_saturation(hsi_img)
    intensity = get_intensity(hsi_img)
    desaturation = 1 - saturation
    # -- initialize the masks to zeros
    mask_intensity_low = np.zeros(intensity.shape)
    mask_intensity_high = np.zeros(intensity.shape)
    mask_desaturation_high = np.zeros(desaturation.shape)
    # -- masks generation
    mask_contrast = np.zeros(contrast_image.shape)
    mask_contrast[contrast_image > 0.9] = 1
    mask_intensity_high[intensity > 0.9] = 1
    mask_desaturation_high[desaturation > 0.9] = 1
    mask_intensity_low[intensity < 0.1] = 1
    mask_low = mask_intensity_low
    mask_high = np.logical_and(mask_desaturation_high, mask_intensity_high)
    final_mask_high = np.logical_or(mask_high, mask_contrast)
    final_mask_low = np.logical_or(mask_low, mask_contrast)
    final_mask_good = np.logical_not(np.logical_or(final_mask_high, final_mask_low))
    return final_mask_high, final_mask_low, final_mask_good

# --- This is the only funciton that needs to be imported ---
def mask_creation_hsi(image):
    ''' The function returns thre three masks in float32 with shape [W,H,C=3] '''
    mask_high, mask_low, mask_good = final_mask(image)
    mask_high = np.array(mask_high, dtype = 'float32')
    mask_low = np.array(mask_low, dtype = 'float32')
    mask_good = np.array(mask_good, dtype = 'float32')
    mask_high = np.stack([mask_high]*3,axis=2)
    mask_low = np.stack([mask_low]*3,axis=2)
    mask_good = np.stack([mask_good]*3,axis=2)
    return mask_high, mask_low, mask_good 