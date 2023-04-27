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
import matplotlib.pyplot as plt


def gaussian_pyramid(img, num_levels):
    ''' The function returns a list of the Gaussian pyramid with a specified number of levels'''
    lower = img.copy()
    gaussian_pyr = [lower]
    for i in range(num_levels):
        lower = cv2.pyrDown(lower)
        gaussian_pyr.append(np.float32(lower))
    return gaussian_pyr


def laplacian_pyramid(gaussian_pyr):
    ''' The function returns a list of the Laplacian pyramid starting from the Gaussian pyramid'''
    laplacian_top = gaussian_pyr[-1]
    num_levels = len(gaussian_pyr) - 1
    laplacian_pyr = [laplacian_top]
    for i in range(num_levels,0,-1):
        size = (gaussian_pyr[i - 1].shape[1], gaussian_pyr[i - 1].shape[0])
        gaussian_expanded = cv2.pyrUp(gaussian_pyr[i], dstsize=size)
        laplacian = np.subtract(gaussian_pyr[i-1], gaussian_expanded)
        laplacian_pyr.append(laplacian)
    return laplacian_pyr


def blend(laplacian_A,laplacian_B,laplacian_C,mask_pyr1, mask_pyr2, mask_pyr3):
    '''' The function performs the blending at each level of the Laplacian pyramids for the three images.
         It takes 6 parameters that must respect the following order:
         Laplacian A is the Laplacian pyramid of the image at which corresponds the pyramidal mask 'mask_pyr1' parameter.
         This means that 'mask_pyr1 is the Gaussian pyramid of the mask which should be applied elementwise to 'laplacian A'.
         The same goes for the couples (laplacian B, mask_pyr2),(laplacian C, mask_pyr3).
         The output is a list of blending at each pyramid level.
    '''
    LS = []
    for la,lb,lc,mask1,mask2,mask3 in zip(laplacian_A,laplacian_B,laplacian_C,mask_pyr1, mask_pyr2, mask_pyr3):
        ls = la * mask1 + lb * mask2 + lc * mask3
        LS.append(ls)
    return LS


def reconstruct(laplacian_pyr):
    ''' The function takes the input from the 'blend' function and returns a list of reconstructed images
        at different resolutions.
        The last element of the list is the reconstructed image at the resolution of the input images'''
    laplacian_top = laplacian_pyr[0]
    laplacian_lst = [laplacian_top]
    num_levels = len(laplacian_pyr) - 1
    for i in range(num_levels):
        size = (laplacian_pyr[i + 1].shape[1], laplacian_pyr[i + 1].shape[0])
        laplacian_expanded = cv2.pyrUp(laplacian_top, dstsize=size)
        laplacian_top = cv2.add(laplacian_pyr[i+1], laplacian_expanded)
        laplacian_lst.append(laplacian_top)
    return laplacian_lst


def blending_laplacian(r_uexp, r_oexp, image, mask_low, mask_high, mask_good, num_levels = 7):
    ''' The function wraps all the previous functions to perform the Laplacian blending pipeline.
        Firstly, the three Laplacian pyramids are generated in the following order:
        1) Laplacians of the image corrected for the under exposure
        2) Laplacians of the image corrected for the over exposure
        3) Laplacians of the original image
        Then, the Gaussian pyramids of the masks are obtained following:
        1) Gaussians of the mask that detects the under exposure
        2) Gaussians of the mask that detects the over exposure
        3) Gaussians of the masks which detects the good exposure 
        Then the blend and reconstruct functions are applied in order to get in output the blended result.
    '''
    # ------- The following order and names are for simplicity, what's really matter is the fact that the (i-th) parameter is associated to the (i-th + 3) one -------
    # 1st Laplacian 
    gaussian_pyr_1 = gaussian_pyramid(r_uexp, num_levels)
    laplacian_pyr_1 = laplacian_pyramid(gaussian_pyr_1)
    # 2nd Laplacian
    gaussian_pyr_2 = gaussian_pyramid(r_oexp, num_levels)
    laplacian_pyr_2 = laplacian_pyramid(gaussian_pyr_2)
    # 3rd Laplacian 
    gaussian_pyr_3 = gaussian_pyramid(image, num_levels)
    laplacian_pyr_3 = laplacian_pyramid(gaussian_pyr_3)
    # 1) Gaussian of mask_low
    mask_pyr_final1 = gaussian_pyramid(mask_low, num_levels)
    mask_pyr_final1.reverse()
    # 2) Gaussian of mask_high
    mask_pyr_final2 = gaussian_pyramid(mask_high, num_levels)
    mask_pyr_final2.reverse()
    # 3) Gaussian of mask_good
    mask_pyr_final3 = gaussian_pyramid(mask_good, num_levels)
    mask_pyr_final3.reverse()

    # Blending togheter
    add_laplace = blend(laplacian_pyr_1,laplacian_pyr_2,laplacian_pyr_3, mask_pyr_final1, mask_pyr_final2, mask_pyr_final3)
    # Reconstruct
    final  = reconstruct(add_laplace)
    output = np.array(final[num_levels], dtype='float32')
    return output  