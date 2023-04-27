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

import sys
import torch
sys.path.insert(0, '../SCI')
from model import Finetunemodel as model_

def model_loading(gpu_id, model_path, ue_degree, oe_degree, is_ue = True):
    ''' The function loads the model on the device specified in the parameter gpu_id from the model_path.
        The boolean parameter is_ue keeps track of the type of correction. If it's set to True the correction refers to the under exposure or vice-versa.
        The variable degree then keeps the amount of correction which is related to the weights of the model. 
        The possible values must be: 'easy', 'medium', 'difficult', 'maximum', 'null'. If the degree is set to None the model returns None, otherwise
        it returns the model chosen.
    '''
    degree = ue_degree if is_ue else oe_degree
    if degree != 'null':
        model = model_(int(gpu_id), model_path)
        model = model.cuda() if int(gpu_id) != -1 else model.to(torch.device('cpu'))
        return model
    else:
        return None

def model_application(model, tensor, ue_degree, oe_degree, is_ue = True):
    ''' The function takes the model loaded and the preprocessed image in order to infer its results.
        If the degree of correction specified is 'null' the function returns the tensor in input, otherwise it returns the tensor after the correction
    '''
    degree = ue_degree if is_ue else oe_degree
    if degree != 'null':
        with torch.no_grad():
            i_uexp,r_uexp = model(tensor)
    else:
        r_uexp = tensor
    return r_uexp 
