import warnings
warnings.filterwarnings("ignore")

import os
import math
import numpy as np
import torch
import pickle
import glob
import itertools
import operator
import functools

import sys
sys.path.insert(0, '..')
sys.path.insert(0, '../..')
sys.path.insert(0, '../../..')
import tools.user as user

conv_params     = ( [(0.0, [20, 20])] )
readout_params  = (0.0, [32, 32])
dRN = 0.4

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import SMEFTNet
from JetModel import JetModel
data_model = JetModel(0,0, events_per_parampoint=1)

from SMEFTNet import SMEFTNet
model = SMEFTNet(
    dRN=dRN,
    conv_params=conv_params,
    readout_params=readout_params,
    learn_from_phi=True,
    num_classes = 2,
    regression=True,
    ).to(device)

def loss( out, truth ):
    return ( (out[:,0] - truth[:,0])**2 + (out[:,1]-truth[:,1])**2 ).sum() 
