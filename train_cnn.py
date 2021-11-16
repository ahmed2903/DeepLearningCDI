import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchsummary import summary
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable

import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable

#import cupy as cp

from torch.utils.dlpack import to_dlpack
from torch.utils.dlpack import from_dlpack

from cnn import *



mynn = CNNTrain()
mynn.SetDeviceType("cpu")
mynn.SetInputData('reci_intes.npy')
mynn.SetTargetData('real_obj.npy')
mynn.SetDimensions()
mynn.SetModel(NNModel)
mynn.SetValidSize(0.1)

mynn.SplitData()
mynn.SetBatchSize(192)

mynn.LoadSplitTrain(loadtype='train')
mynn.LoadSplitTrain(loadtype='test')
mynn.SetLRStepSize(10)
mynn.SetLR(1e-2)
mynn.SetMomentum(0.9)
mynn.SetGamma(0.1)

mynn.SetOptimiser()
mynn.SetScheduler()
mynn.SetNEpochs(50)
mynn.TrainNN()
mynn.PlotLoss()
mynn.SaveParameters()
