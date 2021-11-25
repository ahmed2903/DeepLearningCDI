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
mynn.SetDeviceType('cuda')
mynn.SetInputData('$HOME/scratch/Run_DL/reci_intensity.npy')
mynn.SetTargetDataReal('$HOME/scratch/Run_DL/real_obj.npy')
mynn.SetModel(NNModel)
mynn.SetValidSize(0.1)
mynn.SplitData()
mynn.SetBatchSize(5)
mynn.LoadSplitTrain(loadtype='train')
mynn.LoadSplitTrain(loadtype='test')
mynn.SetLRStepSize(10)
mynn.SetLR(1e-4)
mynn.SetMomentum(0.9)
mynn.SetGamma(0.1)
mynn.SetOptimiser1('SGD')
mynn.SetOptimiser2('SGD')
mynn.SetScheduler1('StepLR')
mynn.SetScheduler2('StepLR')
mynn.SetNEpochs(100)
mynn.TrainNN()
mynn.PlotLoss()
mynn.SaveParameters()
