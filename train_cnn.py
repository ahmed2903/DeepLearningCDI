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
#mynn.SetInputData('/home/ahmm1g15/scratch/Run_DL/reci_intensity.npy')
#mynn.SetTargetDataReal('/home/ahmm1g15/scratch/Run_DL/real_obj.npy')
mynn.SetModel(NNPhase)
#mynn.SetValidSize(0.1)
#mynn.SplitData()
mynn.SetBatchSize(1)
#mynn.LoadSplitTrain(loadtype='train')
#mynn.LoadSplitTrain(loadtype='test')
mynn.InitializeWeightsPre()
mynn.SetLRStepSize(10)
mynn.SetLR(1e-4,2e-3)
mynn.SetMomentum(0.9)
mynn.SetGamma(0.5)
mynn.SetOptimiser1('ASGD')
mynn.SetOptimiser2('SGD')
mynn.SetScheduler1('StepLR')
mynn.SetScheduler2('StepLR')
mynn.SetNEpochs(1000)
#mynn.TrainNN()
mynn.one_Train('GoldInput.npy', mask=100)
mynn.PlotLoss()
#mynn.SaveParameters()
