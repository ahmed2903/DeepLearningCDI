import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as ss
from cnnphase09 import NNModel, CNNPredict

def Predict():
    predict = CNNPredict()
    predict.SetDevice('cuda')
    #predict.SetDevice('cpu')
    predict.SetModel(NNModel, checkpoints=False)
    predict.LoadDiffArrays(diffs = ['ExpDataML_111.npy', 'ExpDataML_110.npy','ExpDataML_212.npy'], mask_vals = [0.35,2.1,11.6])
    predict.SetTrainedNN("CP100_2022-09-16_01.49.pth")
    #predict.InitialiseWeights(nn.init.kaiming_normal_, mode='fan_in', nonlinearity='leaky_relu')
    predict.SetOutputFile('YMO.npy')
    predict.SetLRStepSize(100)
    predict.AddLR(1e-3)
    predict.AddLR(5e-5)
    predict.AddLR(1e-6)
    predict.AddLR(1e-7)
    #predict.SetGamma(0.92)
    predict.AddGamma(0.95)
    predict.AddGamma(0.92)
    predict.AddGamma(0.92)
    predict.AddOptimiser(optim.Adam, eps=1e-7)
    predict.AddOptimiser(optim.Adam, eps=1e-8, weight_decay=0)
    predict.AddOptimiser(optim.Adagrad, eps=1e-9, weight_decay=0)
    predict.AddScheduler(ss.StepLR)
    predict.AddScheduler(ss.StepLR)
    predict.AddScheduler(ss.StepLR)
    predict.SetNEpochs(400)
    #predict.SetOpStep(500)
    predict.AddOpStep(100)
    predict.AddOpStep(200)
    predict.AddOpStep(100)
    predict.TransferPredict(AMP=False)
    predict.SaveParameters(training=False)
    predict.SaveTrainLoss()
    predict.PlotLoss(training=False)



Predict()
