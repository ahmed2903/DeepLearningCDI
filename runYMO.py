#from gendata import GenData
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as ss
from cnnphase12 import NNModel, CNNTrain, CNNPredict

#Predict
def Predict():
	predict = CNNPredict()
	predict.SetDevice('cuda')
	predict.SetModel(NNModel, checkpoints=False)
	#predict.SetExpData('expdata_ML.npy', mask=260, square_root=True) # 897014 - 110
	#predict.SetExpData('expdata_ML2.npy', mask=380, square_root=True) # 896998 - 110
	#predict.SetExpData('expdata_ML.npy', mask=650, square_root=True) # 896387 - 112 NEXT
	predict.SetExpData('expdata_ML.npy', mask=190, square_root=True) # 896781 - 112
	#predict.SetExpData('expdata_ML.npy', mask=240, square_root=True) # 896408 - 112
	#predict.SetExpData('expdata_ML.npy', mask=190, square_root=True) # 8977 - 110
	predict.SetOutputFile('YMO.npy')
	predict.SetSupport('supportML.npy')
	predict.SetTrainedNN("/scratch/ahmm1g15/MLTrainedNet/2023-06-02_15.15/CP150_2023-06-02_15.15.pth")
	#predict.InitialiseWeights(nn.init.kaiming_normal_, mode='fan_in', nonlinearity='leaky_relu')
	predict.SetLRStepSize(2)
	predict.AddLR(5e-4)
	predict.AddLR(5e-5)
	predict.AddLR(1e-6)
	predict.AddLR(1e-7)
	predict.AddGamma(0.975)
	predict.AddGamma(0.99)
	predict.AddGamma(0.96)
	predict.AddGamma(0.99)
	predict.AddOptimiser(optim.Adam, eps=1e-6, weight_decay=0)
	predict.AddOptimiser(optim.Adam, eps=1e-7, weight_decay=0)
	predict.AddOptimiser(optim.Adam, eps=1e-8, weight_decay=0)
	predict.AddOptimiser(optim.Adam, eps=1e-9, weight_decay=0)
	predict.AddScheduler(ss.StepLR)
	predict.AddScheduler(ss.StepLR)
	predict.AddScheduler(ss.StepLR)
	predict.AddScheduler(ss.StepLR)
	predict.SetNEpochs(1200)
	predict.AddOpStep(75)
	predict.AddOpStep(1025)
	predict.AddOpStep(50)
	predict.AddOpStep(50)
	predict.TransferPredict(AMP=False)
	predict.SaveParameters(training=False)
	predict.SaveTrainLoss()
	predict.SaveParameters(training = False)
	predict.PlotLoss()

#ValidationSearch()
#Gen()
#HyperSpaceSearch()
#Train()
Predict()
#PredictSearchParams()