#from gendata import GenData
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as ss
from cnnphase13 import NNModel, CNNTrain, CNNPredict

#Predict
def Predict():
	predict = CNNPredict()
	predict.SetDevice('cuda')
	predict.SetModel(NNModel, checkpoints=False)
	#predict.SetExpData('expdata3.npy', mask=620, square_root=True) # 872209
	#predict.SetExpData('expdata2.npy', mask=550, square_root=True) # 872210
	#predict.SetExpData('expdata2.npy', mask=1150, square_root=True) # 872245
	#predict.SetExpData('expdata2.npy', mask=1400, square_root=True) # 872246
	#predict.SetExpData('expdata2.npy', mask=1000, square_root=True) # 872247
	predict.SetExpData('expdata2.npy', mask=1000, square_root=True) # 872248
	predict.SetOutputFile('BFO_48.npy')
	predict.SetSupport('../support_all.npy')
	predict.SetTrainedNN("/scratch/ahmm1g15/MLTrainedNet/2023-06-02_15.15/CP150_2023-06-02_15.15.pth")
	#predict.SetTrainedNN("/scratch/ahmm1g15/BFO/Multi/2023-06-03_11.05/CP1100_2023-06-03_11.05.pth")
	#predict.SetTrainedNN("/scratch/ahmm1g15/BFO/872210/2023-06-05_11.49/CP1200_2023-06-05_11.49.pth")
	#predict.SetTrainedNN("/scratch/ahmm1g15/BFO/872245/2023-06-05_15.32/CP1200_2023-06-05_15.32.pth")
	#predict.SetTrainedNN("/scratch/ahmm1g15/BFO/872246/2023-06-05_17.22/CP1200_2023-06-05_17.22.pth")
	#predict.InitialiseWeights(nn.init.kaiming_normal_, mode='fan_in', nonlinearity='leaky_relu')
	predict.SetLRStepSize(5)
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