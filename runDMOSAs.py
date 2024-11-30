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
	#predict.SetExpData('expdataML.npy', mask=0.07, square_root=True) # R0.1_1 
	#predict.SetExpData('expdataML.npy', mask=0.08, square_root=True) # R0.1_2
	predict.SetExpData('expdataML.npy', mask=0.4, square_root=True) # R1_1 
	predict.SetOutputFile('R_01.npy')
	predict.SetSupport('NoSupport.npy')
	predict.SetTrainedNN("/scratch/ahmm1g15/MLTrainedNet/2023-06-02_15.15/CP150_2023-06-02_15.15.pth")
	predict.SetLRStepSize(5)
	predict.AddLR(5e-4)
	#predict.AddLR(1e-5)
	#predict.AddLR(1e-6)
	#predict.AddLR(1e-7)
	predict.AddGamma(0.95)
	#predict.AddGamma(0.99)
	#predict.AddGamma(0.96)
	#predict.AddGamma(0.99)
	predict.AddOptimiser(optim.Adam, eps=1e-8, weight_decay=0)
	#predict.AddOptimiser(optim.Adam, eps=1e-7, weight_decay=0)
	#predict.AddOptimiser(optim.Adam, eps=1e-8, weight_decay=0)
	#predict.AddOptimiser(optim.Adam, eps=1e-9, weight_decay=0)
	predict.AddScheduler(ss.StepLR)
	#predict.AddScheduler(ss.StepLR)
	#predict.AddScheduler(ss.StepLR)
	#predict.AddScheduler(ss.StepLR)
	predict.SetNEpochs(400)
	#predict.AddOpStep(75)
	predict.AddOpStep(400)
	#predict.AddOpStep(50)
	#predict.AddOpStep(50)
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