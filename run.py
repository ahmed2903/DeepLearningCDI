from gendata import GenData
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as ss
from cnnphase08 import NNModel, CNNTrain, CNNPredict

# Generate data
def Gen():
	d = GenData()
	d.SetShape([32,32,32])
	d.SetN(10000)
	d.SetMorphology("hexprism")
	#d.SetMorphology("octahedron")
	#d.SetMorphology("monoclinic")
	d.GenShapeData()
	d.SaveData()


#Train
def Train():
	cnn = CNNTrain()
	cnn.SetDevice('cuda')
	cnn.SetInputData('fs_amps.npy', add_noise=True) # noise is a normal random distribution with its mean at mu/2
	cnn.SetTargetData('rs_objs.npy')
	cnn.GenMoreData(n = 3) # the number of data points is muliplied by n and undergo random rotations
	cnn.SetModel(NNModel, momentum = 0.7,  checkpoints=False)
	cnn.SetValidSize(0.1)
	cnn.SplitData()
	cnn.SetBatchSize(32)
	cnn.LoadSplitTrain(loadtype='train')
	cnn.LoadSplitTrain(loadtype='test')
	#cnn.LoadWeights("CP.pth")
	cnn.InitialiseWeights(nn.init.kaiming_normal_, mode='fan_in', nonlinearity='leaky_relu')
	#cnn.InitialiseWeights(nn.init.xavier_normal_)
	cnn.SetLRStepSize(10)
	cnn.AddLR(1e-4)
	cnn.AddLR(1e-6)
	#cnn.SetGamma(0.75)
	cnn.AddGamma(0.9)
	cnn.AddGamma(0.95)
	cnn.AddOptimiser(optim.ASGD)
	cnn.AddOptimiser(optim.Adam, amsgrad=True, eps=1e-8)
	#cnn.AddOptimiser(optim.SGD, momentum=0.9)
	#cnn.AddOptimiser(optim.RMSprop, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0.9, centered=False)
	cnn.AddScheduler(ss.StepLR)
	cnn.AddScheduler(ss.StepLR)
	cnn.SetNEpochs(50)
	cnn.SetOpStep(10)
	cnn.TrainNN(rs_pcc=False)
	cnn.SaveParameters()
	cnn.SaveLoss()
	cnn.PlotLoss()


#Predict
def Predict():
	predict = CNNPredict()
	predict.SetDevice('cuda')
	#predict.SetDevice('cpu')
	predict.SetModel(NNModel, checkpoints=False)
	predict.SetExpData('expdata.npy', mask=680, square_root=True)
	predict.SetTrainedNN("CP145_2022-07-09_23.07.pth")
	predict.SetOutputFile('output.npy')
	predict.SetLRStepSize(10)
	predict.AddLR(1e-3)
	predict.AddLR(1e-6)
	predict.AddLR(1e-8)
	#predict.SetGamma(0.92)
	predict.AddGamma(0.90)
	predict.AddGamma(0.95)
	predict.AddGamma(0.99)
	predict.AddOptimiser(optim.Adam, eps=1e-6, weight_decay=0)
	predict.AddOptimiser(optim.Adam, eps=5e-7, weight_decay=0)
	predict.AddOptimiser(optim.Adam, eps=1e-7, weight_decay=0)
	predict.AddScheduler(ss.StepLR)
	predict.AddScheduler(ss.StepLR)
	predict.AddScheduler(ss.StepLR)
	predict.SetNEpochs(5000)
	predict.SetOpStep(500)
	predict.TransferPredict()
	predict.SaveParameters(training=False)
	predict.SaveTrainLoss()
	predict.PlotLoss(training=False)


#Gen()
#Train()
Predict()

