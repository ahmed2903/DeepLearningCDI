from gendata import GenData
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as ss
from model import NNModel
from train import CNNTrain

# Generate data
def Gen():
	d = GenData()
	d.SetShape([32,32,32])
	d.SetN(12000)
	d.SetMorphology("hexprism")
	#d.SetMorphology("octahedron")
	#d.SetMorphology("monoclinic")
	d.GenShapeData()
	d.SaveData()

#Train
def Train():
	cnn = CNNTrain()
	cnn.SetDevice('cuda')
	cnn.SetInputData('fs_amps.npy', add_noise=None)
	cnn.SetTargetData('rs_objs.npy')
	cnn.SetModel(NNModel, checkpoints=False)
	cnn.SetValidSize(0.2)
	cnn.SplitData()
	cnn.SetBatchSize(64)
	cnn.LoadSplitTrain(loadtype='train')
	cnn.LoadSplitTrain(loadtype='test')
	cnn.InitialiseWeights(nn.init.kaiming_normal_, mode='fan_in', nonlinearity='leaky_relu')
	cnn.SetLRStepSize(5)
	cnn.AddLR(1e-3)
	cnn.AddLR(1e-4)
	cnn.AddLR(1e-5)
	cnn.AddLR(1e-6)
	cnn.AddGamma(0.975)
	cnn.AddGamma(0.999)
	cnn.AddGamma(0.96)
	cnn.AddGamma(0.9999)
	cnn.AddOptimiser(optim.Adadelta, eps=1e-6, weight_decay=0.0)
	cnn.AddOptimiser(optim.Adam, eps=1e-7, weight_decay=0.0)
	cnn.AddOptimiser(optim.Adam, eps=1e-8, weight_decay=0.0)
	cnn.AddOptimiser(optim.Adam, eps=1e-9, weight_decay=0.0)
	cnn.AddScheduler(ss.StepLR)
	cnn.AddScheduler(ss.StepLR)
	cnn.AddScheduler(ss.StepLR)
	cnn.AddScheduler(ss.StepLR)
	cnn.SetNEpochs(150)
	cnn.AddOpStep(25)
	cnn.AddOpStep(50)
	cnn.AddOpStep(50)
	cnn.AddOpStep(25)
	cnn.TrainNN(rs_pcc=False)
	cnn.SaveParameters()
	cnn.SaveLoss()
	cnn.PlotLoss()


Train()
