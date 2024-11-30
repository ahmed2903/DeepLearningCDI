#from gendata import GenData
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as ss
from cnnphase11 import NNModel, CNNTrain, CNNPredict

# Generate data
def Gen():
	d = GenData()
	d.SetShape([32,32,32])
	d.SetN(6200)
	d.SetMorphology("hexprism")
	#d.SetMorphology("octahedron")
	#d.SetMorphology("monoclinic")
	d.GenShapeData()
	d.SaveData()

# HyperParameter search in Training 
def HyperSpaceSearch():
	DOs = [0.0,0.2,0.3]
	BSs = [32, 16]
	SSs = [5, 10]
	LRs = [1e-3, 5e-4, 1e-4, 5e-5, 1e-5]
	
	
	for do in DOs:
		cnn = CNNTrain()
		cnn.SetDevice('cuda')
		cnn.SetInputData('fs_amps.npy', add_noise='Poisson')
		cnn.SetTargetData('rs_objs.npy')
		cnn.SetModel(NNModel, checkpoints=False, dropoutval = do)
		cnn.SetValidSize(0.2)
		cnn.SplitData()

		for bs in BSs:
			cnn.SetBatchSize(bs)
			cnn.LoadSplitTrain(loadtype='train')
			cnn.LoadSplitTrain(loadtype='test')	
			for sss in SSs:
				cnn.SetLRStepSize(sss)
				for lr in LRs:
					cnn.InitialiseWeights(nn.init.kaiming_normal_, mode='fan_in', nonlinearity='leaky_relu')
					cnn.train_loss = []
					cnn.valid_loss = []
					cnn.amp_loss_train = []
					cnn.pha_loss_train = []
					cnn.diff_loss_train = []
					cnn.amp_loss_val = []
					cnn.pha_loss_val = []
					cnn.diff_loss_val = []
					cnn.AddLR(lr)
					cnn.AddGamma(0.95)
					cnn.AddOptimiser(optim.Adagrad)
					cnn.AddScheduler(ss.StepLR)
					cnn.SetNEpochs(50)
					cnn.AddOpStep(50)
					cnn.TrainNN(rs_pcc=False)
					cnn.SaveParameters()
					cnn.SaveLoss()
					cnn.PlotLoss()


def ValidationSearch():
	LRGs = [0.1,0.2,0.3]
	MMs = [0.9, 0.99]
	PHs = [3,4]
	OPs = [optim.SGD, optim.Adam, optim.RMSprop]
	SSs = [100, 200]
	LRs = [5e-4, 1e-4, 5e-5, 1e-5]
	GMs = [0.85, 0.9, 0.95]
	LSs = ['pcc_loss', 'log_cosh_loss']


	# LRGs = [0.1]
	# MMs = [0.7]
	# PHs = [3]
	# OPs = [optim.SGD]
	# SSs = [100]
	# LRs = [1e-3]
	# GMs = [0.9]
	# LSs = ['pcc_loss']


	predict = CNNPredict()
	predict.SetDevice('cuda')
	for lrg in LRGs:
		for mm in MMs:
			for ph in PHs:
				predict.SetModel(NNModel,  LRLUGrad = lrg, momentum = mm, phasemin = -ph, phasemax = ph, checkpoints=False, dropoutval = 0)
				predict.SetExpData('/scratch/ahmm1g15/MLCode/YMO_DiffSim_002.npy', mask=1, square_root=True, validation = True)
				predict.SetTrainedNN("/scratch/ahmm1g15/MLCode/Networks_conv3d_do_modulo/2023-04-13_18.09/CP50_2023-04-13_18.09.pth")
				predict.LoadValidationFile("/scratch/ahmm1g15/MLCode/YMO_Object_002.npy")
				predict.SetOutputFile('YMO_Val.npy')
				for sss in SSs: 
					predict.SetLRStepSize(sss)
					for lr in LRs: 
						predict.AddLR(lr)
						for gm in GMs:
							predict.AddGamma(gm)
							for op in OPs:
								predict.AddOptimiser(optimiser=op)
								predict.AddScheduler(ss.StepLR)
								predict.SetNEpochs(500)
								predict.AddOpStep(2000)
								predict.SetSWSigma(4)
								predict.SetSWThreshold(0.05)
								predict.SetSWCyclelength(40)
								for ls in LSs:
									predict.TransferPredict(AMP=False, validation = True, lossfunc=ls)
									predict.SaveParameters(training=False)
									predict.SaveTrainLoss()
									predict.PlotLoss(validation=True)

#Train
def Train():
	cnn = CNNTrain()
	cnn.SetDevice('cuda')
	cnn.SetInputData('fs_amps.npy', add_noise='Poisson')
	cnn.SetTargetData('rs_objs.npy')
	cnn.SetModel(NNModel, checkpoints=False)
	cnn.SetValidSize(0.2)
	cnn.SplitData()
	cnn.SetBatchSize(32)
	cnn.LoadSplitTrain(loadtype='train')
	cnn.LoadSplitTrain(loadtype='test')
	#cnn.LoadWeights("CP.pth")
	cnn.InitialiseWeights(nn.init.kaiming_normal_, mode='fan_in', nonlinearity='leaky_relu')
	#cnn.InitialiseWeights(nn.init.xavier_normal_)
	cnn.SetLRStepSize(5)
	cnn.AddLR(1e-3)
	cnn.AddLR(1e-4)
	cnn.AddLR(1e-5)
	#cnn.SetGamma(0.9)
	cnn.AddGamma(0.95)
	cnn.AddGamma(0.925)
	cnn.AddGamma(0.95)
	cnn.AddOptimiser(optim.Adagrad, eps=1e-5,weight_decay=0.0)
	cnn.AddOptimiser(optim.Adam, eps=1e-7)
	cnn.AddOptimiser(optim.Adagrad, eps=1e-8)
	#cnn.AddOptimiser(optim.SGD, momentum=0.9)
	#cnn.AddOptimiser(optim.RMSprop, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0.9, centered=False)
	cnn.AddScheduler(ss.StepLR)
	cnn.AddScheduler(ss.StepLR)
	cnn.AddScheduler(ss.StepLR)
	cnn.SetNEpochs(50)
	#cnn.SetOpStep(50)
	cnn.AddOpStep(10)
	cnn.AddOpStep(30)
	cnn.AddOpStep(10)
	cnn.TrainNN(rs_pcc=False)
	cnn.SaveParameters()
	cnn.SaveLoss()
	cnn.PlotLoss()

#Predict
def Predict():

	sss = 100
	ls = 'pcc_loss'

	predict = CNNPredict()
	predict.SetDevice('cuda')
	predict.SetModel(NNModel, checkpoints=False)
	predict.SetExpData('expdata3.npy', mask=630, square_root=True)
	predict.SetOutputFile('BFOwSupp.npy')
	predict.SetSupport('supportSW09.npy')
	predict.SetTrainedNN("/scratch/ahmm1g15/MLTrainedNet/2023-06-02_15.15/CP150_2023-06-02_15.15.pth")
	#predict.SetTrainedNN("/scratch/ahmm1g15/MLCode/2023-05-31_21.16/CP100_2023-05-31_21.16.pth")
	#predict.InitialiseWeights(nn.init.kaiming_normal_, mode='fan_in', nonlinearity='leaky_relu')
	predict.SetLRStepSize(5)
	predict.AddLR(1e-3)
	predict.AddLR(5e-5)
	predict.AddLR(1e-6)
	predict.AddLR(1e-7)
	#predict.AddLR(1e-8)
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
	predict.SetNEpochs(1100)
	predict.AddOpStep(50)
	predict.AddOpStep(1000)
	predict.AddOpStep(25)
	predict.AddOpStep(25)
	predict.TransferPredict(AMP=False, validation = False, lossfunc=ls)
	predict.SaveParameters(training=False)
	predict.SaveTrainLoss()
	predict.SaveParameters(training = False)
	predict.PlotLoss(validation=False)

def PredictSearchParams():

	OPs = [optim.SGD, optim.Adam, optim.Adagrad, optim.RMSprop]
	SSs = [100, 200, 400]
	LRs = [1e-3, 1e-4, 1e-5]
	GMs = [0.85, 0.9, 0.95]
	LSs = ['pcc_loss', 'log_cosh_loss']

	for sss in SSs:
		for lr in LRs:
			for gm in GMs:
				for op in OPs:
					for ls in LSs:
						predict = CNNPredict()
						predict.SetDevice('cuda')
						predict.SetModel(NNModel, checkpoints=False)
						predict.SetExpData('expdata.npy', mask=620, square_root=True)
						predict.SetOutputFile('BFOwSupp.npy')
						predict.SetSupport('support.npy')
						predict.InitialiseWeights(nn.init.kaiming_normal_, mode='fan_in', nonlinearity='leaky_relu')
						predict.SetLRStepSize(sss)
						predict.AddLR(lr)
						predict.AddGamma(gm)
						predict.AddOptimiser(optimiser=op)
						predict.AddScheduler(ss.StepLR)
						predict.SetNEpochs(50)
						predict.AddOpStep(1000)
						predict.TransferPredict(AMP=False, validation = False, lossfunc=ls)
						predict.SaveParameters(training=False)
						predict.SaveTrainLoss()
						predict.SaveParameters(training = False)
						predict.PlotLoss(validation=False)



#ValidationSearch()
#Gen()
#HyperSpaceSearch()
#Train()
Predict()
#PredictSearchParams()