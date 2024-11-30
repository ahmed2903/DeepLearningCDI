#from gendata import GenData
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as ss
from cnnphase13 import NNModel, CNNTrain, CNNPredict
import os 
import numpy as np

def cuboid_support(sx = 80, sy = 80, sz = 40, x = 112, y = 112, z = 112):

	support = np.zeros((x,y,z), dtype = complex)

	x1 = (x-sx)//2
	x2 = x1 + sx
	y1 = (y - sy)//2
	y2 = y1 + sy
	z1 = (z - sz)//2
	z2 = z1 + sz
	support[x1:x2,y1:y2,z1:z2] = 1 + 1j*0

	return support

#Predict
def Predict(mask_val):
	predict = CNNPredict()
	predict.SetDevice('cuda')
	predict.SetModel(NNModel, checkpoints=False)
	predict.SetExpData('expdataML.npy', mask=mask_val, square_root=True) # 0V
	predict.SetOutputFile('R_01.npy')
	predict.SetSupport('../NoSupport.npy')
	predict.SetTrainedNN("/scratch/ahmm1g15/MLTrainedNet/2023-06-02_15.15/CP150_2023-06-02_15.15.pth")
	predict.SetLRStepSize(5)
	predict.AddLR(5e-4)
	predict.AddLR(1e-5)
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


names = {'0.1' : 15.0, 
		 '0.2' : 0.08, 
		 '0.3' : 0.05, 
		 '1.5' : 0.03,
		 '2.0' : 0.03,
		 '5.0' : 0.03, 
		 '-0.1' : .07,
		 '-0.2' : .12, 
		 '-0.3' : .06,
		 '-1.5' : .04,
		 '-2.0' : .03,
		 '-3.0' : .03, 
		 '-5.0' : .03
		 }

keys = names.keys()
values = names.values()

parent_dir = '/scratch/ahmm1g15/Thesis/IDEs/'
for k in keys:

	total_dir = parent_dir + f'scan_{k}V'
	os.chdir(total_dir)
	
	Predict(mask_val = names[k])

	


