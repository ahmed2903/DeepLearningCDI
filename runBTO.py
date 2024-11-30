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

temp_str = ''

#Predict
def Predict(wdecay):
	predict = CNNPredict()
	predict.SetDevice('cuda')
	predict.SetModel(NNModel, LRLU_val = 0.2, checkpoints=False)
	for name, module in predict.model.named_modules():
		if isinstance(module, nn.LeakyReLU):
			print(f"{name}: LeakyReLU negative slope = {module.negative_slope}")
	predict.SetExpData('expdataML.npy', mask=0.1, square_root=True) #
	predict.SetOutputFile('R_NS_NN_.npy')
	predict.SetSupport('NoSupport.npy')
	predict.SetTrainedNN("/scratch/ahmm1g15/MLTrainedNet/2023-06-02_15.15/CP150_2023-06-02_15.15.pth")
	predict.SetLRStepSize(5)
	predict.AddLR(5e-4)
	predict.AddGamma(0.95)
	predict.AddOptimiser(optim.Adam, eps=1e-8, weight_decay=wdecay)
	predict.AddScheduler(ss.StepLR)
	predict.SetNEpochs(400)
	predict.AddOpStep(400)
	#predict.AddOpStep(1025)
	#predict.AddOpStep(50)
	#predict.AddOpStep(50)
	predict.TransferPredict(AMP=False)
	predict.SaveParameters(training=False)
	predict.SaveTrainLoss()
	predict.SaveParameters(training = False)
	predict.PlotLoss()

	global temp_str
	temp_str = predict.datestr


lrlg = [n for n in range(8,0,-1)]
"""
for lval in lrlg:
	Predict(wdecay= 10**(-lval))
	# Ensure the directory exists before renaming
	new_dir_name = f'weight_decay_Adam/wdecay_power_m{lval}'
	if not os.path.exists(new_dir_name):
		os.makedirs(new_dir_name)

	os.rename(temp_str, os.path.join(new_dir_name, os.path.basename(temp_str)))
"""
supp = cuboid_support(110,110,110,112,112,112)
np.save('NoSupport.npy', supp)
Predict(wdecay = 0)
