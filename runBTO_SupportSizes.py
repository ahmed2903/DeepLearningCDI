#from gendata import GenData
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as ss
from cnnphase13 import NNModel, CNNTrain, CNNPredict
import numpy as np
import os 

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
def Predict():
	predict = CNNPredict()
	predict.SetDevice('cuda')
	predict.SetModel(NNModel, checkpoints=False)
	predict.SetExpData('expdataML.npy', mask=0.01, square_root=True) # 872248
	predict.SetOutputFile('R_NS_NN_.npy')
	predict.SetSupport('tempsupport.npy')
	predict.SetTrainedNN("/scratch/ahmm1g15/MLTrainedNet/2023-06-02_15.15/CP150_2023-06-02_15.15.pth")
	predict.SetLRStepSize(5)
	predict.AddLR(5e-4)
	predict.AddGamma(0.975)
	predict.AddOptimiser(optim.Adam, eps=1e-8, weight_decay=0)
	predict.AddScheduler(ss.StepLR)
	predict.SetNEpochs(400)
	predict.AddOpStep(400)
	predict.TransferPredict(AMP=False)
	predict.SaveParameters(training=False)
	predict.SaveTrainLoss()
	predict.SaveParameters(training = False)
	predict.PlotLoss()

	global temp_str
	temp_str = predict.datestr

xsizes = [81 + 4*i for i in range(5)]
ysizes = [82 + 5*i for i in range(5)]
zsizes = [42 + 6*i for i in range(5)]

processed_sizes = set()

for sx_ in xsizes:
	for sy_ in ysizes:
		for sz_ in zsizes:

			size_key = (sx_, sy_, sz_)

			if size_key not in processed_sizes:
				support = cuboid_support(sx = sx_, sy=sy_, sz=sz_, x=112, y = 112, z = 112)

				np.save('tempsupport.npy', support)
				Predict()
				
				# Ensure the directory exists before renaming
				new_dir_name = f'supp_{sx_}x_{sy_}y_{sz_}z'
				if not os.path.exists(new_dir_name):
					os.makedirs(new_dir_name)
				
				os.rename(temp_str, os.path.join(new_dir_name, os.path.basename(temp_str)))
			
