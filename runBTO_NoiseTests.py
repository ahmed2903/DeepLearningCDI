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

def addNoise(arr, noise_type = 'Poisson', noise_level = 0.1):
	
	
	mean = np.mean(arr)
	std_dev = np.std(arr)
	shp = arr.shape
	if noise_type == 'Gaussian':
		arr = arr + np.random.normal(0,std_dev * noise_level, shp).astype(np.complex)			
	
	if noise_type == 'Poisson':
		scaled_mean = np.abs(mean) * noise_level
		arr = arr + np.random.poisson(scaled_mean, shp).astype(np.complex128)	
		
	if noise_type == None: 
		print('Not adding noise')
		pass
		
	return arr
temp_str = ''
#Predict
def Predict():
	predict = CNNPredict()
	predict.SetDevice('cuda')
	predict.SetModel(NNModel, LRLU_val = 0.2, checkpoints=False)
	predict.SetExpData('noiseExpdataML.npy', mask=0.1, square_root=True)
	for name, module in predict.model.named_modules():
		if isinstance(module, nn.LeakyReLU):
			print(f"{name}: LeakyReLU negative slope = {module.negative_slope}")
	predict.SetOutputFile('R_NS_YN_.npy')
	predict.SetSupport('tempsupport.npy')
	predict.SetTrainedNN("/scratch/ahmm1g15/MLTrainedNet/2023-06-02_15.15/CP150_2023-06-02_15.15.pth")
	predict.SetLRStepSize(5)
	predict.AddLR(5e-4)
	predict.AddGamma(0.95)
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



noise_levels = [0.02 * n for n in range(1,20,1)]

parent_dir = '/scratch/ahmm1g15/Thesis/BTO/Support_NoiseTests'
expdata = np.load('expdataML.npy')

for nl in noise_levels:
	
	noisyarr = addNoise(expdata, noise_type = 'Poisson', noise_level = nl)

	np.save('noiseExpdataML.npy', noisyarr)
	support = cuboid_support(sx = 110, sy=110, sz=110, x=112, y = 112, z = 112)
	np.save('tempsupport.npy', support)

	Predict()
	# Ensure the directory exists before renaming
	new_dir_name = f'NoiseNoSupport_{int(nl*100)}'
	if not os.path.exists(new_dir_name):
		os.makedirs(new_dir_name)
	
	os.rename(temp_str, os.path.join(new_dir_name, os.path.basename(temp_str)))
			
