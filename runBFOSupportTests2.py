#from gendata import GenData
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as ss
from cnnphase12 import NNModel, CNNTrain, CNNPredict
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

def RotateSupport(inarray, axis, angle):
	from math import cos,sin,pi,fabs
	if fabs(angle) < 1e-3:
		return inarray
	arrayabs = np.abs(inarray)
	shape = arrayabs.shape
	deg2rad = pi/180.0
	tr = np.zeros((2,2), dtype=np.double)
	tr[0][0] = cos(angle*deg2rad)
	tr[0][1] = -sin(angle*deg2rad)
	tr[1][0] = sin(angle*deg2rad)
	tr[1][1] = cos(angle*deg2rad)
	d = np.zeros((9,2), dtype=np.double)
	nn =np.arange(-0.5,1.0,0.5)
	for i in range(len(nn)):
		for j in range(len(nn)):
			d[len(nn)*i+j][0], d[len(nn)*i+j][1] = nn[i], nn[j]
	d = d.T

	outarray = np.zeros_like(inarray)


	def RotateObject(farrayabs, finarray, foutarray, axis, spread):
		rotxy = np.array(np.where(np.array([0,1,2]) != (axis-1)))[0]
		idxs = np.array(np.where(farrayabs > 0.5))
		idxsnew = idxs.copy()
		cen = np.array(farrayabs.shape)//2
		cen[axis - 1] = 0
		cenidxs = idxs - cen.reshape(3,1)
		cenidxsxy = cenidxs[rotxy,:]
		cenidxsxytr = np.dot(cenidxsxy.T,tr).T
		idxsxytr = cenidxsxytr + cen[rotxy].reshape(2,1)
		n = len(spread)
		for i in range(n):
			xy = spread[:,i,np.newaxis]
			idxsnew[rotxy,:] = (idxsxytr + xy).astype(int)
			idxsnew[0, idxsnew[0,:] > (shape[0] - 1)] = shape[0] - 1
			idxsnew[0, idxsnew[0,:] < 0] = 0
			idxsnew[1, idxsnew[1,:] > (shape[1] - 1)] = shape[1] - 1
			idxsnew[1, idxsnew[1,:] < 0] = 0
			idxsnew[2, idxsnew[2,:] > (shape[2] - 1)] = shape[2] - 1
			idxsnew[2, idxsnew[2,:] < 0] = 0
			foutarray[idxsnew[0,:],idxsnew[1,:],idxsnew[2,:]] = finarray[idxs[0,:],idxs[1,:],idxs[2,:]]

		return foutarray

	out_array =  RotateObject(arrayabs, inarray, outarray, axis, d)

	return out_array

temp_str = ''

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
	predict.SetSupport('tempsupport.npy')
	predict.SetTrainedNN("/scratch/ahmm1g15/MLTrainedNet/2023-06-02_15.15/CP150_2023-06-02_15.15.pth")
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
	global temp_str
	temp_str = predict.datestr

xsizes = [43 + 15*i for i in range(7)]
ysizes = [90 + 7*i for i in range(7)]
zsizes = [29 + 8*i for i in range(7)]

s_size_1 = [43,90,29]
s_size_2 = [102,112,54]
s_size_3 = [134,134,78]


sizes = [[43,90,30],[102,112,54], [134,134,78] ]
processed_sizes = set()

for size in sizes:

	sx_, sy_, sz_ = size[0], size[1], size[2]

	if (sx_>70) & (sy_>100) & (sz_>39):
		support = cuboid_support(sx = sx_, sy=sy_, sz=sz_, x=136, y = 136, z = 80)
	else:
		support = cuboid_support(sx = sx_, sy=sy_, sz=sz_, x=136, y = 136, z = 80)
		support = RotateSupport(support, 1, 7)
		support = RotateSupport(support, 3, 27)

	np.save('tempsupport.npy', support)

	Predict()
	
	# Ensure the directory exists before renaming
	new_dir_name = f'supp_{sx_}x_{sy_}y_{sz_}z'
	if not os.path.exists(new_dir_name):
		os.makedirs(new_dir_name)
	
	os.rename(temp_str, os.path.join(new_dir_name, os.path.basename(temp_str)))