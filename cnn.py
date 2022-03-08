import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.activation import ReLU
import torch.optim as optim
from torchsummary import summary
import torch.optim.lr_scheduler as ss
from torch.autograd import Variable
import os

import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm_

from time import strftime


class double_conv(nn.Module):
	"""
	Main convlutional layer
	Sequentially applying a 3D convolution followed by a batch normalization and a LeakyRelu activation function
	This is done twice however, the second second convolution is divided into three parts

	This concludes one layer in the encoder

	Values that can be tuned are: momentum and Grad of the leaky relu 

	Batch Normalization momentum: smaller batch size should equate to a large value of momentum ~ 0.9 - 0.99
									bigger batch size needs smaller values of momentum
	"""
	def __init__(self, in_ch, out_ch, LRLUGrad=0.2, eps=1e-8, momentum=0.9):
		super(double_conv, self).__init__()
		self.conv = nn.Sequential(
			nn.Conv3d(in_ch, out_ch, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1), bias=False), 
			nn.BatchNorm3d(num_features=out_ch, eps=eps, momentum=momentum, affine=True, track_running_stats=False), 
			nn.LeakyReLU(LRLUGrad, inplace=True),

			nn.Conv3d(out_ch, out_ch, kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0), bias=False), 
			nn.Conv3d(out_ch, out_ch, kernel_size=(1, 3, 1), stride=1, padding=(0, 1, 0), bias=False),
			nn.Conv3d(out_ch, out_ch, kernel_size=(1, 1, 3), stride=1, padding=(0, 0, 1), bias=False),
			nn.BatchNorm3d(num_features=out_ch, eps=eps, momentum=momentum, affine=True, track_running_stats=False), 
			nn.LeakyReLU(LRLUGrad, inplace=True),	
		)
	def forward(self, x):
		x = self.conv(x)
		return x


class inconv(nn.Module):
	"""
	Same as the previous convolutional layer, however, the second convolution is summarized in one operation opposed to three
	"""
	def __init__(self, in_ch, out_ch, LRLUGrad=0.2, eps=1e-8, momentum=0.9):
		super(inconv, self).__init__()
		self.conv = nn.Sequential(
			nn.Conv3d(in_ch, out_ch, kernel_size=(1, 1, 1), stride=1, padding=(0, 0, 0), bias=False), 
			nn.BatchNorm3d(num_features=out_ch, eps=eps, momentum=momentum, affine=True, track_running_stats=False),
			nn.LeakyReLU(LRLUGrad, inplace=False), 
			 
			nn.Conv3d(out_ch, out_ch, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1), bias=False), 
			nn.BatchNorm3d(num_features=out_ch, eps=eps, momentum=momentum, affine=True, track_running_stats=False),
			nn.LeakyReLU(LRLUGrad, inplace=True),
		)
	def forward(self, x):
		x = self.conv(x)
		return x


class down(nn.Module):
	"""
	Main encoder part
	Applying a maxpooling operation followed by the convultional layer
	"""
	def __init__(self, in_ch, out_ch, LRLUGrad=0.2, eps=1e-8, momentum=0.5):
		super(down, self).__init__()
		self.mpconv = nn.Sequential(
			nn.MaxPool3d(kernel_size=(2, 2, 2)),
			double_conv(in_ch, out_ch, LRLUGrad, eps, momentum),
		)
	def forward(self, x):
		x = self.mpconv(x)
		return x


class up01(nn.Module):
	'''
	One branch for the decoder part
	Amplitude recosntruction
	Upsampling operation followed by the convolutional layer
	'''
	def __init__(self, in_ch, out_ch, LRLUGrad=0.2, eps=1e-8, momentum=0.5):
		super(up01, self).__init__()
		self.upconv = nn.Sequential(
			nn.Upsample(scale_factor=2, mode='nearest'),
			double_conv(in_ch, out_ch, LRLUGrad, eps, momentum),
		)
	def forward(self, x):
		x = self.upconv(x)
		return x

class up02(nn.Module):
	'''
	One branch for the decoder part
	phase reconstriction
	upsampling operation followed by the convolutional layer
	'''
	def __init__(self, in_ch, out_ch, LRLUGrad=0.2, eps=1e-8, momentum=0.5):
		super(up02, self).__init__()
		self.upconv = nn.Sequential(
			nn.Upsample(scale_factor=2, mode='nearest'),
			double_conv(in_ch, out_ch, LRLUGrad, eps, momentum),
		)
	def forward(self, x):
		x = self.upconv(x)
		return x

class outconv(nn.Module):
	'''
	Convolutional layer for the final layer of the network
	only includes one conv3D operation, no maxpooling or leakyrelu
	'''
	def __init__(self, in_ch, out_ch):
		super(outconv, self).__init__()
		self.conv = nn.Conv3d(in_ch, out_ch, kernel_size=(1, 1, 1), stride=1, padding=(0, 0, 0), bias=True)
	def forward(self, x):
		x = self.conv(x)
		return x

"""
class NNModel__(nn.Module):
	def __init__(self, nchannels=1, nclasses=1, nchannels_expand=64, image_size=64):
		super(NNModel, self).__init__()
		self.inconv = inconv(nchannels, nchannels_expand)
		self.down = []
		self.up0 = []
		self.up1 = []
		size = image_size//2
		f = 0
		while size > 4:
			self.down.append([nchannels_expand*(2**(f)), nchannels_expand*(2**(f+1))])
			f += 1
			size = size//2
		# 
		for i in range(f, 0, -1):
			self.up0.append([nchannels_expand*(2**(i)), nchannels_expand*(2**(i-1))])
			self.up1.append([nchannels_expand*(2**(i)), nchannels_expand*(2**(i-1))])
		# 
		self.up0.append([nchannels_expand, nclasses])
		self.up1.append([nchannels_expand, nclasses])
	def forward(self, x):
		x = self.inconv(x)
		for d in self.down:
			temp = down(d[0], d[1])
			x = temp(x)
		x0 = x[:, 0::2, :, :] # FIX ME 3D
		x1 = x[:, 1::2, :, :] # FIX ME 3D
		for u in self.up0:
			x0 = up.forward(u)
		for u in self.up1:
			x1 = up.forward(u)
		x0 = torch.relu(x0)
		x1 = torch.relu(x1)
		# x2 = torch.clamp(x2, min=0, max=1)
		xout = torch.cat((x0, x1), 1)
		return xout
"""

class NNcoder(nn.Module):
	def __init__(self, n_channels=1, n_cha_lay=2):
		super(NNcoder, self).__init__()
		nn = int(n_cha_lay)  # dimensional of the model
		self.inc = inconv(n_channels, 32 * nn)
		self.down1 = down(32 * nn, 64 * nn)
		self.down2 = down(64 * nn, 128 * nn)
		self.down3 = down(128 * nn, 256 * nn)
		self.down4 = down(256 * nn, 512 * nn)

	def forward(self, x):
		x = self.inc(x)
		x = self.down1(x)
		x = self.down2(x)
		x = self.down3(x)
		x = self.down4(x)

		# x = self.dropout(x)

		return x

class NNampp(nn.Module):
	def __init__(self, n_classes=1, n_cha_lay=2):
		super(NNampp, self).__init__()

		nn = n_cha_lay  # dimensional of the model

		self.up10 = up01(np.int(512 * nn / 2), np.int(256 * nn / 2))
		self.up20 = up01(np.int(256 * nn / 2), np.int(128 * nn / 2))
		self.up30 = up01(np.int(128 * nn / 2), np.int(64 * nn / 2))
		# self.up40 = up01(np.int(64 * nn / 2), np.int(32 * nn / 2))
		self.outc1 = outconv(np.int(64 * nn / 2), n_classes)

	def forward(self, x11):
		# x11 = self.dropout(x11)

		x11 = self.up10(x11)
		x11 = self.up20(x11)
		x11 = self.up30(x11)
		x11 = self.outc1(x11)

		x11 = F.relu(x11, inplace=False)

		return x11

class NNphaa(nn.Module):
	def __init__(self, n_classes=1, n_cha_lay=2):
		super(NNphaa, self).__init__()

		nn = n_cha_lay  # dimensional of the model

		self.up11 = up02(np.int(512 * nn / 2), np.int(256 * nn / 2))
		self.up21 = up02(np.int(256 * nn / 2), np.int(128 * nn / 2))
		self.up31 = up02(np.int(128 * nn / 2), np.int(64 * nn / 2))

		self.outc2 = outconv(np.int(64 * nn / 2), n_classes)

	def forward(self, x22):
        # x22 = self.dropout(x22)

		x22 = self.up11(x22)
		x22 = self.up21(x22)
		x22 = self.up31(x22)

		x22 = self.outc2(x22)
		x22 = F.relu(x22, inplace=False)

		return x22

class NNPhase(nn.Module):
	with torch.autograd.set_detect_anomaly(True):
		def __init__(self, nn=2):
			super(NNPhase, self).__init__()

			self.coder = NNcoder(n_cha_lay=nn)
			self.aamp = NNampp(n_cha_lay=nn)
			self.ppha = NNphaa(n_cha_lay=nn)

		def forward(self, x):
			x = self.coder(x)
			x11 = x[:, 0::2, :, :]
			x22 = x[:, 1::2, :, :]
			x11 = self.aamp(x11)
			x22 = self.ppha(x22)
			x00 = torch.cat((x11, x22), 1)

			return x00

class NNModel(nn.Module):
	'''
	summing up all the operations to create the full network
	'''
	def __init__(self, n_channels=1, n_classes=1):
		super(NNModel, self).__init__()
		self.inconv = inconv(n_channels, 64)
		self.down1 = down(64, 128)
		self.down2 = down(128, 256)
		self.down3 = down(256, 512)
		self.down4 = down(512, 1024)

		self.up01 = up01(512, 256)
		self.up02 = up01(256, 128)
		self.up03 = up01(128, 64)
		self.outc00 = outconv(64, n_classes)

		self.up11 = up01(512, 256)
		self.up12 = up01(256, 128)
		self.up13 = up01(128, 64)
		self.outc11 = outconv(64, n_classes)

	def forward(self, x):
		x = self.inconv(x) 
		x = self.down1(x)
		x = self.down2(x)
		x = self.down3(x)
		x = self.down4(x)


		x1 = x[:, 0::2, :, :] #decicating half the channels for one branch
		x1 = self.up01(x1)
		x1 = self.up02(x1)
		x1 = self.up03(x1)
		x1 = self.outc00(x1)

		x2 = x[:, 1::2, :, :] #dedicating the other half for the other branch
		x2 = self.up11(x2)
		x2 = self.up12(x2)
		x2 = self.up13(x2)
		x2 = self.outc11(x2)

		x1 = torch.relu(x1) #activation function in the final layer is a relu opposed to a leakReLU
		x2 = torch.relu(x2) #activation function in the final layer is a relu opposed to a leakReLU
		x2 = torch.clamp(x2, min=-3.14, max=3.14) #clamping the phase values to be between -pi and pi 
		x0 = torch.cat((x1, x2), 1) # comnbining the two branches together 

		return x0

class CNNTrain():
	def __init__(self):
		self.verbose = True
		self.print_every = 8 
		self.device_type = 'cuda'
		self.device = None
		self.input_data = None
		self.target_data0 = None
		self.target_data1 = None
		self.data_shape = None
		self.nchannels = 1
		self.nclasses= 1
		self.nchannels_expand = 64
		self.image_size = 64 
		self.model = None
		self.train_idx = None
		self.test_idx = None
		self.batch_size = 16 #number of datasets to be included within one batch -preferrably powers of 2
		self.valid_size = 0.05 #fraction of the training set to be used for validation
		self.loader_train = None
		self.loader_test = None
		self.initilization = 'Kaiming'
		self.lrate_step_size = 5 #updates the learning rate after n epochs according to the schedular 
		self.op_step_size = 50 #optimiser step size, after n epochs, it changes the optimiser from optimiser1 to optimiser2 
		self.lr01 = 1e-2 #learning rate for the optimiser steps
		self.lr02 = 1e-4
		self.momentum = 0.9 #momentum for the SGD algorithm
		self.gamma = 0.1 #multiplicative factor for the LR schedular 
		self.scheduler = 'StepLR'
		self.no_optimisers = 1
		self.optimiser1 = None
		self.optimiser2 = None
		self.scheduler1 = None
		self.scheduler2 = None
		self.epochs = 250 #number of loops over the entire training set
		self.train_loss = []
		self.valid_loss = []
		self.datestr = ''

	def SetDeviceType(self, device_type='cuda'):
		"""
		Sets the device to be used to either a cpu or a gpu if it is available
		"""

		if torch.cuda.is_available() and device_type=='cuda':
			self.device = torch.device("cuda")
		else:
			self.device = torch.device("cpu")

	def SetInputData(self, fname1,fname2):

		"""
		Loading diffraction data of the training set, shape should be (n,1,x,y,z)
		and creating another version of the data without any channels such that shape (n,x,y,z)
		"""

		self.input_data = np.load(fname1)
		self.target_data1 = np.load(fname2)

		self.data_shape = self.input_data.shape

	def SetTargetData(self, fname):
		"""
		loads the target data of the training set
		shape should be (n,2,x//2,y//2,z//2)
		"""
		#Check inputs are as expected. WRITE

		self.target_data0 = np.load(fname)

	def SetDimensions(self):
		"""
		function defining the dimenstion of the input data
		"""
		self.nchannels = self.data_shape[1]
		self.nclasses= self.data_shape[1]
		self.nchannels_expand = self.data_shape[-1]
		self.image_size = self.data_shape[-1]
	
	def SetModel(self, model):
		"""
		Selecting the model to be used for the Neural network
		"""
		self.model = model(nn=2).to(self.device)

	def SetValidSize(self, valid_size):
		"""
		sets the validation size of the training set
		"""
		self.valid_size = valid_size

	def SplitData(self):
		"""
		splits the training set into two sets: one for training and one for vaidation
		"""
		num_train = len(self.input_data)
		#print(num_train)
		indices = list(range(num_train))
		split = int(np.floor(self.valid_size * num_train))
		np.random.shuffle(indices)
		self.train_idx, self.test_idx = indices[split:], indices[:split]
		#print(len(self.train_idx), len(self.test_idx))

	def SetBatchSize(self, batch_size):
		"""
		sets the batch size to be used in the training
		"""
		self.batch_size = batch_size
			
	def _LoadSplitTrain(self, index):
		"""
		selects random samples to be used for the validation
		puts batches of the traning set together
		"""
		datax = self.input_data[index].astype('float64')
		datay = self.target_data0[index].astype('float64')
		dataz = self.target_data1[index].astype('float64')
		
		num_train = len(datax)

		indices = list(range(num_train))
		np.random.shuffle(indices)

		idx = indices[0:]
		sampler = SubsetRandomSampler(idx)

		dx = torch.Tensor(datax)
		dy = torch.Tensor(datay)
		dz = torch.Tensor(dataz)

		data_xyz = torch.utils.data.TensorDataset(dx, dy, dz)
		
		trainloader = DataLoader(data_xyz,
					   sampler=sampler, shuffle=False, batch_size = self.batch_size) #think about including num_workers for mulit process data loading 

		del datax, datay, dataz, dx, dy, dz, data_xyz
		return trainloader

	def LoadSplitTrain(self, loadtype='test'):
		"""
		performs the splitting for the training and the validation sets
		"""
		if loadtype == 'train':
			self.loader_train = self._LoadSplitTrain(self.train_idx)
		elif loadtype == 'test':
			self.loader_test = self._LoadSplitTrain(self.test_idx)
		else:
			print('error starts here')

	def XavierNormal(self, m):

		"""
		function to initialize the weights and biases in the network
		values are set per type of layer (conv3D and batchnorm3D)
		helps avoid exploding gradients
		"""

		if isinstance(m, nn.Conv3d):
			nn.init.xavier_normal_(m.weight.data)
			if m.bias is not None:
				nn.init.constant_(m.bias.data, 0)
		elif isinstance(m, nn.BatchNorm3d):
			nn.init.constant_(m.weight.data, 1)
			nn.init.constant_(m.bias.data, 0)
	
	def KaimingNormal(self,m):

		"""
		function to initialize the weights and biases in the network
		values are set per type of layer (conv3D and batchnorm3D)
		helps avoid exploding gradients
		"""

		if isinstance(m, nn.Conv3d):
			nn.init.kaiming_normal_(m.weight.data, mode='fan_in', nonlinearity='leaky_relu')
			if m.bias is not None:
				nn.init.constant_(m.bias.data, 0)
		elif isinstance(m, nn.BatchNorm3d):
			nn.init.constant_(m.weight.data, 1)
			nn.init.constant_(m.bias.data, 0)
	
	def InitializeWeightsPre(self, filename):
		self.model.load_state_dict(torch.load(filename))

	def InitializeWeights(self, initilization = 'Kaiming'):
		"""
		applied the weight initization to the set network
		"""
		if initilization == 'Kaiming':
			self.model.apply(self.KaimingNormal)
			self.initilization = 'Kaiming Normal'
		elif initilization == 'Xavier':
			self.model.apply(self.XavierNormal)
			self.initilization = 'Xavier Normal'


	def InitializeWeights2(self):
		for p in self.model.parameters():
			p.data.normal_(0.02,0.01)
		for p in self.model.aamp.parameters():
			p.data.normal_(0.02,0.01)
			# p.weight,data.normal_(0.02,0.01)
			# p.bias.data.fill_(0)
			# p.data.fill_(0.02)
		for p in self.model.ppha.parameters():
			p.data.normal_(0.010, 0.00750)
			# p.data.fill_(0.02)
			# p.data.uniform_(0.020, 0.05)
			# print(p)	



	def SetLRStepSize(self, lrate_step_size):
		"""
		sets the learning rate step size, i.e. the number of epochs after which the learning rate will update
		"""
		self.lrate_step_size = lrate_step_size

	def SetLR(self, lr01, lr02):
		"""
		sets the learning rate
		"""
		self.lr01 = lr01
		self.lr02 = lr02

	def SetMomentum(self, momentum):
		"""
		sets the momentum of the SGD algorithm
		"""
		self.momentum = momentum
	def SetGamma(self, gamma):
		"""
		update rule for the LR, after n epochs the LR will be multiplied by gamma 
		"""
		self.gamma = gamma

	def SetOptimiser1(self, optimiser = 'SGD'):
		"""
		function to set the optimizer to be used
		more can be added - available on the pytorch documentation website
		"""
		if optimiser == 'SGD':
			self.optimiser1 = optim.SGD([
										{'params': self.model.coder.parameters()}, 
										{'params': self.model.aamp.parameters(), 'lr': self.lr01 * 1.0}, 
										{'params': self.model.ppha.parameters(), 'lr': self.lr01 * 1.0}
										], lr = self.lr01 * 2.0, momentum=self.momentum, nesterov=False)
				
		elif optimiser == 'ADAM':
			self.optimiser1 = optim.Adam(self.model.parameters(), lr=self.lr01, amsgrad=True, eps=1e-8)
		elif optimiser == 'AdaGrad':
			self.optimiser1 = optim.Adagrad(self.model.parameters(), lr=self.lr01)
		elif optimiser == 'RMSprop':
			self.optimiser1 = optim.RMSprop(self.model.parameters(), lr=self.lr01, alpha=0.99, eps=1e-08, weight_decay=0, momentum=self.momentum, centered=False)
		elif optimiser == 'ASGD':
			self.optimiser1 = optim.ASGD([
										{'params': self.model.coder.parameters()}, 
										{'params': self.model.aamp.parameters(), 'lr': self.lr01 * 1.0}, 
										{'params': self.model.ppha.parameters(), 'lr': self.lr01* 1.0}
										], lr = self.lr01 * 2.0)
		else:
			print('Optimiser 1 not defined')

	def SetOptimiser2(self, optimiser = 'ADAM'):
		"""
		a second optimiser function in case the user wants to alternate between different optimisers
		the training function needs to be slightly updates in this case 
		some function are already written for the scheduling and they need to be commented out 
		"""
		if optimiser == 'SGD':
			self.optimiser2 = optim.SGD(self.model.parameters(), lr=self.lr02, momentum=self.momentum, nesterov=True)
		elif optimiser == 'ADAM':
			self.optimiser2 = optim.Adam(self.model.parameters(), lr=self.lr02, amsgrad=True, eps=1e-8)
		elif optimiser == 'AdaGrad':
			self.optimiser2 = optim.Adagrad(self.model.parameters(), lr=self.lr02)
		else: 
			print('Optimiser 2 not defined')
		
	def SetScheduler1(self, scheduler='StepLR'):
		"""
		setting the type of function that the scheduling of the LR should take
		StepLR just multiplies LR by gamma by n epochs
		more info is available on the pytorch documentation
		"""
		if scheduler == 'StepLR':
			self.scheduler1 = ss.StepLR(self.optimiser1, step_size=self.lrate_step_size, gamma=self.gamma)
			self.scheduler = 'StepLR'
		elif scheduler == 'ReduceLROnPlateau':
			self.scheduler1 = ss.ReduceLROnPlateau(self.optimiser1, mode='min', factor=0.5, patience=5)
			self.scheduler = "ReduceLROnPlateau"
		else: 
			print('Schedular 1 not defined')


	def SetScheduler2(self, scheduler='ReduceLROnPlateau'):
		"""
		second scheduling function in case user wants to alteranate between different scheduling functions 
		-should not have to do that though-
		"""
		if scheduler == 'StepLR':
			self.scheduler2 = ss.StepLR(self.optimiser2, step_size=self.lrate_step_size, gamma=self.gamma)
		elif scheduler == 'ReduceLROnPlateau':
			self.scheduler2 = ss.ReduceLROnPlateau(self.optimiser2, mode='min', factor=0.5, patience=10)
		else: 
			print('Scheduler 2 not defined')

	def chi_loss(self, output, target):
		"""
		function to compute the chi squared error of two sets of data
		"""
		loss = torch.mean(torch.abs((output-target))**2)/(torch.mean(target**2)+1e-40)
		return loss 

	def pcc_loss(self, output, target):
		"""
		Pearson correlation coefficient
		"""
		x = torch.abs(output)
		y = torch.abs(target)
		vx = torch.abs(x - torch.mean(x))
		vy = torch.abs(y - torch.mean(y))
		loss = torch.mean(vx * vy) / (torch.sqrt(torch.mean(vx ** 2) * torch.mean(vy ** 2))+1e-40)
		return 1. - loss
	
	def all_loss(self, output, target, input):

		"""
		Function to define the total loss in the training
		loss in the phase channel is computed (loss2)
		loss in the amplitude channel is computed (loss1)
		loss in the fourier transform of the object is computed (loss3)

		all are merged together in loss function 

		user can add their own loss function and call it in criterion function
		"""
		X = self.data_shape[-3]
		Y= self.data_shape[-2]
		Z = self.data_shape[-1]
		X2 = X//2
		X4 = X//4
		Y2 = Y//2
		Y4 = Y//4
		Z2 =  Z//2
		Z4 = Z//4

		output0 = torch.zeros(output.shape[0], output.shape[1], X, Y, Z, device=self.device, requires_grad=False)
		output0[:, :, (X2 - X4):(X2 + X4), (Y2 - Y4):(Y2 + Y4), (Z2 - Z4):(Z2 + Z4)] = output
		target0 = torch.zeros(output.shape[0], output.shape[1], X, Y, Z, device=self.device, requires_grad=False)
		target0[:, :, (X2 - X4):(X2 + X4), (Y2 - Y4):(Y2 + Y4), (Z2 - Z4):(Z2 + Z4)] = target

		loss1 = self.pcc_loss(output0[:, 0, :, :, :],target0[:, 0, :, :, :])
		loss2 = self.pcc_loss(output0[:, 1, :, :, :],target0[:, 1, :, :, :])


		#complex output object
		#get complex object of the output i.e. an object that includes the phase and amplitude
		#perform fourier transform on the object in order to obtain it in reci space 
		#compare the reci space output with the reci space input in loss3
		obj_comp = torch.zeros((output0.shape[0]), 2, X, Y, Z, requires_grad=False, device = self.device) 

		obj_comp[:, 0, (X2-X4):(X2+X4), (Y2-Y4):(Y2+Y4), (Z2 - Z4):(Z2 + Z4)] = output[:, 0, :, :, :] * torch.cos(2*torch.pi * (output[:,1,:,:,:]))
		obj_comp[:, 1, (X2-X4):(X2+X4), (Y2-Y4):(Y2+Y4), (Z2 - Z4):(Z2 + Z4)] = output[:, 0, :, :, :] * torch.sin(2*torch.pi * (output[:,1,:,:,:]))
 
		obj_comp = obj_comp[:,0,:,:,:] +1j * obj_comp[:,1,:,:,:]
		
		obj_comp = torch.fft.fftn(obj_comp, dim= (-3,-2,-1))

		#amp_out = torch.zeros((output.shape[0], X, Y, Z), requires_grad=False, device=self.device, dtype=float)
		amp_out = torch.abs(obj_comp)

		#input has the shape (n,1,X,Y,Z)

		loss3 = self.pcc_loss(amp_out, input) 
		alpha, beta, gamma = 1., 1., 1.
		loss = (alpha * loss1 + beta * loss2 + gamma * loss3) / (alpha + beta + gamma)

		return loss

	def all_loss2(self, output, target):
		"""
		Function to define the total loss in the training
		loss in the phase channel is computed (loss2)
		loss in the amplitude channel is computed (loss1)
		loss in the fourier transform of the object is computed (loss3)

		all are merged together in loss function 

		user can add their own loss function and call it in critereon function
		"""
		X = self.data_shape[-3]
		Y= self.data_shape[-2]
		Z = self.data_shape[-1]
		X2 = X//2
		X4 = X//4
		Y2 = Y//2
		Y4 = Y//4
		Z2 =  Z//2
		Z4 = Z//4

		output0 = torch.zeros(output.shape[0], output.shape[1], X, Y, Z, device=self.device, requires_grad=False)
		output0[:, :, (X2 - X4):(X2 + X4), (Y2 - Y4):(Y2 + Y4), (Z2 - Z4):(Z2 + Z4)] = output

		target0 = torch.zeros(output.shape[0], output.shape[1], X, Y, Z, device=self.device, requires_grad=False)
		target0[:, :, (X2 - X4):(X2 + X4), (Y2 - Y4):(Y2 + Y4), (Z2 - Z4):(Z2 + Z4)] = target

		loss1 = self.pcc_loss(output0[:, 0, :, :, :],target0[:, 0, :, :, :])
		loss2 = self.pcc_loss(output0[:, 1, :, :, :],target0[:, 1, :, :, :])


		alpha, beta = 1., 1.
		loss = (alpha * loss1 + beta * loss2) / (alpha + beta)

		return loss
	
	def one_loss(self, output, input):

		"A loss function for when there is no target, and only the experimental diffraction input"

		X = 64
		Y= 64
		Z = 64
		X2 = X//2
		X4 = X//4
		Y2 = Y//2
		Y4 = Y//4
		Z2 =  Z//2
		Z4 = Z//4

		obj_comp = torch.zeros(1, 2, X, Y, Z, requires_grad=False, device = self.device) 

		obj_comp[:, 0, (X2-X4):(X2+X4), (Y2-Y4):(Y2+Y4), (Z2 - Z4):(Z2 + Z4)] = output[:, 0, :, :, :] * torch.cos(2*torch.pi * (output[:,1,:,:,:]))
		obj_comp[:, 1, (X2-X4):(X2+X4), (Y2-Y4):(Y2+Y4), (Z2 - Z4):(Z2 + Z4)] = output[:, 0, :, :, :] * torch.sin(2*torch.pi * (output[:,1,:,:,:]))
 
		obj_comp = obj_comp[:,0,:,:,:] +1j * obj_comp[:,1,:,:,:]
		
		obj_comp = torch.fft.fftn(obj_comp, dim= (-3,-2,-1))

		#amp_out = torch.zeros((output.shape[0], X, Y, Z), requires_grad=False, device=self.device, dtype=float)
		amp_out = torch.abs(obj_comp)

		#input has the shape (n,1,X,Y,Z)

		loss = self.pcc_loss(amp_out, input) 

		return loss

		
	def criterion(self, output, target0, target1):
		""" 
		calling the desired loss function to be used 
		"""
		return self.all_loss(output, target0, target1)

		#return self.one_loss(output, input)

	def GetLR(self, optimiser):
		"""
		a function to record the learning rate as it is updating in the training
		"""
		for param_group in optimiser.param_groups:
			return param_group['lr']

	def SetNEpochs(self, epochs):
		"""
		set the number of loops over the entire training data set during the training
		"""
		self.epochs = epochs
	
	def SetNOptimisers(self, N = 1, Sw = 50):

		self.op_step_size = Sw

		self.no_optimisers = N

	def TrainNN(self):
		"""
		funciton to perform the training of the network

		a for loop over the data set for desired number of epochs 
		a second for loop for the batches in the traning data set
			variable are set to the devive (cpu or cuda)
			the gradients are all set to zero initialy to avoid accumulation of the gradients
			the network is used ot predict an output
			loss function is computed
			loss is backward propagated and gradients are computed with loss.backward()
			clipping the gradients in order to avoid exploding gradient problem
			weights are updated in self.optimiser.step()
			loss values are recorded 

		lr is updates if condition is satisfied

		validaiton takes place in another for loop
			model.eval() turns off parts of the network that interfere with the validation

		Two function exist in the method to account for changes when using the StepLR schdeuler or the ReduceLROnPlateau

		training error and validation errors are recorded 

		model is saved

		"""

		def TrainLR():
			for epoch in range(self.epochs):  # loop over the dataset multiple times
				train_loss_tmp = 0.0
				self.model.train()

				sw_op_flag = (epoch // self.op_step_size) % 2

				for ii, loader_batch_train in enumerate(self.loader_train, 0):
					
					# get the inputs; data is a list of [inputs, labels]
					# x_train = input
					# y_train = target 
					x_train, y_train, z_train = loader_batch_train
					x_train, y_train, z_train = Variable(x_train).to(self.device), Variable(y_train).to(self.device), Variable(z_train).to(self.device)



					# sets all the gradients to zero; to avoid accumulation of gradients from the previous epoch
					# this is potentially causing accumlation of gradients when we are switching from one optimizer to the next. best to set both to zero anyway?
					
					#self.optimiser1.zero_grad()
					#self.optimiser2.zero_grad()
					for param in self.model.parameters():
						param.grad = None


					#forward propagation
					y_train_predict = self.model.forward(x_train)

					#define the loss and then backward propagate
					loss1 = self.criterion(y_train_predict, y_train, z_train)
					
					loss1.backward()

					#incorporate a clip on the values of the gradients, to avoid exploding gradients 
					clip_grad_norm_(self.model.coder.parameters(), 2)
					clip_grad_norm_(self.model.ppha.parameters(), 2)
					clip_grad_norm_(self.model.aamp.parameters(), 1.25)



					#optimize the weights and biases
					
					if self.no_optimisers == 1:
						self.optimiser1.step()
					
					elif self.no_optimisers == 2:
						if sw_op_flag == 0:
							self.optimiser1.step()
						elif sw_op_flag == 1:
							self.optimiser2.step()

					train_loss_tmp += loss1.item()
					
					# print info if needed
					if self.verbose:
						if ii % self.print_every == 0: 
							print('[%d, %5d] Batch loss:: train %.5f'%(epoch + 1, ii + 1, train_loss_tmp / (ii + 1)))

				#update the learning rate

				if self.no_optimisers == 1:
					self.scheduler1.step()
					lr = self.GetLR(self.optimiser1)

				elif self.no_optimisers == 2:
					if sw_op_flag == 0:
						self.scheduler1.step()
						lr = self.GetLR(self.optimiser1)
						print('Using Optimiser 1')
					elif sw_op_flag == 1:
						self.scheduler2.step()
						lr = self.GetLR(self.optimiser2)
						print('Using Optimiser 2')
				
				#validation step
				with torch.no_grad():
					valid_loss_tmp = 0.0
					self.model.eval() # turn off some specific parts of the model for the evaluation with model.eval()
					for loader_batch_test in self.loader_test:
						x_test, y_test, z_test = loader_batch_test
						x_test, y_test, z_test = x_test.to(self.device), y_test.to(self.device), z_test.to(self.device)
						y_pred = self.model.forward(x_test)
						loss2 = self.criterion(y_pred, y_test, z_test)
						valid_loss_tmp += loss2.item()
				
				print(len(self.loader_test))
				self.train_loss.append(train_loss_tmp / len(self.loader_train))
				self.valid_loss.append(valid_loss_tmp / len(self.loader_test))

				if np.isfinite(self.train_loss[-1]):
					pass
				else:
					print('model is breaking')
					break

				# if ii % print_every == 0:                  # print every mini-batches
				if self.verbose:
					print('Epoch-loss:: train %.5f  valid %.5f lr %.5f' %(self.train_loss[-1], self.valid_loss[-1], lr)) 
					print('-'*20)
			
				# save
				if epoch % (self.epochs -1) == 0:
					self.SaveModel(epoch)

		def TrainRLROP():

			for epoch in range(self.epochs):  # loop over the dataset multiple times
				train_loss_tmp = 0.0
				self.model.train()

				sw_op_flag = (epoch // self.op_step_size) % 2

				for ii, loader_batch_train in enumerate(self.loader_train, 0):
					
					# get the inputs; data is a list of [inputs, labels]
					# x_train = input
					# y_train = target 
					x_train, y_train, _ = loader_batch_train
					x_train, y_train = Variable(x_train).to(self.device), Variable(y_train).to(self.device)

					# sets all the gradients to zero; to avoid accumulation of gradients from the previous epoch
					# this is potentially causing accumlation of gradients when we are switching from one optimizer to the next. best to set both to zero anyway?
					
					#self.optimiser1.zero_grad()
					#self.optimiser2.zero_grad()
					for param in self.model.parameters():
						param.grad = None


					#forward propagation
					y_train_predict = self.model.forward(x_train)

					#define the loss and then backward propagate
					loss1 = self.criterion(y_train_predict, y_train)
					
					loss1.backward()

					#incorporate a clip on the values of the gradients, to avoid exploding gradients 
					clip_grad_norm_(self.model.coder.parameters(), 2)
					clip_grad_norm_(self.model.ppha.parameters(), 2)
					clip_grad_norm_(self.model.aamp.parameters(), 1.25)

					#optimize the weights and biases
					if self.no_optimisers == 1:
						self.optimiser1.step()
					
					elif self.no_optimisers == 2:
						if sw_op_flag == 0:
							self.optimiser1.step()
						elif sw_op_flag == 1:
							self.optimiser2.step()

					train_loss_tmp += loss1.item()
					
					# print info if needed
					if self.verbose:
						if ii % self.print_every == 0: 
							print('[%d, %5d] Batch loss:: train %.5f'%(epoch + 1, ii + 1, train_loss_tmp / (ii + 1)))
				
				#validation step
				with torch.no_grad():
					valid_loss_tmp = 0.0
					self.model.eval() # turn off some specific parts of the model for the evaluation with model.eval()
					for loader_batch_test in self.loader_test:
						x_test, y_test, _ = loader_batch_test
						x_test, y_test  = x_test.to(self.device), y_test.to(self.device)
						y_pred = self.model.forward(x_test)
						loss2 = self.criterion(y_pred, y_test)
						valid_loss_tmp += loss2.item()
				
				print(len(self.loader_test))
				self.train_loss.append(train_loss_tmp / len(self.loader_train))
				self.valid_loss.append(valid_loss_tmp / len(self.loader_test))

				if self.no_optimisers == 1:
					self.scheduler1.step(self.valid_loss[-1])
					lr = self.GetLR(self.optimiser1)

				elif self.no_optimisers == 2:
					if sw_op_flag == 0:
						self.scheduler1.step(self.valid_loss[-1])
						lr = self.GetLR(self.optimiser1)
						print('Using Optimiser 1')
					elif sw_op_flag == 1:
						self.scheduler2.step(self.valid_loss[-1])
						lr = self.GetLR(self.optimiser2)
						print('Using Optimiser 2')

				if np.isfinite(self.train_loss[-1]):
					pass
				else:
					print('model is breaking')
					break

				# if ii % print_every == 0:                  # print every mini-batches
				if self.verbose:
					print('Epoch-loss:: train %.5f  valid %.5f lr %.5f' %(self.train_loss[-1], self.valid_loss[-1], lr)) 
					print('-'*20)
			
				# save
				if epoch % (self.epochs -1) == 0:
					self.SaveModel(epoch)

		
		if self.scheduler == 'StepLR':
			return TrainLR()
		elif self.scheduler == 'ReduceLROnPlateau':
			return TrainRLROP()


	def one_Train(self, fname, mask=100):

		

		self.expdata = np.load(fname)
		self.expdata = self.expdata > mask

		maxElement = np.amax(self.expdata)

		self.expdata = self.expdata / maxElement

		i = self.expdata.shape[0]
		j = self.expdata.shape[1]
		k = self.expdata.shape[2]

		torcharray = np.zeros((1,1,i,j,k), dtype=np.double)
		torcharray[0,0,:,:,:]  = self.expdata[:,:,:]
		x_input = torch.from_numpy(torcharray)
		x_input = x_input.to(device = self.device, dtype = torch.float)

		for epoch in range(self.epochs):  # loop over the dataset multiple times
			train_loss_tmp = 0.0
			self.model.train()


			self.optimiser1.zero_grad()

			y_output = self.model.forward(x_input)

			loss1 = self.criterion(y_output, x_input)

			loss1.backward

			train_loss_tmp += loss1.item()

			clip_grad_norm_(self.model.coder.parameters(), 2)
			clip_grad_norm_(self.model.ppha.parameters(), 2)
			clip_grad_norm_(self.model.aamp.parameters(), 1.25)

			self.optimiser1.step

			self.scheduler1.step()
			lr = self.GetLR(self.optimiser1)

			self.train_loss.append(train_loss_tmp)

			if np.isfinite(self.train_loss[-1]):
				pass
			else:
				print('model is breaking')
				break

			if self.verbose:
				print('Epoch-loss:: train %.5f   lr %.5f' %(self.train_loss[-1], lr)) 
				print('-'*20)

			if epoch % (self.epochs -1) == 0:
				self.SaveModel(epoch)

	def SaveModel(self, epoch=0):
		"""
		saving the model
		"""
		self.datestr = strftime("%m-%d_%H.%M")
		torch.save(self.model.state_dict(),'CP{}'.format(epoch+1)+'_'+self.datestr+'.pth')
		
	def PlotLoss(self):
		"""
		plotting the loss values of the training and validations sets per epoch
		"""
		# plot the validation loss
		plt.plot(self.train_loss, label='Training loss')
		plt.plot(self.valid_loss, label='Validation loss')
		plt.legend(frameon=False)
		plt.savefig('validation_error_'+self.datestr+'.png')
		#plt.show()
			
	def SaveParameters(self):
		"""
		saving some of the important parameters in the network
		"""
		params = ""
		params += "Device Type: %s \n"%self.device_type
		params += "Optimiser 1: %s \n"%self.optimiser1
		if self.no_optimisers == 2:
			params += "Optimiser 2: %s \n"%self.optimiser2
		params += "Initilizing the parameters using: %s \n"%self.initilization
		params += "Validation Size: %f  \n"%self.valid_size
		params += "Batch Size: %d \n" %self.batch_size
		params += "Learning Rate: %2.6f \n" %self.lr01
		params += "Learning Rate Step Size: %2.6f \n" %self.lrate_step_size
		params += "Momentum: %f \n" %self.momentum
		params += "Gamma: %f \n" %self.gamma
		params += "Number of Epochs: %d \n" %self.epochs
		params += "Valdiation loss: %f \n" %self.valid_loss[-1]
		params += "Training Loss: %f \n" %self.train_loss[-1]
		params += "-"*20
		#params += 'Model: \n\n', self.model, '\n'

		f = open('CNN_Training_Params_'+self.datestr+'.txt', "w")
		f.write(params)
		f.close()

class CNNPredict(CNNTrain):
	"""
	a class to be used for prediction after obtaining a trained neural network
	still to be tested
	"""
	def __init__(self, device_type='cuda'):
		super().__init__()
		self.expdata = None
		self.trained_network = None
		self.output = None
		self.devive = None
		self.model = None
		self.torcharray = None
		self.device_type = device_type


	def SetDeviceType(self, device_type='cpu'):
		"""
		Sets the device to be used to either a cpu or a gpu if it is available
		"""

		if torch.cuda.is_available() and device_type=='cuda':
			self.device = torch.device("cuda")
		else:
			self.device = torch.device("cpu")

	def SetModel(self, model):
		"""
		Selecting the model to be used for the Neural network
		"""
		self.model = model(nn=2).to(self.device)


	def SetExpData(self, fname, mask=100):
		"""
		selects the diffraction data to be used for prediction
		masks out the noise, by selecting a threshold value below which everything is set to zero
		normalizes the array : dividing by the maximum value in the array such that all pixels are in range (0,1)
		shape of array (x,y,z)
		"""
		self.expdata = np.load(fname)
		self.expdata = self.expdata > mask

		maxElement = np.amax(self.expdata)

		self.expdata = self.expdata / maxElement
		self.expdata = np.sqrt(self.expdata)
		
	def SetTrainedNN(self, fname):
		"""
		loads the trained neural network, must be a .pth file 
		"""
		self.trained_network = fname

	def SetOutputFile(self, fname):
		"""
		name of the output file, i.e. the reconstructed object
		"""
		self.output = fname

	def Predict(self):
		"""
		forward propagates the diffraction pattern through the trained neural network 
		obtains an output complex object 
		"""

		i = self.expdata.shape[0]
		j = self.expdata.shape[1]
		k = self.expdata.shape[2]

		self.torcharray = np.zeros((1,1,i,j,k), dtype=np.double)
		self.torcharray[0,0,:,:,:]  = self.expdata[:,:,:]
		self.torcharray = torch.from_numpy(self.torcharray)
		

		if torch.cuda.is_available() and self.device_type=='cuda':
			self.model.load_state_dict(torch.load(self.trained_network))
			self.torcharray = self.torcharray.to(device = self.device, dtype = torch.float)
		else:
			self.model.load_state_dict(torch.load(self.trained_network, map_location = 'cpu'))

		self.model.eval()
			
		with torch.no_grad():
			sequence = self.model(self.torcharray)
			
		sequence = sequence.cpu()

		amp = np.zeros((i//2,j//2,k//2), dtype=np.double)
		pha = np.zeros((i//2,j//2,k//2), dtype=np.double)

		amp[:] = sequence[0,0,:,:,:]
		pha[:] = sequence[0,1,:,:,:]

		com = amp * np.cos(pha) + 1j * amp * np.sin(pha)

		np.save(self.output, com)
		
	def iterate_predict(self):

		x_input = self.torcharray
		i = self.expdata.shape[0]
		j = self.expdata.shape[1]
		k = self.expdata.shape[2]

		for epoch in range(self.epochs):  # loop over the dataset multiple times
			train_loss_tmp = 0.0
			self.model.train()


			self.optimiser1.zero_grad()

			y_output = self.model.forward(x_input)

			loss1 = self.criterion(y_output, x_input)

			loss1.backward()

			train_loss_tmp += loss1.item()

			clip_grad_norm_(self.model.coder.parameters(), 2)
			clip_grad_norm_(self.model.ppha.parameters(), 2)
			clip_grad_norm_(self.model.aamp.parameters(), 1.25)

			self.optimiser1.step()

			self.scheduler1.step()
			lr = self.GetLR(self.optimiser1)

			self.train_loss.append(train_loss_tmp)

			if np.isfinite(self.train_loss[-1]):
				pass
			else:
				print('model is breaking')
				break

			if self.verbose:
				print('Epoch-loss:: train %.5f   lr %.5f' %(self.train_loss[-1], lr)) 
				print('-'*20)

			if epoch % (self.epochs -1) == 0:
				self.model.eval()
				with torch.no_grad():
					sequence = self.model.forward(self.torcharray)
				
			sequence = sequence.cpu()

			amp = np.zeros((i//2,j//2,k//2), dtype=np.double)
			pha = np.zeros((i//2,j//2,k//2), dtype=np.double)

			amp[:] = sequence[0,0,:,:,:]
			pha[:] = sequence[0,1,:,:,:]

			com = amp * np.cos(pha) + 1j * amp * np.sin(pha)

			np.save('iter_'+self.output, com)

	


if __name__ == '__main__':
	mynn = CNNTrain()
	mynn.SetDeviceType('cuda')
	mynn.SetInputData('reci_intensity.npy')
	mynn.SetTargetData('real_obj.npy')
	mynn.SetModel(NNModel)
	mynn.SetValidSize(0.1)
	mynn.SplitData()
	mynn.SetBatchSize(5)
	mynn.LoadSplitTrain(loadtype='train')
	mynn.LoadSplitTrain(loadtype='test')
	mynn.InitializeWeights('Kaiming') #takes in Kaiming or Xavier
	mynn.SetLRStepSize(25)
	mynn.SetLR(1e-4,1e-6)
	mynn.SetMomentum(0.9)
	mynn.SetGamma(0.75)
	mynn.SetNOptimisers(N=1)
	mynn.SetOptimiser1('ASGD')
	mynn.SetOptimiser2('ADAM')
	mynn.SetScheduler1('StepLR')
	mynn.SetScheduler2('StepLR')
	mynn.SetNEpochs(100)
	mynn.TrainNN()
	mynn.SaveParameters()
	mynn.PlotLoss()