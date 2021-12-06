import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchsummary import summary
import torch.optim.lr_scheduler as ss
from torch.autograd import Variable

import numpy as np
import matplotlib.pyplot as plt

#from python_utils import get_lr
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm_, clip_grad_value_

#import cupy as cp

from torch.utils.dlpack import to_dlpack
from torch.utils.dlpack import from_dlpack

from time import strftime


class double_conv(nn.Module):
	def __init__(self, in_ch, out_ch, LRLUGrad=0.2, eps=1e-8, momentum=0.1):
		super(double_conv, self).__init__()
		self.conv = nn.Sequential(
			nn.Conv3d(in_ch, out_ch, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1), bias=True), 
			nn.BatchNorm3d(num_features=out_ch, eps=eps, momentum=momentum, affine=True, track_running_stats=False), 
			nn.LeakyReLU(LRLUGrad, inplace=True),

			nn.Conv3d(out_ch, out_ch, kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0), bias=True), 
			nn.Conv3d(out_ch, out_ch, kernel_size=(1, 3, 1), stride=1, padding=(0, 1, 0), bias=True),
			nn.Conv3d(out_ch, out_ch, kernel_size=(1, 1, 3), stride=1, padding=(0, 0, 1), bias=True),
			nn.BatchNorm3d(num_features=out_ch, eps=eps, momentum=momentum, affine=True, track_running_stats=False), 
			nn.LeakyReLU(LRLUGrad, inplace=True),	
		)
	def forward(self, x):
		x = self.conv(x)
		return x


class inconv(nn.Module):
	def __init__(self, in_ch, out_ch, LRLUGrad=0.2, eps=1e-8, momentum=0.1):
		super(inconv, self).__init__()
		self.conv = nn.Sequential(
			nn.Conv3d(in_ch, out_ch, kernel_size=(1, 1, 1), stride=1, padding=(0, 0, 0), bias=True), 
			nn.BatchNorm3d(num_features=out_ch, eps=eps, momentum=momentum, affine=True, track_running_stats=False),
			nn.LeakyReLU(LRLUGrad, inplace=False), 
			 
			nn.Conv3d(out_ch, out_ch, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1), bias=True), 
			nn.BatchNorm3d(num_features=out_ch, eps=eps, momentum=momentum, affine=True, track_running_stats=False),
			nn.LeakyReLU(LRLUGrad, inplace=True),
		)
	def forward(self, x):
		x = self.conv(x)
		return x


class down(nn.Module):
	def __init__(self, in_ch, out_ch, LRLUGrad=0.2, eps=1e-8, momentum=0.1):
		super(down, self).__init__()
		self.mpconv = nn.Sequential(
			nn.MaxPool3d(kernel_size=(2, 2, 2)),
			double_conv(in_ch, out_ch, LRLUGrad, eps, momentum),
		)
	def forward(self, x):
		x = self.mpconv(x)
		return x


class up01(nn.Module):
	def __init__(self, in_ch, out_ch, LRLUGrad=0.2, eps=1e-8, momentum=0.3):
		super(up01, self).__init__()
		self.upconv = nn.Sequential(
			nn.Upsample(scale_factor=2, mode='nearest'),
			double_conv(in_ch, out_ch, LRLUGrad, eps, momentum),
		)
	def forward(self, x):
		x = self.upconv(x)
		return x

class up02(nn.Module):
	def __init__(self, in_ch, out_ch, LRLUGrad=0.2, eps=1e-8, momentum=0.1):
		super(up02, self).__init__()
		self.upconv = nn.Sequential(
			nn.Upsample(scale_factor=2, mode='nearest'),
			double_conv(in_ch, out_ch, LRLUGrad, eps, momentum),
		)
	def forward(self, x):
		x = self.upconv(x)
		return x

class outconv(nn.Module):
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
	
class NNModel(nn.Module):
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
		# self.up04 = up(128, 64)
		self.outc00 = outconv(64, n_classes)

		self.up11 = up01(512, 256)
		self.up12 = up01(256, 128)
		self.up13 = up01(128, 64)
		# self.up04 = up(128, 64)
		self.outc11 = outconv(64, n_classes)

	def forward(self, x):
		x = self.inconv(x)
		x = self.down1(x)
		x = self.down2(x)
		x = self.down3(x)
		x = self.down4(x)


		x1 = x[:, 0::2, :, :]
		x1 = self.up01(x1)
		x1 = self.up02(x1)
		x1 = self.up03(x1)
		# x1 = self.up04(x, x1)
		x1 = self.outc00(x1)

		x2 = x[:, 1::2, :, :]
		x2 = self.up11(x2)
		x2 = self.up12(x2)
		x2 = self.up13(x2)
		# x = self.up4(x, x2)
		x2 = self.outc11(x2)

		x1 = torch.relu(x1)
		x2 = torch.relu(x2)
		# x2 = torch.clamp(x2, min=0, max=1)
		x0 = torch.cat((x1, x2), 1)

		return x0
class CNNTrain():
	def __init__(self, device_type='cpu'):
		self.verbose = True
		self.print_every = 3
		self.device_type = device_type
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
		self.batch_size = 45
		self.valid_size=0.05
		self.loader_train = None
		self.loader_test = None
		self.lrate_step_size = 5 #updates the learning rate after n epochs according to the schedular 
		self.op_step_size = 10 #optimiser step size, after n epochs, it changes the optimiser from optimiser1 to optimiser2 
		self.lr = 1e-2
		self.momentum = 0.9
		self.gamma = 0.1
		self.optimiser1 = None
		self.optimiser2 = None
		self.scheduler1 = None
		self.scheduler2 = None
		self.epochs = 250
		self.train_loss = []
		self.valid_loss = []
		self.datestr = ''
	def SetDeviceType(self, device_type='cpu'):
		if torch.cuda.is_available() and device_type=='cuda':
			self.device = torch.device("cuda")
		else:
			self.device = torch.device("cpu")
	def SetInputData(self, fname):
		self.input_data = np.load(fname)
		self.data_shape = self.input_data.shape
		self.target_data1 = np.zeros((self.data_shape[0], self.data_shape[-1], self.data_shape[-2], self.data_shape[-3]), dtype='float32')
		self.target_data1[:] = self.input_data[:,0,:,:,:]
	def SetTargetDataReal(self, fname):
		self.target_data0 = np.load(fname)

	def SetDimensions(self):
		self.nchannels = self.data_shape[1]
		self.nclasses= self.data_shape[1]
		self.nchannels_expand = self.data_shape[-1]
		self.image_size = self.data_shape[-1]
	def SetModel(self, model):
		self.model = model(1, 1).to(self.device)
	def SetValidSize(self, valid_size):
		self.valid_size = valid_size
	def SplitData(self):
		num_train = len(self.input_data)
		print(num_train)
		indices = list(range(num_train))
		split = int(np.floor(self.valid_size * num_train))
		np.random.shuffle(indices)
		self.train_idx, self.test_idx = indices[split:], indices[:split]
		print(len(self.train_idx), len(self.test_idx))
	def SetBatchSize(self, batch_size):
        	self.batch_size = batch_size
	def _LoadSplitTrain(self, index):
		datax = self.input_data[index].astype('float32')
		datay = self.target_data0[index].astype('float32')
		dataz = self.target_data1[index].astype('float32')
		
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

		if loadtype == 'train':
			self.loader_train = self._LoadSplitTrain(self.train_idx)
		elif loadtype == 'test':
			self.loader_test = self._LoadSplitTrain(self.test_idx)
		else:
			print('error starts here')


	def SetLRStepSize(self, lrate_step_size):
		self.lrate_step_size = lrate_step_size
	def SetLR(self, lr):
		self.lr = lr
	def SetMomentum(self, momentum):
		self.momentum = momentum
	def SetGamma(self, gamma):
		self.gamma = gamma

	def SetOptimiser1(self, optimiser = 'SGD'):
		if optimiser == 'SGD':
			self.optimiser1 = optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum)
		elif optimiser == 'ADAM':
			self.optimiser1 = optim.Adam(self.model.parameters(), lr=self.lr, amsgrad=False, eps=1e-10)
		elif optimiser == 'AdaGrad':
			self.optimiser1 = optim.Adagrad(self.model.parameters(), lr=self.lr)
		else: 
			print('Optimiser 1 not defined')

	def SetOptimiser2(self, optimiser = 'ADAM'):
		if optimiser == 'SGD':
			self.optimiser2 = optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum)
		elif optimiser == 'ADAM':
			self.optimiser2 = optim.Adam(self.model.parameters(), lr=self.lr, amsgrad=False, eps=1e-8)
		elif optimiser == 'AdaGrad':
			self.optimiser2 = optim.Adagrad(self.model.parameters(), lr=self.lr)
		else: 
			print('Optimiser 2 not defined')
		
	def SetScheduler1(self, scheduler='StepLR'):

		if scheduler == 'StepLR':
			self.scheduler1 = ss.StepLR(self.optimiser1, step_size=self.lrate_step_size, gamma=self.gamma)
		elif scheduler == 'ReduceLROnPlateau':
			self.scheduler1 = ss.ReduceLROnPlateau(self.optimiser1, mode='min', factor=0.9, patience=100)
		else: 
			print('Schedular 1 not defined')


	def SetScheduler2(self, scheduler='ReduceLROnPlateau'):
		if scheduler == 'StepLR':
			self.scheduler2 = ss.StepLR(self.optimiser1, step_size=self.lrate_step_size, gamma=self.gamma)
		elif scheduler == 'ReduceLROnPlateau':
			self.scheduler2 = ss.ReduceLROnPlateau(self.optimiser1, mode='min', factor=0.9, patience=100)
		else: 
			print('Scheduler 2 not defined')

	# def pull_out_gpu_fft(self, output):
	# 	X = self.data_shape[-2]
	# 	Y= self.data_shape[-1]
	# 	X2 = X//2
	# 	X4 = X//4
	# 	Y2 = Y//2
	# 	Y4 = Y//4
	# 	dx = to_dlpack(output) # type/tensor conversion
	# 	cx = np.fromDlpack(dx) # type/tensor conversion
	# 	cx1, cx2 = cx[:, 0, :, :], cx[:, 1, :, :] #split channels 

	# 	cx11 = np.zeros(([cx1.shape[0], 1, X, Y]), dtype='float32') #empty arrays
	# 	cx22 = np.zeros(([cx2.shape[0], 1, X, Y]), dtype='float32') #empty arrays

	# 	#increase dimensions to 64x64 (input dimensions for FT)

	# 	cx11[:, 0, X4:X4+X2, Y4:Y4+Y2] = cx1.copy() # cenre array of 32x32 in array of 64x64
	# 	cx22[:, 0, X4:X4+X2, Y4:Y4+Y2] = cx2.copy() # cenre array of 32x32 in array of 64x64

	# 	cx1122 = cx11 * np.exp(4j * np.pi * cx22)

	# 	cx1122 = np.abs(np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(cx1122, axes=(-2, -1)), axes=(-2, -1)), axes=(-2, -1)))

	# 	return from_dlpack(cx1122.toDlpack())

	def chi_loss(self, output, target):
		# chi squared error
		loss = torch.mean(torch.abs((output-target))**2)/(torch.mean(target**2)+1e-40)
		return loss 

	def pcc_loss(self, output, target):
		# Pearson correlation coefficient
		x = torch.abs(output)
		y = torch.abs(target)
		vx = torch.abs(x - torch.mean(x))
		vy = torch.abs(y - torch.mean(y))
		loss = torch.mean(vx * vy) / (torch.sqrt(torch.mean(vx ** 2) * torch.mean(vy ** 2))+1e-40)
		return 1. - loss
	
	def all_loss(self, output, target, input):
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

		loss1 = torch.sqrt(torch.mean((output0[:, 0, :, :, :] - target0[:, 0, :, :]) ** 2) /
						   (torch.mean(target0[:, 0, :, :, :] ** 2)))
		loss2 = torch.sqrt(torch.mean((output0[:, 1, :, :, :] - target0[:, 1, :, :]) ** 2) /
						   (torch.mean(target0[:, 1, :, :, :] ** 2)))

		#complex output object
		#get complex object of the output i.e. an object that includes the phase and amplitude
		#perform fourier transform on the object in order to obtain it in reci space 
		#compare the reci space output with the reci space input in loss3
		obj_comp = torch.zeros((output0.shape[0]), 2, X, Y, Z, requires_grad=False, device = self.device) 

		obj_comp[:, 0, (X2-X4):(X2+X4), (Y2-Y4):(Y2+Y4), (Z2 - Z4):(Z2 + Z4)] = output[:, 0, :, :, :] * torch.cos(2*torch.pi * (output[:,1,:,:,:]))
		obj_comp[:, 1, (X2-X4):(X2+X4), (Y2-Y4):(Y2+Y4), (Z2 - Z4):(Z2 + Z4)] = output[:, 0, :, :, :] * torch.sin(2*torch.pi * (output[:,1,:,:,:]))

		obj_comp = obj_comp[:,0,:,:,:] +1j * obj_comp[:,1,:,:,:]

		obj_comp = torch.fft.fftn(obj_comp, dim= (-3,-2,-1))

		#amp_out = torch.zeros((output.shape[0], X, Y), requires_grad=False, device=self.device, dtype=float)
		amp_out = torch.sqrt(torch.abs(obj_comp[:,:,:,:]) **2 + torch.abs(obj_comp[:,:,:,:]) **2)

		#input has the shape (n,1,X,Y)

		loss3 = self.pcc_loss(amp_out, input) 
		alpha, beta, gamma = 1., 1., 4.
		loss = (alpha * loss1 + beta * loss2 + gamma * loss3) / (alpha + beta + gamma)

		return loss
		
	def criterion(self, output, target, input):
		return self.all_loss(output, target, input)

	def GetLR(self, optimiser):
		for param_group in optimiser.param_groups:
			return param_group['lr']

	def SetNEpochs(self, epochs):
		self.epochs = epochs

	def TrainNN(self):
		for epoch in range(self.epochs):  # loop over the dataset multiple times
			train_loss_tmp = 0.0

			sw_op_flag = (epoch // self.op_step_size) % 2
			for ii, loader_batch_train in enumerate(self.loader_train, 0):
				
				# get the inputs; data is a list of [inputs, labels]
				# x_train = input
				# y_train = target 
				x_train, y_train, z_train = loader_batch_train
				x_train, y_train, z_train = Variable(x_train).to(self.device), Variable(y_train).to(self.device), Variable(z_train).to(self.device)

				# sets all the gradients to zero; to avoid accumulation of gradients from the previous epoch
				# this is potentially causing accumlation of gradients when we are switching from one optimizer to the next. best to set both to zero anyway?
				# if sw_op_flag == 0:
				# 	self.optimiser1.zero_grad()
				# elif sw_op_flag == 1:
				# 	self.optimiser2.zero_grad()
				
				self.optimiser1.zero_grad()
				#self.optimiser2.zero_grad()

				# forward propagation
				y_train_predict = self.model.forward(x_train)
				
				#define the loss and then backward propagate
				loss1 = self.criterion(y_train_predict, y_train, z_train)
				
				loss1.backward()

				#incorporate a clip on the values of the gradients, to avoid exploding gradients 
				#clip_grad_norm_(self.model.parameters(), max_norm = 1.0, norm_type=2)

				#optimize the weights and biases
				# if sw_op_flag == 0:
				# 	self.optimiser1.step()
				# elif sw_op_flag == 1:
				# 	self.optimiser2.step()

				self.optimiser1.step()

				train_loss_tmp += loss1.item()
				
				# print info if needed
				if self.verbose:
					if ii % self.print_every == 0: 
						print('[%d, %5d] Batch loss:: train %.5f'%(epoch + 1, ii + 1, train_loss_tmp / (ii + 1)))

			# #update the learning rate
			# if sw_op_flag == 0:
			# 	self.scheduler1.step()
			# 	lr = self.GetLR(self.optimiser1)
			# 	print('Using Optimiser 1')
			# elif sw_op_flag == 1:
			# 	self.scheduler2.step()
			# 	lr = self.GetLR(self.optimiser2)
			# 	print('Using Optimiser 2')
			
			self.scheduler1.step()
			lr = self.GetLR(self.optimiser1)

			
			
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
			
			
			self.model.train()

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
	##
	def SaveModel(self, epoch=0):
		torch.save(self.model.state_dict(), 'CP{}'.format(epoch+1)+self.datestr+'.pth')
		
	def PlotLoss(self):
		# plot the validation loss
		plt.plot(self.train_loss, label='Training loss')
		plt.plot(self.valid_loss, label='Validation loss')
		plt.legend(frameon=False)
		plt.savefig('validation_error'+self.datestr+'.png')
		plt.show()
			
	def SaveParameters(self):
		
		self.datestr = strftime("%Y-%m-%d_%H.%M.%S")
		params = ""
		params += "Device Type: %s \n"%self.device_type
		params += "Optimiser: %s \n"%self.optimiser1
		params += "Validation Size: %1.5e  \n"%self.valid_size
		params += "Batch Size: %2.6f \n" %self.batch_size
		params += "Learning Rate: %2.6f \n" %self.lr
		params += "Learning Rate Step Size: %2.6f \n" %self.lrate_step_size
		params += "Momentum: %d \n" %self.momentum
		params += "Gamma: %d \n" %self.gamma
		params += "Number of Epochs: %d \n" %self.epochs
		params += "-"*20
		#params += 'Model: \n\n', self.model, '\n'

		f = open('CNN_Training_Params_'+self.datestr+'.txt', "w")
		f.write(params)
		f.close()


if __name__ == '__main__':
	mynn = CNNTrain()
	mynn.SetDeviceType('cuda')
	mynn.SetInputData('/home/ahmm1g15/scratch/Run_DL')
	mynn.SetTargetDataReal('/home/ahmm1g15/scratch/Run_DL')
	mynn.SetModel(NNModel)
	mynn.SetValidSize(0.1)
	mynn.SplitData()
	mynn.SetBatchSize(5)
	mynn.LoadSplitTrain(loadtype='train')
	mynn.LoadSplitTrain(loadtype='test')
	mynn.SetLRStepSize(10)
	mynn.SetLR(1e-2)
	mynn.SetMomentum(0.9)
	mynn.SetGamma(0.1)
	mynn.SetOptimiser1('SGD')
	mynn.SetOptimiser2('ADAM')
	mynn.SetScheduler1('StepLR')
	mynn.SetScheduler2('StepLR')
	mynn.SetNEpochs(100)
	mynn.SaveParameters()
	mynn.TrainNN()
	mynn.PlotLoss()