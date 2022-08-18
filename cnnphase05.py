# ###########################################
# Filename: cnnphase.py
# Convolution Neural Network for Phase Retrieval.
# Derived from work by Longlong Wu.
# 
# Authors: Marcus Newton, Ahmed Mohamed.
# 
# Version 0.5
# Licence: GNU GPL 3
#
# ###########################################

import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')

from time import strftime
import os
import inspect
import threading

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.activation import ReLU
import torch.optim as optim
import torch.optim.lr_scheduler as ss
from torch.autograd import Variable
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm_, clip_grad_value_
from torch.utils.checkpoint import checkpoint_sequential
from torch.cuda.amp import GradScaler


class double_conv(nn.Module):
	"""
	Main convlutional layer
	Sequentially applying a 3D convolution followed by a batch normalization and a LeakyRelu activation function
	This is done twice however, the second second convolution is divided into three parts

	This concludes one layer in the encoder

	Values that can be tuned are: momentum and Grad of the leaky relu 
	"""
	def __init__(self, in_ch, out_ch, LRLUGrad=0.2, eps=1e-8, momentum=0.9):
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
	"""
	Same as the previous convolutional layer, however, the second convolution is summarized in one operation opposed to three
	"""
	def __init__(self, in_ch, out_ch, LRLUGrad=0.2, eps=1e-8, momentum=0.9):
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
	"""
	Main encoder part
	Applying a maxpooling operation followed by the convultional layer
	"""
	def __init__(self, in_ch, out_ch, LRLUGrad=0.2, eps=1e-8, momentum=0.9, checkpoints=False):
		super(down, self).__init__()
		self.checkpoints = checkpoints
		self.mpconv = nn.Sequential(
			nn.MaxPool3d(kernel_size=(2, 2, 2)),
			double_conv(in_ch, out_ch, LRLUGrad, eps, momentum),
		)
	def forward(self, x):
		if self.checkpoints:
			x = checkpoint_sequential(self.mpconv, 2, x)
		else:
			x = self.mpconv(x)
		return x


class up01(nn.Module):
	'''
	One branch for the decoder part
	Amplitude recosntruction
	Upsampling operation followed by the convolutional layer
	'''
	def __init__(self, in_ch, out_ch, LRLUGrad=0.2, eps=1e-8, momentum=0.9, checkpoints=False):
		super(up01, self).__init__()
		self.checkpoints = checkpoints
		self.upconv = nn.Sequential(
			nn.Upsample(scale_factor=2, mode='nearest'),
			double_conv(in_ch, out_ch, LRLUGrad, eps, momentum),
		)
	def forward(self, x):
		if self.checkpoints:
			x = checkpoint_sequential(self.upconv, 2, x)
		else:
			x = self.upconv(x)
		return x

class up02(nn.Module):
	'''
	One branch for the decoder part
	phase reconstriction
	upsampling operation followed by the convolutional layer
	'''
	def __init__(self, in_ch, out_ch, LRLUGrad=0.2, eps=1e-8, momentum=0.9, checkpoints=False):
		super(up02, self).__init__()
		self.checkpoints = checkpoints
		self.upconv = nn.Sequential(
			nn.Upsample(scale_factor=2, mode='nearest'),
			double_conv(in_ch, out_ch, LRLUGrad, eps, momentum),
		)
	def forward(self, x):
		if self.checkpoints:
			x = checkpoint_sequential(self.upconv, 2, x)
		else:
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

class NNModel(nn.Module):
	'''
	summing up all the operations to create the full network
	'''
	def __init__(self, n_channels=1, n_classes=1, momentum = 0.9, checkpoints = False):
		super(NNModel, self).__init__()
		self.inconv = inconv(n_channels, 64)
		self.down1 = down(64, 128, momentum = momentum, checkpoints = checkpoints)
		self.down2 = down(128, 256, momentum = momentum, checkpoints = checkpoints)
		self.down3 = down(256, 512, momentum = momentum, checkpoints = checkpoints)
		self.down4 = down(512, 1024, momentum = momentum, checkpoints = checkpoints)

		self.up01 = up01(512, 256, momentum = momentum, checkpoints = checkpoints)
		self.up02 = up01(256, 128, momentum = momentum, checkpoints = checkpoints)
		self.up03 = up01(128, 64, momentum = momentum, checkpoints = checkpoints)
		self.outc00 = outconv(64, n_classes)

		self.up11 = up01(512, 256, momentum = momentum, checkpoints = checkpoints)
		self.up12 = up01(256, 128, momentum = momentum, checkpoints = checkpoints)
		self.up13 = up01(128, 64, momentum = momentum, checkpoints = checkpoints)
		self.outc11 = outconv(64, n_classes)
	
	def forward(self, x):
		x = self.inconv(x)
		x = self.down1(x)
		x = self.down2(x)
		x = self.down3(x)
		x = self.down4(x)
		
		x1 = x[:, 0::2, :, :] #dedicating half the channels for one branch
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
		x2 = torch.clamp(x2, min=0, max=1) #clamping the phase values to be between -pi and pi 
		x0 = torch.cat((x1, x2), 1) # comnbining the two branches together 

		return x0



class CNNTrain():
	def __init__(self):
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.cuda_device_count = 0
		self.hyperpars = {}
		self.data = {}
		self.model = None
		self.valid_size = 0.05
		self.batch_size = 16
		self.train_test_idxs = {}
		self.loader = {}
		self.optimisers = {}
		self.schedulers = {}
		self._initfn = None
		self._initfn_kwargs = {}
		self.train_loss = []
		self.valid_loss = []
		self.verbose = True
		self.print_every = 200
		self.datestr = strftime("%Y-%m-%d_%H.%M")
		self.nthreads = len(os.sched_getaffinity(0))
	def GetKwArgs(self, obj, kwargs):
		obj_sigs = []
		obj_args = {}
		for arg in inspect.signature(obj).parameters.values():
			if not arg.default is inspect._empty:
				obj_sigs.append(arg.name)
		for key, value in kwargs.items():
			if key in obj_sigs:
				obj_args[key] = value
		return obj_args
	def SetDevice(self, device='cuda'):
		"""
		Sets the device to either CPU ('cpu') or GPU ('cuda'), if available.
		"""
		if torch.cuda.is_available() and device == 'cuda':
			self.device = torch.device("cuda")
		else:
			self.device = torch.device("cpu")
	def SetInputData(self, fname):
		"""
		Load Fourier-space amplitude diffraction data 
		from training. Shape is: (n,1,x,y,z).
		Create another version of the data without 
		channels with shape (n,x,y,z). Wrap order of fft.
		"""
		self.data['input_data'] = np.load(fname)
		shp = self.data['input_data'].shape
		self.data['target_data1'] = np.zeros((shp[0], shp[-3], shp[-2], shp[-1]), dtype='float32')
		self.data['target_data1'][:] = self.data['input_data'][:,0,:,:,:]
		self.WrapAroundData(self.data['target_data1'])
	def WrapAroundData(self, data):
		def CalcThread(idxrange):
			for i in range(idxrange[0], idxrange[1], 1):
				data[i,:,:,:] = np.fft.fftshift(data[i,:,:,:])
		xs = []
		n = data.shape[0]
		blk = n//self.nthreads
		for i in range(self.nthreads):
			xs.append([blk*i, blk*(i+1)])
		xs[-1][1] = n
		threads = []
		for t in range(self.nthreads):
			thread = threading.Thread(target=CalcThread, args=(xs[t],))
			thread.start()
			threads.append(thread)
		for thread in threads:
			if thread.is_alive():
				thread.join()
	def SetTargetData(self, fname):
		"""
		Load target data from training set.
		Shape is: (n,2,x//2,y//2,z//2)
		[:,0,:,:,:] is amplitude, [:,1,:,:,:] is phase.
		"""
		self.data['target_data0'] = np.load(fname)
	def SetModel(self, model, **kwargs):
		"""
		Selecting the model to be used for the network.
		Keywords are passed to model if they exist in the model.
		"""
		model_args = self.GetKwArgs(model, kwargs)
		self.model = model(**model_args).to(self.device)
		cuda_device_count = torch.cuda.device_count()
		if self.device.type == "cuda" and cuda_device_count > 1:
			self.cuda_device_count = cuda_device_count
			self.model = nn.DataParallel(self.model)
			self.model.to(self.device)
	def SetValidSize(self, valid_size):
		"""
		Set the validation size of the training set
		"""
		if valid_size > 0.0 and valid_size < 1.0:
			self.valid_size = valid_size
	def SplitData(self):
		"""
		Split the training set into two sets: 
		one for training and one for vaidation
		"""
		num_train = len(self.data['input_data'])
		indices = list(range(num_train))
		split = int(np.floor(self.valid_size * num_train))
		np.random.shuffle(indices)
		self.train_test_idxs['train'], self.train_test_idxs['test'] = indices[split:], indices[:split]
	def SetBatchSize(self, batch_size): #FIXME
		"""
		sets the batch size to be used in the training
		"""
		# Do we need to send an upper bound on this number.  
		# Should be less than dataset length?
		# Also zero lower bound?
		self.batch_size = batch_size
	def _LoadSplitTrain(self, index, num_workers=0):
		"""
		selects random samples to be used for the validation
		puts batches of the traning set together
		"""
		datax = self.data['input_data'][index].astype('float32')
		datay = self.data['target_data0'][index].astype('float32')
		dataz = self.data['target_data1'][index].astype('float32')
		
		num_train = len(datax)

		indices = list(range(num_train))
		np.random.shuffle(indices)

		idx = indices[0:] # What is this doing?
		sampler = SubsetRandomSampler(idx)

		dx = torch.Tensor(datax)
		dy = torch.Tensor(datay)
		dz = torch.Tensor(dataz)

		data_xyz = torch.utils.data.TensorDataset(dx, dy, dz)
		
		trainloader = DataLoader(data_xyz,
					   sampler=sampler, shuffle=False, batch_size = self.batch_size, num_workers=num_workers)

		del datax, datay, dataz, dx, dy, dz, data_xyz
		return trainloader

	def LoadSplitTrain(self, loadtype='test'):
		"""
		performs the splitting for the training and the validation sets
		"""
		if loadtype == 'test':
			self.loader['test'] = self._LoadSplitTrain(self.train_test_idxs['test'])
		else:
			self.loader['train'] = self._LoadSplitTrain(self.train_test_idxs['train'])
	def LoadWeights(self, filename):
		"""
		Load weights from file.
		"""
		self.model.load_state_dict(torch.load(filename))
	def _InitialiseWeights(self, m):
		if isinstance(m, nn.Conv3d):
			self._initfn(m.weight.data, **self._initfn_kwargs)
			if m.bias is not None:
				nn.init.constant_(m.bias.data, 0)
		elif isinstance(m, nn.BatchNorm3d):
			nn.init.constant_(m.weight.data, 1)
			nn.init.constant_(m.bias.data, 0)
	def InitialiseWeights(self, normfn, **kwargs):
		"""
		Apply weight initialisation to the model.
		"""
		self._initfn = normfn
		self._initfn_kwargs = kwargs
		self.model.apply(self._InitialiseWeights)
	def InitialiseWeights2(self):
		for p in self.model.parameters():
			p.data.normal_(0.02,0.01)
	def SetLRStepSize(self, lrate_step_size):
		"""
		Set learning rate step size; i.e. number of 
		epochs after which the learning rate will update.
		"""
		self.hyperpars['lrate_step_size'] = lrate_step_size
	def AddLR(self, lr):
		"""
		Add learning rate.  Must occur before 
		corresponding AddOptimiser.
		"""
		n = 1
		lr_str = 'lr%d'%n
		while lr_str in self.hyperpars.keys():
			n += 1
			lr_str = 'lr%d'%n
		self.hyperpars[lr_str] = lr
	def RemoveLR(self):
		"""
		Remove previously added learning rate.  
		"""
		n = 1
		lr_str = 'lr%d'%n
		while lr_str in self.hyperpars.keys():
			n += 1
			lr_str = 'lr%d'%n
		if n > 1:
			self.hyperpars.pop('lr%d'%(n-1), None)
	def SetGamma(self, gamma):
		"""
		After n epochs the LR will be multiplied by gamma.
		"""
		self.hyperpars['gamma'] = gamma
	def AddOptimiser(self, optimiser, **kwargs):
		"""
		Add an optimiser. Must occur before 
		corresponding AddScheduler. 
		Specify key words for optimiser.
		"""
		n = len(self.optimisers)
		optimiser_args = self.GetKwArgs(optimiser, kwargs)
		if not "lr" in optimiser_args:
			optimiser_args['lr'] = self.hyperpars['lr%d'%(n+1)]
		self.optimisers['optimiser%d'%(n+1)] = optimiser(self.model.parameters(), **optimiser_args)
	def RemoveOptimiser(self):
		"""
		Remove previously added optimiser.
		"""
		if len(self.optimisers) > 0:
			self.optimisers.popitem()
	def AddScheduler(self, scheduler, **kwargs):
		"""
		Add scheduler function to schedule LR update.
		StepLR just multiplies LR by gamma by n epochs.
		"""
		n = len(self.schedulers)
		scheduler_args = self.GetKwArgs(scheduler, kwargs)
		if scheduler is ss.StepLR and not "step_size" in scheduler_args:
			scheduler_args['step_size'] = self.hyperpars['lrate_step_size']
		if scheduler is ss.StepLR and not "gamma" in scheduler_args:
			scheduler_args['gamma'] = self.hyperpars['gamma']
		self.schedulers['scheduler%d'%(n+1)] = scheduler(self.optimisers['optimiser%d'%(n+1)], **scheduler_args)
	def RemoveScheduler(self):
		"""
		Remove Scheduler.
		"""
		if len(self.schedulers) > 0:
			self.schedulers.popitem()
	def chi_loss(self, output, target):
		"""
		Compute the chi squared error of two sets of data.
		"""
		loss = torch.mean(torch.abs((output-target))**2)/(torch.mean(target**2)+1e-40)
		return loss 
	def pcc_loss(self, output, target):
		"""
		Pearson correlation coefficient.
		"""
		x = torch.abs(output)
		y = torch.abs(target)
		vx = torch.abs(x - torch.mean(x))
		vy = torch.abs(y - torch.mean(y))
		loss = torch.mean(vx * vy) / (torch.sqrt(torch.mean(vx ** 2) * torch.mean(vy ** 2))+1e-40)
		return 1.0 - loss
	def all_loss(self, output, target, input, alpha=1.0, beta=1.0, gamma=1.0, rs_pcc=False):
		"""
		Total loss in the training. 
		loss1: amplitude channel.
		loss2: phase channel.
		loss3: fourier transform of the object.
		
		All losses are merged together in a single loss function.
		"""
		_, __, X,Y,Z = self.data['input_data'].shape
		X2 = X//2
		X4 = X//4
		Y2 = Y//2
		Y4 = Y//4
		Z2 =  Z//2
		Z4 = Z//4
		
		if rs_pcc:
			loss1 = self.pcc_loss(output[:, 0, :, :, :], target[:, 0, :, :, :])
			loss2 = self.pcc_loss(output[:, 1, :, :, :], target[:, 1, :, :, :])
		else:
			loss1 = self.chi_loss(output[:, 0, :, :, :], target[:, 0, :, :, :])
			loss2 = self.chi_loss(output[:, 1, :, :, :], target[:, 1, :, :, :])

		obj_comp = torch.zeros((output.shape[0]), 2, X, Y, Z, requires_grad=False, device = self.device) 
		obj_comp[:, 0, (X2-X4):(X2+X4), (Y2-Y4):(Y2+Y4), (Z2 - Z4):(Z2 + Z4)] = output[:, 0, :, :, :] * torch.cos(2*torch.pi * (output[:,1,:,:,:]))
		obj_comp[:, 1, (X2-X4):(X2+X4), (Y2-Y4):(Y2+Y4), (Z2 - Z4):(Z2 + Z4)] = output[:, 0, :, :, :] * torch.sin(2*torch.pi * (output[:,1,:,:,:]))
		obj_comp = obj_comp[:,0,:,:,:] +1j * obj_comp[:,1,:,:,:]
		obj_comp = torch.fft.fftn(obj_comp, dim= (-3,-2,-1))

		amp_out = torch.sqrt(torch.abs(obj_comp[:,:,:,:]) **2 + torch.abs(obj_comp[:,:,:,:]) **2 +1e-40)
		loss3 = self.pcc_loss(amp_out, input) 
		
		loss = (alpha * loss1 + beta * loss2 + gamma * loss3) / (alpha + beta + gamma)
		return loss
		
	def criterion(self, output, target, input, alpha=1.0, beta=1.0, gamma=1.0, rs_pcc=False):
		"""
		Call the desired loss function.
		"""
		return self.all_loss(output, target, input, alpha=alpha, beta=beta, gamma=gamma, rs_pcc=rs_pcc)

	def GetLR(self, optimiser):
		"""
		Get learning rate.
		"""
		for param_group in optimiser.param_groups:
			return param_group['lr']

	def SetNEpochs(self, epochs):
		"""
		Set number of loops over the entire 
		training data set during the training.
		"""
		self.hyperpars['epochs'] = epochs
	def SetOpStep(self, step):
		"""
		Set the optimiser step size in epochs after
		which the optimiser is changed.  
		"""
		self.hyperpars['op_step_size'] = int(step)
	def TrainNN(self, **loss_params):
		"""
		Perform the training of the network.
		loss_params = {alpha=1.0, # loss ratio amp
									beta=1.0, # loss ratio phase
									gamma=1.0, # loss ratio Fourier amp
									rs_pcc=False # Use Pearson for Real space}
		"""
		self.datestr = strftime("%Y-%m-%d_%H.%M")
		loss_args = self.GetKwArgs(self.criterion, loss_params)
		for epoch in range(self.hyperpars['epochs']):  # loop over the dataset multiple times
			train_loss_tmp = 0.0
			self.model.train()
			# #
			sw_op = len(self.optimisers)
			sw_sch = len(self.schedulers)
			sw_op_flag = (epoch // self.hyperpars['op_step_size']) % sw_op
			for ii, loader_batch_train in enumerate(self.loader['train'], 0):
				# Get inputs.  Note: 'Variable' call is now depreciated.  
				x_train, y_train, z_train = loader_batch_train
				x_train, y_train, z_train = Variable(x_train).to(self.device), Variable(y_train).to(self.device), Variable(z_train).to(self.device)
				#for param in self.model.parameters():
				#	param.grad = None
				# Set all the gradients to zero; to avoid accumulation 
				# of gradients from the previous epoch.
				for idi in range(sw_op):
					if sw_op_flag == idi:
						list(self.optimisers.values())[idi].zero_grad()
				# Forward propagation
				y_train_predict = self.model.forward(x_train)
				# Define the loss and then backward propagate
				loss1 = self.criterion(y_train_predict, y_train, z_train, **loss_args)
				loss1.backward()
				#incorporate a clip on the values of the gradients, to avoid exploding gradients 
				# Not recognised
				#clip_grad_norm_(self.model.coder.parameters(), 2)
				#clip_grad_norm_(self.model.ppha.parameters(), 2)
				#clip_grad_norm_(self.model.aamp.parameters(), 1.25)
				clip_grad_norm_(self.model.parameters(), max_norm = 10.0, norm_type=2)
				# Optimise the weights and biases
				for idi in range(sw_op):
					if sw_op_flag == idi:
						list(self.optimisers.values())[idi].step()
				# Sum losses
				train_loss_tmp += loss1.item()
				# print info if needed
				if self.verbose:
					if ii % self.print_every == 0:
						print('Epoch: %d, Batch: %5d, Batch loss: train: %.5f'%(epoch + 1, ii + 1, train_loss_tmp / (ii + 1)))
				# # Loop end
			# Update learning rate with scheduler
			for idi in range(sw_sch):
				if sw_op_flag == idi:
					list(self.schedulers.values())[idi].step()
					lr = self.GetLR(list(self.optimisers.values())[idi])
			# Validation step
			with torch.no_grad():
				valid_loss_tmp = 0.0
				self.model.eval() # turn off some specific parts of the model for the evaluation with model.eval()
				for loader_batch_test in self.loader['test']:
					x_test, y_test, z_test = loader_batch_test
					x_test, y_test, z_test = x_test.to(self.device), y_test.to(self.device), z_test.to(self.device)
					y_pred = self.model.forward(x_test)
					loss2 = self.criterion(y_pred, y_test, z_test, **loss_args)
					valid_loss_tmp += loss2.item()
			# Update graph data
			self.train_loss.append(train_loss_tmp / len(self.loader['train']))
			self.valid_loss.append(valid_loss_tmp / len(self.loader['test']))
			# Check for instability
			if np.isfinite(self.train_loss[-1]):
				pass
			elif self.verbose:
				print('Model is unstable.')
				break
			# Print some useful information
			if self.verbose:
				print('Epoch-loss: train %.5f  valid %.5f lr %.2e' %(self.train_loss[-1], self.valid_loss[-1], lr)) 
			# Save
			if (epoch != 0) and (epoch % (self.hyperpars['epochs'] -1) == 0):
				self.SaveModel(epoch)

	def SaveModel(self, epoch=0):
		"""
		Save model.
		"""
		torch.save(self.model.state_dict(),'CP{}'.format(epoch+1)+'_'+self.datestr+'.pth')
	def SaveParameters(self):
		"""
		Save parameters.
		"""
		params = ""
		params += "Device Type: %s \n"%self.device.type
		params += "Cuda device count: %d \n"%self.cuda_device_count
		for key, value in self.optimisers.items():
			params += "Optimiser: %s \n"%value.__class__.__name__
		params += "Validation Size: %1.5e  \n"%self.valid_size
		params += "Batch Size: %2.6f \n" %self.batch_size
		for key, value in self.hyperpars.items():
			if key.startswith('lr') and key != 'lrate_step_size':
				params += "Learning Rate: %2.6f \n" %value
		params += "Learning Rate Step Size: %2.6f \n" %self.hyperpars['lrate_step_size']
		params += "Gamma: %2.6f \n" %self.hyperpars['gamma']
		params += "Number of Epochs: %d \n" %self.hyperpars['epochs']
		params += "Valdiation loss: %2.4f \n" %self.valid_loss[-1]
		params += "Training Loss: %2.4f \n" %self.train_loss[-1]
		params += "-"*20
		# #
		f = open('CNN_Training_Params_'+self.datestr+'.txt', "w")
		f.write(params)
		f.close()
	def SaveLoss(self):
		lossdata = np.array([self.train_loss, self.valid_loss])
		np.save('lossdata_'+self.datestr+'.npy', lossdata)
	def PlotLoss(self, training = True):
		"""
		Plot the loss values.
		"""
		if training:
			plt.plot(self.train_loss, label='Training loss')
			plt.plot(self.valid_loss, label='Validation loss')
		else:
			plt.plot(self.train_loss, label='Prediction loss')
		plt.yscale('log')
		plt.legend(frameon=False)
		plt.savefig('validation_error_'+self.datestr+'.png')
		#plt.show()


class CNNPredictOld():
	"""
	Prediction from trained neural network.
	"""
	def __init__(self):
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.expdata = None
		self.trained_network = None
		self.output = None
		self.model = None
	def GetKwArgs(self, obj, kwargs):
		obj_sigs = []
		obj_args = {}
		for arg in inspect.signature(obj).parameters.values():
			if not arg.default is inspect._empty:
				obj_sigs.append(arg.name)
		for key, value in kwargs.items():
			if key in obj_sigs:
				obj_args[key] = value
		return obj_args
	def SetDevice(self, device='cuda'):
		"""
		Sets the device to either CPU ('cpu') or GPU ('cuda'), if available.
		"""
		if torch.cuda.is_available() and device == 'cuda':
			self.device = torch.device("cuda")
		else:
			self.device = torch.device("cpu")
	def SetModel(self, model, **kwargs):
		"""
		Selecting the model to be used for the network.
		Keywords are passed to model if they exist in the model.
		"""
		model_args = self.GetKwArgs(model, kwargs)
		self.model = model(**model_args).to(self.device)
		if self.device.type == "cuda" and torch.cuda.device_count() > 1:
			self.model = nn.DataParallel(self.model)
	def SetExpData(self, fname, mask=100, square_root=True):
		"""
		Set diffraction data used for prediction.
		Mask out the noise by selecting a threshold 
		value below which everything is set to zero.
		Normalizes the array to [0,1] interval.
		Square root if data is intensity.
		"""
		self.expdata = np.abs(np.load(fname))
		self.expdata[self.expdata < mask] = 0.0
		if square_root:
			self.expdata = np.sqrt(self.expdata)
		max = 1.0/self.expdata.max()
		self.expdata = self.expdata * max
	def SetTrainedNN(self, fname):
		"""
		Load the trained neural network.
		Must be a .pth file.
		"""
		self.trained_network = fname
	def SetOutputFile(self, fname):
		"""
		Set output file, i.e. the 
		reconstructed object.
		"""
		self.output = fname
	def Predict(self):
		"""
		Forward propagate the diffraction pattern 
		through the trained neural network and  
		obtain an output complex object 
		"""
		i = self.expdata.shape[0]
		j = self.expdata.shape[1]
		k = self.expdata.shape[2]

		torcharray = np.zeros((1,1,i,j,k), dtype=np.float32)
		torcharray[0,0,:,:,:]  = self.expdata[:,:,:]
		torcharray = torch.from_numpy(torcharray)
		
		if self.device.type == 'cuda':
			self.model.load_state_dict(torch.load(self.trained_network))
			torcharray = torcharray.to(device = self.device, dtype = torch.float)
		else:
			self.model.load_state_dict(torch.load(self.trained_network, map_location = 'cpu'))

		self.model.eval()
			
		with torch.no_grad():
			sequence = self.model(torcharray)
			
		sequence = sequence.cpu()

		amp = np.zeros((i//2,j//2,k//2), dtype=np.double)
		pha = np.zeros((i//2,j//2,k//2), dtype=np.double)

		amp[:] = sequence[0,0,:,:,:]
		pha[:] = sequence[0,1,:,:,:] * 2.0 * np.pi
		pha[:] -= np.pi

		com = amp * np.cos(pha) + 1j * amp * np.sin(pha)

		np.save(self.output, com)


class CNNPredict(CNNTrain):
	"""
	Prediction from trained neural network.
	"""
	def __init__(self):
		super().__init__()
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.expdata = None
		self.torcharray = None
		self.trained_network = None
		self.output = None
		self.model = None
	def SetExpData(self, fname, mask=100, square_root=True):
		"""
		Set diffraction data used for prediction.
		Mask out the noise by selecting a threshold 
		value below which everything is set to zero.
		Normalizes the array to [0,1] interval.
		Square root if data is intensity.
		"""
		self.expdata = np.abs(np.load(fname))
		self.expdata[self.expdata < mask] = 0.0
		if square_root:
			self.expdata = np.sqrt(self.expdata)
		max = 1.0/self.expdata.max()
		self.expdata = self.expdata * max
		
		self.expdata[:] = np.fft.fftshift(self.expdata)

		i,j,k = self.expdata.shape
		self.torcharray = np.zeros((1,1,i,j,k), dtype=np.float32)
		self.torcharray[0,0,:,:,:]  = self.expdata[:,:,:]
		self.torcharray = torch.from_numpy(self.torcharray)

		del self.expdata 
	def SetTrainedNN(self, fname):
		"""
		Load the trained neural network.
		Must be a .pth file.
		"""
		self.trained_network = fname

		if self.device.type == 'cuda':
			self.model.load_state_dict(torch.load(self.trained_network))
			self.torcharray = self.torcharray.to(device = self.device, dtype = torch.float)
		else:
			self.model.load_state_dict(torch.load(self.trained_network, map_location = 'cpu'))
	def SetOutputFile(self, fname):
		"""
		Set output file, i.e. the 
		reconstructed object.
		"""
		self.output = fname
	def all_loss(self, output, input):
		X,Y,Z = self.expdata.shape
		X2 = X//2
		X4 = X//4
		Y2 = Y//2
		Y4 = Y//4
		Z2 =  Z//2
		Z4 = Z//4

		obj_comp = torch.zeros((output.shape[0]), 2, X, Y, Z, requires_grad=False, device = self.device) 
		obj_comp[:, 0, (X2-X4):(X2+X4), (Y2-Y4):(Y2+Y4), (Z2 - Z4):(Z2 + Z4)] = output[:, 0, :, :, :] * torch.cos(2*torch.pi * (output[:,1,:,:,:]))
		obj_comp[:, 1, (X2-X4):(X2+X4), (Y2-Y4):(Y2+Y4), (Z2 - Z4):(Z2 + Z4)] = output[:, 0, :, :, :] * torch.sin(2*torch.pi * (output[:,1,:,:,:]))
		obj_comp = obj_comp[:,0,:,:,:] +1j * obj_comp[:,1,:,:,:]
		obj_comp = torch.fft.fftn(obj_comp, dim= (-3,-2,-1))

		amp_out = torch.sqrt(torch.abs(obj_comp[:,:,:,:]) **2 + torch.abs(obj_comp[:,:,:,:]) **2 +1e-40)
		loss = self.pcc_loss(amp_out, input) 
		del obj_comp
		del amp_out
		return loss
	def criterion(self, output, input):
		"""
		Call the desired loss function.
		"""
		return self.all_loss(output, input)	
	def Predict(self):
		"""
		Forward propagate the diffraction pattern 
		through the trained neural network and  
		obtain an output complex object 
		"""
		self.model.eval()
			
		with torch.no_grad():
			sequence = self.model(self.torcharray)
			
		sequence = sequence.cpu()

		i,j,k = self.expdata.shape
		amp = np.zeros((i//2,j//2,k//2), dtype=np.double)
		pha = np.zeros((i//2,j//2,k//2), dtype=np.double)

		amp[:] = sequence[0,0,:,:,:]
		pha[:] = sequence[0,1,:,:,:] * 2.0 * np.pi
		pha[:] -= np.pi

		com = amp * np.cos(pha) + 1j * amp * np.sin(pha)

		np.save(self.output, com)

	def TransferPredict(self):
		scaler = GradScaler(init_scale=1.0, growth_factor=2.0, backoff_factor=0.5, growth_interval=100, enabled=True)
		for epoch in range(self.hyperpars['epochs']):  # loop over the dataset multiple times
			train_loss_tmp = 0.0
			self.model.train()
			# #
			sw_op = len(self.optimisers)
			sw_sch = len(self.schedulers)
			sw_op_flag = (epoch // self.hyperpars['op_step_size']) % sw_op
			
			for idi in range(sw_op):
				if sw_op_flag == idi:
					list(self.optimisers.values())[idi].zero_grad()
			with torch.cuda.amp.autocast():
				# Forward propagation
				y_train_predict = self.model.forward(self.torcharray)
				# Define the loss and then backward propagate
				loss1 = self.criterion(y_train_predict, self.torcharray)
			#loss1.backward()
			scaler.scale(loss1).backward()
			#incorporate a clip on the values of the gradients, to avoid exploding gradients 
			clip_grad_norm_(self.model.parameters(), max_norm = 10.0, norm_type=2)
			# Optimise the weights and biases
			for idi in range(sw_op):
				if sw_op_flag == idi:
					#list(self.optimisers.values())[idi].step()
					scaler.step(list(self.optimisers.values())[idi])
			scaler.update()
			# Sum losses
			train_loss_tmp += loss1.item()
			# Update learning rate with scheduler
			for idi in range(sw_sch):
				if sw_op_flag == idi:
					list(self.schedulers.values())[idi].step()
					lr = self.GetLR(list(self.optimisers.values())[idi])
			# Update graph data
			self.train_loss.append(train_loss_tmp)
			# Check for instability
			if np.isfinite(self.train_loss[-1]):
				pass
			elif self.verbose:
				print('Model is unstable.')
				break
			# Print some useful information
			if self.verbose:
				print('Epoch-loss: train %.5f lr %.2e' %(self.train_loss[-1], lr)) 
			# Save
			if epoch % (self.hyperpars['epochs'] -1) == 0:
				self.model.eval()
				with torch.no_grad():
					sequence = self.model.forward(self.torcharray)

				sequence = sequence.cpu()

				i,j,k = self.expdata.shape
				amp = np.zeros((i//2,j//2,k//2), dtype=np.double)
				pha = np.zeros((i//2,j//2,k//2), dtype=np.double)

				amp[:] = sequence[0,0,:,:,:]
				pha[:] = sequence[0,1,:,:,:] * 2.0 * np.pi
				pha[:] -= np.pi

				com = amp * np.cos(pha) + 1j * amp * np.sin(pha)

				np.save(self.output, com)
	def SaveTrainLoss(self):
		lossdata = np.array(self.train_loss)
		np.save('trainlossdata_'+self.datestr+'.npy', lossdata)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 
def Train():
	cnn = CNNTrain()
	cnn.SetDevice('cuda')
	cnn.SetInputData('fs_amps.npy')
	cnn.SetTargetData('rs_objs.npy')
	cnn.SetModel(NNModel, checkpoints=False)
	cnn.SetValidSize(0.1)
	cnn.SplitData()
	cnn.SetBatchSize(5)
	cnn.LoadSplitTrain(loadtype='train')
	cnn.LoadSplitTrain(loadtype='test')
	#cnn.LoadWeights("CP.pth")
	cnn.InitialiseWeights(nn.init.kaiming_normal_, mode='fan_in', nonlinearity='leaky_relu')
	#cnn.InitialiseWeights(nn.init.xavier_normal_)
	cnn.SetLRStepSize(25)
	cnn.AddLR(1e-4)
	cnn.AddLR(1e-6)
	cnn.SetGamma(0.75)
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

def Predict():
	predict = CNNPredict()
	predict.SetDevice('cuda')
	#predict.SetDevice('cpu')
	predict.SetModel(NNModel, checkpoints=True)
	predict.SetExpData('expdata.npy', mask=500, square_root=True)
	predict.SetTrainedNN("CP.pth")
	predict.SetOutputFile('output.npy')
	predict.SetLRStepSize(25)
	predict.AddLR(1e-3)
	predict.AddLR(5e-4)
	predict.AddLR(1e-4)
	predict.AddLR(5e-5)
	predict.SetGamma(0.9)
	predict.AddOptimiser(optim.ASGD)
	predict.AddOptimiser(optim.Adam, amsgrad=True, eps=1e-8)
	predict.AddOptimiser(optim.ASGD)
	predict.AddOptimiser(optim.Adam, amsgrad=True, eps=1e-8)
	predict.AddScheduler(ss.StepLR)
	predict.AddScheduler(ss.StepLR)
	predict.AddScheduler(ss.StepLR)
	predict.AddScheduler(ss.StepLR)
	predict.SetNEpochs(100)
	predict.SetOpStep(10)
	predict.TransferPredict()
	predict.SaveTrainLoss()
	predict.PlotLoss(training=False)


if __name__ == '__main__':
	Train()
	#Predict()



