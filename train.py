# ###########################################
# Filename: train.py
# Training class for CNN phase retrieval.
# Derived from work by Longlong Wu.
#
# Authors: Ahmed H. Mokhtar, Marcus Newton
#
# Version 0.13_beta
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
from torch.autograd import Variable
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_


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
		self.amp_loss_train = []
		self.pha_loss_train = []
		self.diff_loss_train = []
		self.amp_loss_val = []
		self.pha_loss_val = []
		self.diff_loss_val = []
		self.verbose = True
		self.print_every = 100
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
			torch.cuda.empty_cache()
			self.device = torch.device("cuda")
		else:
			self.device = torch.device("cpu")
	def SetInputData(self, fname, add_noise=None):
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

		if add_noise == 'Gaussian':
			print('Adding Gaussian noise to diffraction data ...')
			for i in range(0,shp[0]):
				mean = np.mean(self.data['input_data'][i,0,:,:,:])
				std_dev = np.std(self.data['input_data'][i,0,:,:,:])

				self.data['input_data'][i,0,:,:,:] = self.data['input_data'][i,0,:,:,:]**2 + np.random.normal(mean/2,std_dev/2,(shp[-3],shp[-2],shp[-1]))
				self.data['input_data'][i,0,:,:,:] = np.sqrt(self.data['input_data'][i,0,:,:,:])
			print('Adding Gaussian noise: Complete')

		if add_noise == 'Poisson':
			print('Adding Poisson noise to diffraction data ...')
			for i in range(0,shp[0]):
				mean = np.mean(self.data['input_data'][i,0,:,:,:])
				std_dev = np.std(self.data['input_data'][i,0,:,:,:])

				self.data['input_data'][i,0,:,:,:] = self.data['input_data'][i,0,:,:,:]**2 + np.random.poisson(mean/2,(shp[-3],shp[-2],shp[-1]))
				self.data['input_data'][i,0,:,:,:] = np.sqrt(self.data['input_data'][i,0,:,:,:])
			print('Adding Poisson noise: Complete')

		if add_noise == None:
			print('Not adding noise')
			pass
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
		if self.device.type == "cuda":
			cuda_device_count = torch.cuda.device_count()
			if cuda_device_count > 1:
				self.cuda_device_count = cuda_device_count
				self.model = nn.DataParallel(self.model)
				self.model.to(self.device)
			print('Using %s'%torch.cuda.get_device_name(0))
		else:
			print('Using %s'%self.device.type)

		self.model = self.model.float()
	def SetValidSize(self, valid_size):
		"""
		Set the validation size of the training set
		"""
		if valid_size >= 0.0 and valid_size < 1.0:
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
	def AddGamma(self, gamma):
		"""
		Per scheduler gamma
		After n epochs the LR will be multiplied by gamma.
		"""
		n = 1
		gamma_str = 'gamma%d'%n
		while gamma_str in self.hyperpars.keys():
			n += 1
			gamma_str = 'gamma%d'%n
		self.hyperpars[gamma_str] = gamma
	def RemoveGamma(self):
		"""
		Remove previously added gamma.
		"""
		n = 1
		gamma_str = 'gamma%d'%n
		while gamma_str in self.hyperpars.keys():
			n += 1
			gamma_str = 'gamma%d'%n
		if n > 1:
			self.hyperpars.pop('gamma%d'%(n-1), None)
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
		import torch.optim.lr_scheduler as ss
		n = len(self.schedulers)
		scheduler_args = self.GetKwArgs(scheduler, kwargs)
		if scheduler is ss.StepLR and not "step_size" in scheduler_args:
			scheduler_args['step_size'] = self.hyperpars['lrate_step_size']
		if scheduler is ss.StepLR and not "gamma" in scheduler_args:
			gamma_str = 'gamma%d'%(n+1)
			if gamma_str in self.hyperpars:
				scheduler_args['gamma'] = self.hyperpars[gamma_str]
			else:
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

	def all_loss(self, output, target, input, alpha=1.0, beta=1.0, gamma=4.0, rs_pcc=False):
		"""
		Total loss in the training.
		loss1: amplitude channel.
		loss2: phase channel.
		loss3: fourier transform of the object.

		All losses are merged together in a single loss function.
		"""
		_, _, X,Y,Z = self.data['input_data'].shape
		X2 = X//2
		X4 = X//4
		Y2 = Y//2
		Y4 = Y//4
		Z2 = Z//2
		Z4 = Z//4

		if rs_pcc:
			loss1 = self.pcc_loss(output[:, 0, :, :, :], target[:, 0, :, :, :])
			loss2 = self.pcc_loss(output[:, 1, :, :, :], target[:, 1, :, :, :])
		else:
			loss1 = self.chi_loss(output[:, 0, :, :, :], target[:, 0, :, :, :])
			loss2 = self.chi_loss(output[:, 1, :, :, :], target[:, 1, :, :, :])

		obj_comp = torch.zeros((output.shape[0]), 2, X, Y, Z, requires_grad=False, device = self.device)
		obj_comp[:, 0, (X2-X4):(X2+X4), (Y2-Y4):(Y2+Y4), (Z2 - Z4):(Z2 + Z4)] = output[:, 0, :, :, :] * torch.cos(torch.pi * (output[:,1,:,:,:]))
		obj_comp[:, 1, (X2-X4):(X2+X4), (Y2-Y4):(Y2+Y4), (Z2 - Z4):(Z2 + Z4)] = output[:, 0, :, :, :] * torch.sin(torch.pi * (output[:,1,:,:,:]))
		obj_comp = obj_comp[:,0,:,:,:] +1j * obj_comp[:,1,:,:,:]
		obj_comp = torch.fft.fftn(obj_comp, dim= (-3,-2,-1))

		amp_out = torch.abs(obj_comp)
		loss3 = self.pcc_loss(amp_out, input)

		loss = (alpha * loss1 + beta * loss2 + gamma * loss3 ) / (alpha + beta + gamma)
		return loss1, loss2, loss3, loss

	def criterion(self, output, target, input, alpha=1.0, beta=1.0, gamma=4.0, rs_pcc=False):
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
	def AddOpStep(self, step):
		"""
		Set the optimiser step size in epochs after
		which the optimiser is changed.
		"""
		n = 1
		op_step_size_str = 'op_step_size%d'%n
		while op_step_size_str in self.hyperpars.keys():
			n += 1
			op_step_size_str = 'op_step_size%d'%n
		self.hyperpars[op_step_size_str] = int(step)
	def RemoveOpStep(self):
		"""
		Remove previously added optimiser step .
		"""
		n = 1
		op_step_size_str = 'op_step_size%d'%n
		while op_step_size_str in self.hyperpars.keys():
			n += 1
			op_step_size_str = 'op_step_size%d'%n
		if n > 1:
			self.hyperpars.pop('op_step_size%d'%(n-1), None)
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
			train_amp_tmp = 0.0
			train_pha_tmp = 0.0
			train_diff_tmp = 0.0
			self.model.eval()
			# #
			sw_op = len(self.optimisers)
			sw_sch = len(self.schedulers)
			# #
			op_n = 1
			op_step_size_str = 'op_step_size%d'%op_n
			op_step_sum = 0
			while op_step_size_str in self.hyperpars:
				op_step_sum += self.hyperpars[op_step_size_str]
				if epoch < op_step_sum or op_n == sw_op:
					sw_op_flag = op_n - 1
					break
				op_n += 1
				op_step_size_str = 'op_step_size%d'%op_n
			if 'op_step_size' in self.hyperpars:
				sw_op_flag = min(epoch // self.hyperpars['op_step_size'], sw_op - 1)
			# #
			for ii, loader_batch_train in enumerate(self.loader['train'], 0):
				# Get inputs.  Note: 'Variable' call is now depreciated.
				x_train, y, z_train = loader_batch_train
				x_train, y, z_train = Variable(x_train).to(self.device), Variable(y).to(self.device), Variable(z_train).to(self.device)
				#for param in self.model.parameters():
				#	param.grad = None
				# Set all the gradients to zero; to avoid accumulation
				# of gradients from the previous epoch.
				for idi in range(sw_op):
					if sw_op_flag == idi:
						list(self.optimisers.values())[idi].zero_grad()
				# Forward propagation
				y_predict = self.model.forward(x_train)
				# Define the loss and then backward propagate
				loss_tra1, loss_tra2, loss_tra3, loss_train = self.criterion(y_predict, y, z_train, **loss_args)
				loss_train.backward()
				#incorporate a clip on the values of the gradients, to avoid exploding gradients
				clip_grad_norm_(self.model.parameters(), max_norm = 10.0, norm_type=2)
				# Optimise the weights and biases
				for idi in range(sw_op):
					if sw_op_flag == idi:
						list(self.optimisers.values())[idi].step()
				# Sum losses
				train_loss_tmp += loss_train.item()
				train_amp_tmp += loss_tra1.item()
				train_pha_tmp += loss_tra2.item()
				train_diff_tmp += loss_tra3.item()
				# print info if needed
				if self.verbose:
					if ii % self.print_every == 0:
						print('Epoch: %d, Batch: %5d, Batch loss: loss_train: %.5f, diff loss: %.5f, amp_loss: %.5f, phase_loss: %.5f'%(epoch + 1, ii + 1, train_loss_tmp / (ii + 1), train_diff_tmp/(ii+1), train_amp_tmp/(ii+1), train_pha_tmp/(ii+1)))
				# # Loop end
			# Update learning rate with scheduler
			for idi in range(sw_sch):
				if sw_op_flag == idi:
					list(self.schedulers.values())[idi].step()
					lr = self.GetLR(list(self.optimisers.values())[idi])
			# Validation step
			with torch.no_grad():
				valid_loss_tmp = 0.0
				valid_amp_tmp = 0.0
				valid_pha_tmp = 0.0
				valid_diff_tmp = 0.0
				self.model.eval() # turn off some specific parts of the model for the evaluation with model.eval()
				for loader_batch_test in self.loader['test']:
					x_test, y_test, z_test = loader_batch_test
					x_test, y_test, z_test = x_test.to(self.device), y_test.to(self.device), z_test.to(self.device)
					y_pred = self.model.forward(x_test)
					loss_val1, loss_val2, loss_val3, loss_val = self.criterion(y_pred, y_test, z_test, **loss_args)
					valid_loss_tmp += loss_val.item()
					valid_amp_tmp += loss_val1.item()
					valid_pha_tmp += loss_val2.item()
					valid_diff_tmp += loss_val3.item()
			# Update graph data
			self.train_loss.append(train_loss_tmp / len(self.loader['train']))
			self.amp_loss_train.append(train_amp_tmp / len(self.loader['train']))
			self.pha_loss_train.append(train_pha_tmp / len(self.loader['train']))
			self.diff_loss_train.append(train_diff_tmp / len(self.loader['train']))

			if self.train_test_idxs['test']:
				self.valid_loss.append(valid_loss_tmp / len(self.loader['test']))
				self.amp_loss_val.append(valid_amp_tmp / len(self.loader['test']))
				self.pha_loss_val.append(valid_pha_tmp / len(self.loader['test']))
				self.diff_loss_val.append(valid_diff_tmp / len(self.loader['test']))
			else:
				self.valid_loss.append(0.0)
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
			if epoch == (self.hyperpars['epochs'] -1):
				self.SaveModel(epoch)


	def SaveModel(self, epoch=0):
		"""
		Save model.
		"""
		os.mkdir(self.datestr)
		torch.save(self.model.state_dict(),self.datestr+'/'+'CP{}'.format(epoch+1)+'_'+self.datestr+'.pth')
	def SaveParameters(self, training=True):
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
				params += "Learning Rate: %1.4e \n" %value
		params += "Learning Rate Step Size: %2.6f \n" %self.hyperpars['lrate_step_size']
		for key, value in self.hyperpars.items():
			if key.startswith('gamma') and key != 'gamma':
				params += "Gamma: %2.6f \n" %value
		if 'gamma' in self.hyperpars:
			params += "Gamma (global): %2.6f \n" %self.hyperpars['gamma']
		params += "Number of Epochs: %d \n" %self.hyperpars['epochs']
		for key, value in self.hyperpars.items():
			if key.startswith('op_step_size') and key != 'op_step_size':
				params += "Optimiser Step Size: %d \n" %value
		if 'op_step_size' in self.hyperpars:
			params += "Optimiser Step Size (global): %d \n" %self.hyperpars['op_step_size']
		if training:
			params += "Valdiation Loss: %2.4f \n" %self.valid_loss[-1]
		params += "Training Loss: %2.4f \n" %self.train_loss[-1]
		params += "-"*20
		# #
		f = open(self.datestr+'/CNN_Training_Params_'+self.datestr+'.txt', "w")
		f.write(params)
		f.close()
	def SaveLoss(self):
		lossdata = np.array([self.train_loss, self.valid_loss, self.amp_loss_train, self.pha_loss_train, self.diff_loss_train, self.amp_loss_val, self.pha_loss_val, self.diff_loss_val])
		np.save(self.datestr+'/'+'lossdata_'+self.datestr+'.npy', lossdata)
	def PlotLoss(self, training=True):
		"""
		Plot the loss values.
		"""
		if training:
			fig1 = plt.figure()
			plt.plot(self.train_loss, label = 'Training Loss')
			plt.plot(self.valid_loss, label = 'Validation Loss')
			plt.yscale('log')
			plt.legend(frameon=False)
			plt.savefig(self.datestr+'/'+'Full_error_'+self.datestr+'.png')

			fig2 = plt.figure()
			plt.plot(self.amp_loss_train, label = 'Amplitude Training Loss')
			plt.plot(self.pha_loss_train, label = 'Phase Training Loss')
			plt.plot(self.diff_loss_train, label = 'Fourier Space Training Loss')
			plt.yscale('log')
			plt.legend(frameon=False)
			plt.savefig(self.datestr+'/'+'Training_error_'+self.datestr+'.png')

			fig3 = plt.figure()
			plt.plot(self.amp_loss_val, label = 'Amplitude Validation Loss')
			plt.plot(self.pha_loss_val, label = 'Phase Validation Loss')
			plt.plot(self.diff_loss_val, label = 'Fourier Space Validation Loss')
			plt.yscale('log')
			plt.legend(frameon=False)
			plt.savefig(self.datestr+'/'+'Validation_error_'+self.datestr+'.png')

			fig4 = plt.figure()
			plt.plot(self.diff_loss_train, label = 'Diffraction Training Loss')
			plt.plot(self.diff_loss_val, label = 'Diffraction Validation Loss')
			plt.yscale('log')
			plt.legend(frameon=False)
			plt.savefig(self.datestr+'/'+'Diffraction_error_'+self.datestr+'.png')

		else:
			plt.plot(self.train_loss, label='Prediction loss')
			plt.yscale('log')
			plt.legend(frameon=False)
			plt.savefig(self.datestr+'/'+'Prediction_error_'+self.datestr+'.png')
