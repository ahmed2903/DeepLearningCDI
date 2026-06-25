# ###########################################
# Filename: predict.py
# Prediction and support constraint classes for CNN phase retrieval.
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

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.cuda.amp import GradScaler

from scipy.ndimage.measurements import center_of_mass as com
from scipy.ndimage.interpolation import shift

from train import CNNTrain


class ShrinkWrap():
	"""
	Performs the shrinkwrap method for a given threshold and sigma values
	Only works for the prediction model, NOT the training
	"""
	def __init__(self):
		super().__init__()
		self.sw_threshold = 0.2
		self.sigma = 4
		self.outdata = None
		self.shp = None
		self.kernel_size = 3
		self.kernel = None
		self.padding = 1
		self.support = None


	def SetSWData(self, outdata):
		self.outdata = outdata[0,0,:,:,:]

	def SetSWSigma(self,sigma):
		self.sw_sigma = sigma

	def SetSWThreshold(self,thresh):
		self.sw_threshold = thresh

	def SetSWCyclelength(self,cycle):
		self.sw_cycle = cycle

	def GaussianFill(self):

		# Define Gaussian kernel
		self.kernel_size = int(4*self.sw_sigma+1)
		sigma = self.sw_sigma
		x = torch.arange(self.kernel_size, dtype=torch.float32) - self.kernel_size // 2
		self.kernel = torch.exp(-0.5 * (x ** 2) / sigma ** 2)
		self.kernel = self.kernel / torch.sum(self.kernel)
		self.kernel = self.kernel.view(1, 1, self.kernel_size).repeat(1, self.kernel_size, 1).repeat(self.kernel_size, 1, 1)

		self.kernel = self.kernel.unsqueeze(0).unsqueeze(0)

		self.kernel = self.kernel.cuda()

	def SWUpdateSupport(self):
		print("Updating support ...")

		self.support = torch.abs(self.outdata).clone()
		maxvalue = torch.abs(self.support).max()

		self.support = torch.where(self.support<(self.sw_threshold*maxvalue), 0, 1)
		self.support = self.support.float()
		#self.support = self.support.view(1, self.shp[0]//2, self.shp[1]//2, self.shp[2]//2)
		#self.support = torch.nn.functional.conv3d(self.support.unsqueeze(0).cuda(), self.tempdata.unsqueeze(0).cuda())
		#self.support = F.conv3d(self.support.cuda(), self.kernel.unsqueeze(0).unsqueeze(0), padding=self.padding)

		self.support = F.conv3d(self.support.unsqueeze(0).unsqueeze(0), self.kernel, padding=self.kernel_size//2)

		self.support = torch.where(self.support<self.sw_threshold, 0, 1)

		self.support.detach()
		del self.kernel

		print("done.")

class CNNPredict(CNNTrain, ShrinkWrap):
	"""
	Prediction from trained neural network.
	"""
	def __init__(self):
		super().__init__()
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.expdata = None
		self.mask_val = 10
		self.max = 10000
		self.NQ = 5
		self.shp = None
		self.torcharray = None
		self.noisydata = None
		self.trained_network = None
		self.val_amp = None
		self.val_pha = None
		self.multiloader = None
		self.output = None
		self.model = None
		self.batch_size = 1
		self.loss_pcc = []
		self.loss_chi = []
		self.supp_loss = []
		self.pen_phase = []
	def SetExpData(self, fname, mask=100, square_root=True, validation = False):
		"""
		Set diffraction data used for prediction.
		Mask out the noise by selecting a threshold
		value below which everything is set to zero.
		Normalizes the array to [0,1] interval.
		Square root if data is intensity.
		"""
		self.mask_val = mask
		self.expdata = np.abs(np.load(fname))
		self.expdata[self.expdata < self.mask_val] = 0.0

		if square_root:
			self.expdata = np.sqrt(self.expdata)

		self.max = self.expdata.max()
		max = 1.0/self.expdata.max()
		self.expdata = self.expdata * max

		#self.targetdata = np.fft.fftshift(self.expdata)
		self.shp = self.expdata.shape

		self.expdata = torch.from_numpy(self.expdata)
		#self.expdata = self.CenterResize(self.expdata.unsqueeze(0).float().to(self.device))
		self.expdata = self.expdata.unsqueeze(0).float().to(self.device)
		self.expdata = self.expdata.unsqueeze(0)

		max_idx_flat = torch.argmax(self.expdata)
		max_idx = torch.tensor(np.unravel_index(max_idx_flat.cpu(), self.expdata.shape)).to(self.expdata.device)
		print(max_idx)

		print(self.expdata.dtype)

	def LoadValidationFile(self, fname):
		realspacearr = np.load(fname)

		amp_val = np.abs(realspacearr)
		pha_val = np.angle(realspacearr)

		self.val_amp = torch.from_numpy(amp_val).to(self.device)
		self.val_pha = torch.from_numpy(pha_val).to(self.device)

		del amp_val, pha_val, realspacearr

	def SetTrainedNN(self, fname):
		"""
		Load the trained neural network.
		Must be a .pth file.
		"""
		self.trained_network = fname

		if self.device.type == 'cuda':
			self.model.load_state_dict(torch.load(self.trained_network))
			#self.torcharray = self.torcharray.to(device = self.device, dtype = torch.float)
		else:
			self.model.load_state_dict(torch.load(self.trained_network, map_location = 'cpu'))
	def SetOutputFile(self, fname):
		"""
		Set output file, i.e. the
		reconstructed object.
		"""
		self.output = fname.split(".npy")[0]+"_"+self.datestr+'.npy'
	def DoShrinkWrap(self, ampdata, sigma = 4.5, thresh = 0.05, cycleL = 100):
		self.SetSWData(ampdata)
		self.GaussianFill()
		self.SWUpdateSupport()

	def SetSupport(self, fname):

		self.support = torch.abs(torch.from_numpy(np.load(fname)))
		self.support = self.support.to(self.device)
		self.support = self.support.unsqueeze(0)

	def CenterResize(self, in_tensor):

		max_idx_flat = torch.argmax(in_tensor)
		max_idx = torch.tensor(np.unravel_index(max_idx_flat.cpu(), in_tensor.shape)).to(in_tensor.device)
		print(max_idx)
		shp = torch.tensor(in_tensor.shape, device=in_tensor.device)

		arraycentered = torch.zeros_like(in_tensor)

		idx_ns = (shp // 2) - max_idx
		idx_ne = shp + (shp // 2) - max_idx


		idx_s = torch.tensor([0, 0, 0, 0], device=in_tensor.device)

		idx_e = shp.clone().detach()

		mask = idx_ns < 0
		idx_ns[mask] = 0
		idx_s[mask] = max_idx[mask] - (shp[mask] // 2)

		mask = idx_ne > shp
		idx_ne[mask] = shp[mask]
		idx_e[mask] = max_idx[mask] + (shp[mask] // 2)

		arraycentered[idx_ns[0]:idx_ne[0],idx_ns[1]:idx_ne[1],idx_ns[2]:idx_ne[2], idx_ns[3]:idx_ne[3]] = in_tensor[idx_s[0]:idx_e[0],idx_s[1]:idx_e[1],idx_s[2]:idx_e[2],idx_s[3]:idx_e[3]]

		max_idx_flat = torch.argmax(arraycentered)
		max_idx = torch.tensor(np.unravel_index(max_idx_flat.cpu(), arraycentered.shape)).to(arraycentered.device)
		print(max_idx)

		return arraycentered

	def LossSuppPenaltyOsc(self, output, target):

		"""
		Takes in:
			The output of the neural network
			Input diffraction Pattern

		Computes:
			Loss between Fourier transform of the output and input
			Penalty term on the values outside the support

		Returns:
			Scalar loss value

		"""

		X,Y,Z = self.shp
		X4 = X//4
		Y4 = Y//4
		Z4 = Z//4
		X2 = X//2
		Y2 = Y//2
		Z2 = Z//2


		# Compute Penalty Term For Values Outside Support
		unmasked_amp = torch.ones_like(self.support)
		unmasked_amp[self.support > 0.5] = 0
		unmasked_amp = output[:,0,:,:,:] * unmasked_amp
		unmasked_amp = torch.sum(unmasked_amp)
		total_amp = torch.sum(output[:,0,:,:,:])
		amp_penalty = (unmasked_amp/total_amp) + 1e-4
		del unmasked_amp
		output[:,0,:,:,:] = output[:,0,:,:,:] * self.support

		# Perform Fourier Transform on Output
		pad = nn.ConstantPad3d((Z4,Z4,Y4,Y4,X4,X4), 0)
		obj_comp = pad(output[:,0,:,:,:]) * torch.cos(pad(output[:,1,:,:,:])) + 1j * pad(output[:,0,:,:,:]) * torch.sin(pad(output[:,1,:,:,:]))
		amp_out = torch.fft.fftshift(torch.fft.ifftn(obj_comp, dim= (-3,-2,-1)), dim= (-3,-2,-1))
		amp_out = torch.abs(amp_out[:,:,:,:])

		max_idx_flat = torch.argmax(amp_out)
		max_idx = torch.tensor(np.unravel_index(max_idx_flat.cpu(), amp_out.shape)).to(amp_out.device)
		print(max_idx)

		#amp_out = self.CenterResize(amp_out)

		"""
		amp_out = torch.flip(amp_out, dims=(-3, -2, -1))
		max_idx_flat = torch.argmax(amp_out)
		max_idx = torch.tensor(np.unravel_index(max_idx_flat.cpu(), amp_out.shape)).to(amp_out.device)
		print(max_idx)
		"""

		amp_out = amp_out/(torch.max(amp_out)+1e-8)


		# Fourier Space Loss
		pcc_loss = self.pcc_loss(amp_out, target)
		chi_loss = self.chi_loss(amp_out, target)

		epoch = len(self.train_loss)

		pcc_period = 110
		supp_period = 10
		osc_period = pcc_period + supp_period
		osc_phase = epoch % osc_period

		if osc_phase<pcc_period:
			alpha = 2.0
			gamma = 1.0
			beta = 2.0
		else:
			alpha = 2.0
			gamma = 1.0
			beta = 2.0


		tot_loss = (alpha*pcc_loss + beta*chi_loss + gamma*amp_penalty )/(alpha+beta+gamma)

		del obj_comp
		del amp_out

		return tot_loss, pcc_loss, chi_loss, amp_penalty


	def criterion(self,  output, target):
		"""
		Call the desired loss function.
		"""
		#return self.PredLoss(output, input)
		return self.LossSuppPenaltyOsc( output = output, target = target)
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

		i,j,k = self.shp
		amp = np.zeros((i//2,j//2,k//2), dtype=np.double)
		pha = np.zeros((i//2,j//2,k//2), dtype=np.double)

		amp[:] = sequence[0,0,:,:,:]
		pha[:] = sequence[0,1,:,:,:] * 2.0 * np.pi

		com = amp * np.cos(pha) + 1j * amp * np.sin(pha)

		np.save(self.output, com)

	def ShiftCentreOfMass(self, amp, pha):
		coms = com(amp)
		deltas = (int(round(self.shp[0]//4 - coms[0])), int(round(self.shp[1]//4 - coms[1])), int(round(self.shp[2]//4 - coms[2])))
		amp_shift = shift(amp, shift= deltas)
		pha_shift = shift(pha, shift= deltas)

		return amp_shift, pha_shift

	def MultiPredict(self, AMP = False):
		if AMP:
			scaler = GradScaler(init_scale=1.0, growth_factor=2.0, backoff_factor=0.5, growth_interval=100, enabled=True)
		for epoch in range(self.hyperpars['epochs']):  # loop over the dataset multiple times
			train_loss_tmp = 0.0
			self.model.train()
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
			for idi in range(sw_op):
				if sw_op_flag == idi:
					list(self.optimisers.values())[idi].zero_grad()

			for ii in enumerate(self.multiloader):

				diff_i = self.multiloader

				if AMP:
					with torch.cuda.amp.autocast():
						# Forward propagation
						y_predict = self.model.forward(diff_i)
						# Define the loss and then backward propagate
						loss1 = self.criterion(y_predict, diff_i)
					#
					scaler.scale(loss1).backward()
					for idi in range(sw_op):
						if sw_op_flag == idi:
							scaler.unscale_(list(self.optimisers.values())[idi])
				else:
					# Forward propagation
					y_predict = self.model.forward(diff_i)
					# Define the loss and then backward propagate
					loss1 = self.criterion(y_predict, diff_i)
					loss1.backward()
				#incorporate a clip on the values of the gradients, to avoid exploding gradients
				clip_grad_norm_(self.model.parameters(), max_norm = 10.0, norm_type=2)
				# Optimise the weights and biases
				for idi in range(sw_op):
					if sw_op_flag == idi:
						if AMP:
							scaler.step(list(self.optimisers.values())[idi])
						else:
							list(self.optimisers.values())[idi].step()
				if AMP:
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
					print('Epoch: %d, Epoch-loss: train %.5f lr %.2e' %(epoch + 1, self.train_loss[-1], lr))
				# Save
			if epoch == (self.hyperpars['epochs'] -1):
				self.model.eval()
				with torch.no_grad():
					self.model.DisableCheckpoints()
					sequence = self.model.forward(diff_i)

				sequence = sequence.cpu()

				os.mkdir(self.datestr)

				for o in range(self.NQ):
					i,j,k = self.shp
					amp = np.zeros((i//2,j//2,k//2), dtype=np.double)
					pha = np.zeros((i//2,j//2,k//2), dtype=np.double)

					amp[:] = sequence[o,0,:,:,:]
					pha[:] = sequence[o,1,:,:,:]

					amp, pha = self.ShiftCentreOfMass(amp, pha)

					comp = amp * np.cos(pha) + 1j * amp * np.sin(pha)

					np.save(self.datestr+'/'+str(o)+self.output, comp)

	def TransferPredict(self, AMP=False, validation = False):
		torch.autograd.set_detect_anomaly(True)
		scaler = GradScaler(init_scale=1.0, growth_factor=2.0, backoff_factor=0.5, growth_interval=100, enabled=True)
		for epoch in range(self.hyperpars['epochs']):  # loop over the dataset multiple times
			train_loss_tmp = 0.0
			amp_loss_tmp = 0.0
			pha_loss_tmp = 0.0
			pcc_loss_tmp = 0.0
			chi_loss_tmp = 0.0
			self.model.train()
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

			for idi in range(sw_op):
				if sw_op_flag == idi:
					list(self.optimisers.values())[idi].zero_grad()


			if AMP:
				with torch.cuda.amp.autocast():
					if validation:
						# Forward propagation
						y_predict = self.model(self.torcharray)
						# Define the loss and then backward propagate
						losses = self.criterion(y_predict, self.targetdata)
						loss1 = losses[0]
						loss_amp = losses[1]
						loss_pha = losses[2]
					else:
						y_predict = self.model(self.expdata)
						loss1 = self.criterion(y_predict, self.targetdata)
				#
				scaler.scale(loss1).backward()
				for idi in range(sw_op):
					if sw_op_flag == idi:
						scaler.unscale_(list(self.optimisers.values())[idi])
			else:
				if validation:
					# Forward propagation
					y_predict = self.model(self.torcharray)
					# Define the loss and then backward propagate
					losses = self.criterion(y_predict, self.targetdata)
					loss1 = losses[0]
					loss_amp = losses[1]
					loss_pha = losses[2]
				else:
					y_predict = self.model(self.expdata)
					#y_predict = torch.where(self.support>0.5, y_predict, 0)
					loss1, losses_pcc, losses_chi, losses_supp = self.criterion( y_predict, self.expdata)

				loss1.backward(retain_graph=False)
			#incorporate a clip on the values of the gradients, to avoid exploding gradients
			clip_grad_norm_(self.model.parameters(), max_norm = 10.0, norm_type=2)
			# Optimise the weights and biases
			for idi in range(sw_op):
				if sw_op_flag == idi:
					if AMP:
						scaler.step(list(self.optimisers.values())[idi])
					else:
						list(self.optimisers.values())[idi].step()
			if AMP:
				scaler.update()
			# Sum losses
			train_loss_tmp += loss1.item()
			amp_loss_tmp += losses_supp.item()
			pcc_loss_tmp += losses_pcc.item()
			chi_loss_tmp += losses_chi.item()

			if validation:
				amp_loss_tmp += loss_amp.item()
				pha_loss_tmp += loss_pha.item()
			# Update learning rate with scheduler
			for idi in range(sw_sch):
				if sw_op_flag == idi:
					list(self.schedulers.values())[idi].step()
					lr = self.GetLR(list(self.optimisers.values())[idi])
			# Update graph data
			self.train_loss.append(train_loss_tmp)
			self.loss_pcc.append(pcc_loss_tmp)
			self.loss_chi.append(chi_loss_tmp)
			self.supp_loss.append(amp_loss_tmp)

			if validation:
				self.amp_loss_val.append(amp_loss_tmp)
				self.pha_loss_val.append(pha_loss_tmp)

			# Check for instability
			if np.isfinite(self.train_loss[-1]):
				pass
			elif self.verbose:
				print('Model is unstable.')
				break
			# Print some useful information
			if self.verbose:
				print('Epoch: %d, Epoch-loss: train %.5f lr %.2e' %(epoch + 1, self.train_loss[-1], lr))

			# # Update Support
			# if epoch>0 and epoch%self.sw_cycle == 0 :
			# 	self.DoShrinkWrap(y_predict)
			# 	self.support = self.support.detach()
			# Save
			if epoch == (self.hyperpars['epochs'] -1):
				print("... Saving Output")
				self.model.eval()
				with torch.no_grad():
					self.model.DisableCheckpoints()
					sequence = self.model(self.expdata)

				sequence = sequence.cpu()

				i,j,k = self.shp
				amp = np.zeros((i//2,j//2,k//2), dtype=np.double)
				pha = np.zeros((i//2,j//2,k//2), dtype=np.double)

				amp[:] = sequence[0,0,:,:,:]
				pha[:] = sequence[0,1,:,:,:] #* torch.pi
				#pha = np.arctan2(np.sin(pha), np.cos(pha))

				com = amp * np.cos(pha) + 1j * amp * np.sin(pha)

				os.mkdir(self.datestr)
				np.save(self.datestr+'/'+self.output, com)
				#torch.save(self.model.state_dict(),self.datestr+'/'+'CP{}'.format(epoch+1)+'_'+self.datestr+'.pth')
				print("Saving Model...")
				print("Done")



	def SaveTrainLoss(self):
		lossdata = np.array(self.train_loss)
		np.save(self.datestr+'/'+'trainlossdata_'+self.datestr+'.npy', lossdata)


	def PlotLoss(self, validation = False):

		if validation:
			fig1 = plt.figure()
			plt.plot(self.train_loss, label = 'Prediction Loss')
			plt.yscale('log')
			plt.legend(frameon=False)
			plt.savefig(self.datestr+'/'+'Prediction_error_'+self.datestr+'.png')

			fig2 = plt.figure()
			plt.plot(self.amp_loss_val, label = 'Amplitude Validation Loss')
			plt.plot(self.pha_loss_val, label = 'Phase Validation Loss')
			plt.yscale('log')
			plt.legend(frameon=False)
			plt.savefig(self.datestr+'/'+'Validation_error_'+self.datestr+'.png')

		else:
			fig1 = plt.figure()
			plt.plot(self.train_loss, label='Prediction loss')
			plt.yscale('log')
			plt.legend(frameon=False)
			plt.savefig(self.datestr+'/'+'Prediction_error_'+self.datestr+'.png')

			fig2 = plt.figure()
			plt.plot(self.loss_pcc, label = 'PCC Loss')
			plt.yscale('log')
			plt.legend(frameon=False)
			plt.savefig(self.datestr+'/'+'PCCLoss_'+self.datestr+'.png')

			fig3 = plt.figure()
			plt.plot(self.supp_loss, label = 'Support Penalty')
			plt.yscale('log')
			plt.legend(frameon=False)
			plt.savefig(self.datestr+'/'+'SuppPen_'+self.datestr+'.png')

			fig4 = plt.figure()
			plt.plot(self.loss_chi, label = 'Chi Loss')
			plt.yscale('log')
			plt.legend(frameon=False)
			plt.savefig(self.datestr+'/'+'ChiLoss_'+self.datestr+'.png')
