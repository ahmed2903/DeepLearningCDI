# ###########################################
# Filename: model.py
# Neural network architecture for CNN phase retrieval.
# Derived from work by Longlong Wu.
#
# Authors: Ahmed H. Mokhtar, Marcus Newton
#
# Version 0.14
# Licence: GNU GPL 3
#
# ###########################################

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint_sequential


class double_conv(nn.Module):
	"""
	Main convlutional layer
	Sequentially applying a 3D convolution followed by a batch normalization and a LeakyRelu activation function
	This is done twice however, the second second convolution is divided into three parts

	This concludes one layer in the encoder

	Values that can be tuned are: momentum and Grad of the leaky relu
	"""
	def __init__(self, in_ch, out_ch, LRLUGrad=0.2, eps=1e-8, momentum=0.7, pad = 1, dilation = 1):
		super(double_conv, self).__init__()
		self.conv = nn.Sequential(
			nn.Conv3d(in_ch, out_ch, kernel_size=(3, 3, 3), stride=1, padding=(pad, pad, pad), bias=True, dilation = dilation),
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


class double_conv_tr(nn.Module):
	"""
	Main convlutional layer
	Sequentially applying a 3D convolution followed by a batch normalization and a LeakyRelu activation function
	This is done twice however, the second second convolution is divided into three parts

	This concludes one layer in the encoder

	Values that can be tuned are: momentum and Grad of the leaky relu
	"""
	def __init__(self, in_ch, out_ch, LRLUGrad, eps=1e-8, momentum=0.9, pad = 1, dilation = 1):
		super(double_conv_tr, self).__init__()
		self.conv = nn.Sequential(
			nn.ConvTranspose3d(in_ch, out_ch, kernel_size=(3, 3, 3), stride=1, padding=(pad, pad, pad), bias=False, dilation = dilation),
			nn.BatchNorm3d(num_features=out_ch, eps=eps, momentum=momentum, affine=True),
			nn.LeakyReLU(LRLUGrad, inplace=True),
			nn.ConvTranspose3d(out_ch, out_ch, kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0), bias=True),
			nn.ConvTranspose3d(out_ch, out_ch, kernel_size=(1, 3, 1), stride=1, padding=(0, 1, 0), bias=True),
			nn.ConvTranspose3d(out_ch, out_ch, kernel_size=(1, 1, 3), stride=1, padding=(0, 0, 1), bias=True),
			nn.BatchNorm3d(num_features=out_ch, eps=eps, momentum=momentum, affine=True),
			nn.LeakyReLU(LRLUGrad, inplace=True),
		)
	def forward(self, x):
		x = self.conv(x)
		return x

class inconv(nn.Module):
	"""
	Same as the previous convolutional layer, however, the second convolution is summarized in one operation opposed to three
	"""
	def __init__(self, in_ch, out_ch, LRLUGrad=0.2, eps=1e-8, momentum=0.7):
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
	def __init__(self, in_ch, out_ch, LRLUGrad=0.2, eps=1e-8, momentum=0.9, checkpoints=False, pad = 1, dilation = 1):
		super(down, self).__init__()
		self.checkpoints = checkpoints
		self.mpconv = nn.Sequential(
			nn.MaxPool3d(kernel_size=(2, 2, 2)),
			double_conv(in_ch, out_ch, LRLUGrad, eps, momentum, pad, dilation),
		)
	def forward(self, x):
		if self.checkpoints is True:
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
	def __init__(self, in_ch, out_ch, LRLUGrad=0.2, eps=1e-8, momentum=0.9, checkpoints=False, pad = 1, dilation = 1):
		super(up01, self).__init__()
		self.checkpoints = checkpoints
		self.upconv = nn.Sequential(
			nn.Upsample(scale_factor=2, mode='nearest'),
			double_conv(in_ch, out_ch, LRLUGrad, eps, momentum, pad, dilation),
		)
	def forward(self, x):
		if self.checkpoints is True:
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
	def __init__(self, in_ch, out_ch, LRLUGrad, eps=1e-8, momentum=0.9, checkpoints=False, pad =1, dilation = 1, drop_out = 0.0):
		super(up02, self).__init__()
		self.checkpoints = checkpoints
		self.upconv = nn.Sequential(
			nn.Upsample(scale_factor=2, mode='nearest'),
			double_conv(in_ch, out_ch, LRLUGrad, eps, momentum, pad, dilation),
			nn.Dropout3d(drop_out),
		)
	def forward(self, x):
		if self.checkpoints is True:
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
		self.conv = nn.Conv3d(in_ch, out_ch, kernel_size=(1, 1, 1), stride=1, padding=(0, 0, 0), bias=True, dilation = 1)
	def forward(self, x):
		x = self.conv(x)
		return x

class outconv_tr(nn.Module):
	'''
	Convolutional layer for the final layer of the network
	only includes one conv3D operation, no maxpooling or leakyrelu
	'''
	def __init__(self, in_ch, out_ch):
		super(outconv_tr, self).__init__()
		self.conv = nn.ConvTranspose3d(in_ch, out_ch, kernel_size=(1, 1, 1), stride=1, padding=(0, 0, 0), bias=True)
	def forward(self, x):
		x = self.conv(x)
		return x


class NNModel(nn.Module):
	'''
	summing up all the operations to create the full network
	'''
	def __init__(self, n_channels=1, n_classes=1, LRLU_val=0.15, dropout= 0.0, checkpoints=False):
		super(NNModel, self).__init__()
		self.inconv = inconv(n_channels, 64)
		self.down1 = down(64, 128, checkpoints=checkpoints)
		self.down2 = down(128, 256,checkpoints=checkpoints)
		self.down3 = down(256, 512, checkpoints=checkpoints)
		self.down4 = down(512, 1024, checkpoints=checkpoints)

		self.up01 = up01(512, 256, checkpoints=checkpoints)
		self.up02 = up01(256, 128, checkpoints=checkpoints)
		self.up03 = up01(128, 64, checkpoints=checkpoints)
		self.outc00 = outconv(64, n_classes)

		self.up11 = up02(512, 256, checkpoints=checkpoints,LRLUGrad=0.2 ,drop_out= dropout)
		self.up12 = up02(256, 128, checkpoints=checkpoints,LRLUGrad=0.2 ,drop_out= dropout)
		self.up13 = up02(128, 64, checkpoints=checkpoints,LRLUGrad=0.2 ,drop_out= dropout)
		self.outc11 = outconv(64, n_classes)

		self.final_relu = nn.LeakyReLU(LRLU_val)

		self.tanh = nn.Tanh()

	def DisableCheckpoints(self):
		self.down1.checkpoints = False
		self.down2.checkpoints = False
		self.down3.checkpoints = False
		self.down4.checkpoints = False
		self.up01.checkpoints = False
		self.up02.checkpoints = False
		self.up03.checkpoints = False
		self.up11.checkpoints = False
		self.up12.checkpoints = False
		self.up13.checkpoints = False

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
		#x2 = torch.relu(x2) #activation function in the final layer is a relu opposed to a leakReLU
		#x2 = self.final_relu(x2)
		#x2 = self.tanh(x2) * torch.pi
		x0 = torch.cat((x1, x2), 1) # comnbining the two branches together

		return x0
