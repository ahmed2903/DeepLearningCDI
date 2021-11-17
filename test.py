# import torch

# x = torch.rand(10,10, dtype=torch.complex64)
# fft2 = torch.fft.fft2(x)

# print(x.shape)
# print(fft2.shape)

import numpy 
k = numpy.load('reci_intes.npy')
r = numpy.load('real_obj.npy')

print(k.shape, r.shape)
