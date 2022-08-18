# ###########################################
# Filename: gendata.py
# Generate Training Data for CNN Phase Retrieval.
# Derived from work by Longlong Wu.
# 
# Author: Marcus Newton
# 
# Version 0.3
# Licence: GNU GPL 3
#
# ###########################################

from math import sin, cos, pi, acos, sqrt
import numpy as np
import scipy.signal as sps

import threading
from os import sched_getaffinity

class GenData():
	def __init__(self):
		self.sqrtof3 = sqrt(3.0)
		self.rs_shp = [32,32,32]
		self.n = 1
		self.rsmask = 0.025
		self.fsmask = 0.01
		self.rs_obj = None
		self.fs_amp = None
		self.morph_fn = self.MakeOctahedron
		self.nthreads = len(sched_getaffinity(0))
	def SetShape(self, shape=[32,32,32]):
		self.rs_shp = shape
	def SetN(self, n = 1):
		self.n = n
	def SetRSMask(self, rsmask = 0.025):
		self.rsmask = rsmask
	def SetFSMask(self, fsmask = 0.01):
		self.fsmask = fsmask
	def SetMorphology(self, shape="octahedron"):
		if shape == "octahedron":
			self.morph_fn = self.MakeOctahedron
		elif shape == "hexprism":
			self.morph_fn = self.MakePrism
		elif shape == "monoclinic":
			self.morph_fn = self.MakeMonoclinic
		else:
			pass
	def gaussian(self, x, mu, sig):
		return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))
	def GaussKernel(self):
		c_kx, c_ky, c_kz = np.mgrid[-1:1:3j, -1:1:3j, -1:1:3j]
		cov_k = self.gaussian(c_kx, 0, 1) * self.gaussian(c_ky, 0, 1) * self.gaussian(c_kz, 0, 1)
		return cov_k
	def MakeMesh(self, shp):
		X,Y,Z = shp
		mesh = np.meshgrid(np.linspace(-X / 2, X / 2, X, endpoint=True),
				np.linspace(-Y / 2, Y / 2, Y, endpoint=True),
				np.linspace(-Z / 2, Z/ 2, Z, endpoint=True), indexing='ij')
		mesh = np.array(mesh).view()
		mesh.shape = (3, shp[0]*shp[1]*shp[2])
		return mesh
	def SuperVolumeOctahedron(self, x, y, z, a, b, c, mm, nn):
		Epl1 = (mm - 1) * 1 / 4 + 0.15
		Epl2 = (nn - 1) * 1 / 4 + 0.15
		rr = ((np.abs(x / a)) ** (2 / Epl1) + (np.abs(y / b)) ** (2 / Epl1)) ** (Epl1 / Epl2) + (np.abs(z / c)) ** (2 / Epl2)
		return rr
	def MakeOctahedron(self, x, y, z):
		s1,s2,s3 = self.rs_shp[0]//8,self.rs_shp[1]//8,self.rs_shp[2]//8
		# Random shape
		s = 9
		m1 = 1 + (s * np.random.rand(1)).astype(int)
		n1 = 1 + (s * np.random.rand(1)).astype(int)
		# Random size
		a1 = s1 + s1 * np.random.rand(1)
		b1 = s2 + s2 * np.random.rand(1)
		c1 = s3 + s3 * np.random.rand(1)
		# Make shape
		r0 = self.SuperVolumeOctahedron(x, y, z, a1, b1, c1, m1, n1)
		rs_amp = np.zeros((self.rs_shp[0], self.rs_shp[1], self.rs_shp[2]))
		rs_amp[r0 < 1.0] = 1.0
		return rs_amp
	def SuperVolumePrism(self, x, y, z, a, b, c, mm, nn):
		Epl1 = (mm - 1.0)  + 1e-2
		Epl2 = (nn - 1.0) + 1e-2
		rr = ((np.abs((self.sqrtof3*x + y)/ a)) ** (2 / Epl1) + (np.abs((self.sqrtof3*x - y)/ a)) ** (2 / Epl1) + (np.abs( 2*y / b)) ** (2 / Epl1)) ** (Epl1 / 2) + (np.abs(z / c)) ** (2 / Epl2)
		return rr
	def MakePrism(self, x, y, z):
		s1,s2,s3 = self.rs_shp[0]//8,self.rs_shp[1]//8,self.rs_shp[2]//8
		# Random shape
		s = 0.25
		m1 = 1.0 + s * np.random.rand(1)
		n1 = 1.0 + s * np.random.rand(1)
		# Random size
		a1 = 2.0*s1 + 4.0*s1 * np.random.rand(1)
		b1 = a1 - 0.25*a1 + 0.5*a1*np.random.rand(1)
		#b1 = 0.5*s2 + 1.0*s2 * np.random.rand(1)
		c1 = 0.5*s3 + 2.0*s3 * np.random.rand(1)
		# Make shape
		r0 = self.SuperVolumePrism(x, y, z, a1, b1, c1, m1, n1)
		rs_amp = np.zeros((self.rs_shp[0], self.rs_shp[1], self.rs_shp[2]))
		rs_amp[r0 < 1.0] = 1.0
		return rs_amp
	def SuperVolumeMonoclinic(self, x, y, z, a, b, c, mm, nn, beta=122.6):
		alpha = beta*pi/180.0 - pi/2.0
		Epl1 = (mm - 1.0)  + 1e-2
		Epl2 = (nn - 1.0) + 1e-2
		rr = ((np.abs((x*cos(alpha) - y*sin(alpha))/ a)) ** (2 / Epl1) + (np.abs( y / b)) ** (2 / Epl1)) ** (Epl1 / 2) + (np.abs(z / c)) ** (2 / Epl2)
		return rr
	def MakeMonoclinic(self, x, y, z, beta=122.6):
		s1,s2,s3 = self.rs_shp[0]//8,self.rs_shp[1]//8,self.rs_shp[2]//8
		# Random shape
		s = 0.25
		m1 = 1.0 + s * np.random.rand(1)
		n1 = 1.0 + s * np.random.rand(1)
		# Random size
		a1 = 0.5*s1 + 1.0*s1 * np.random.rand(1)
		b1 = a1 - 0.25*a1 + 0.5*a1*np.random.rand(1)
		#b1 = 0.5*s2 + 1.0*s2 * np.random.rand(1)
		c1 = 0.5*s3 + 2.0*s3 * np.random.rand(1)
		# Make shape
		r0 = self.SuperVolumeMonoclinic(x, y, z, a1, b1, c1, m1, n1, beta=beta)
		rs_amp = np.zeros((self.rs_shp[0], self.rs_shp[1], self.rs_shp[2]))
		rs_amp[r0 < 1.0] = 1.0
		return rs_amp
	def RSPhase(self, N, rL, h, clx, cly, clz):
		# uncorrelated Gaussian random rough surface distribution
		np.random.seed()
		ZZ = h * np.random.randn(N[0], N[1], N[2])
		x, y, z = np.linspace(-rL[0] / 2, rL[0] / 2, num=N[0], endpoint=True), \
					np.linspace(-rL[1] / 2, rL[1] / 2, num=N[1], endpoint=True), \
					np.linspace(-rL[2] / 2, rL[2] / 2, num=N[2], endpoint=True)
		# #
		[X, Y, Z] = np.meshgrid(x, y, z, indexing='ij')
		# Gaussian filter
		F = np.exp(-X ** 2 / ((clx ** 2) / 2) - Y ** 2 / ((cly ** 2) / 2) - Z ** 2 / ((clz ** 2) / 2))
		# correlated surface generation including convolution (faltning) and inverse
		# Fourier transform and normalizing prefactors
		#f = 2 / np.sqrt(np.pi) * (np.mean(rL) / np.mean(N) / np.sqrt(clx * cly * clz)) * np.fft.ifftn(np.fft.fftn(ZZ) * np.fft.fftn(F))
		f = 2 / np.sqrt(np.pi) * (1.0/np.sqrt(clx * cly * clz)) * np.fft.ifftn(np.fft.fftn(ZZ) * np.fft.fftn(F))
		return f, x, y, z
	def MakePhase(self):
		N=self.rs_shp
		rL=self.rs_shp
		h = 1.0 * np.random.rand(1)
		clx = 6 + 3 * np.random.rand(1)
		cly = 6 + 3 * np.random.rand(1)
		clz = 6 + 3 * np.random.rand(1)
		rs_pha, _, _, _ = self.RSPhase(N, rL, h, clx, cly, clz)
		rs_pha = np.real(rs_pha)
		# phase on [0,1] interval
		rs_pha -= rs_pha.min()
		rs_pha /= rs_pha.max()
		return rs_pha
	def SingleParticle(self, mesh, kernel):
		# Create random rotation array
		u, v = np.random.rand(1), np.random.rand(1)
		theta, phi = 2 * pi * u, acos(2 * v - 1)
		r_tp = np.array([[sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta)],
				[cos(theta) * cos(phi), cos(theta) * sin(phi), -sin(theta)],
				[-sin(phi), cos(phi), 0]], dtype=np.float32)
		# Rotate mesh points
		mesh_rot = np.dot(r_tp,mesh)
		mesh_rot.shape = (3, self.rs_shp[0],self.rs_shp[1],self.rs_shp[2])
		x, y, z = mesh_rot[0], mesh_rot[1], mesh_rot[2]
		# Create super volume
		rs_amp = self.morph_fn(x, y, z)
		# Convolve amp with Gaussian
		rs_amp = np.abs(sps.fftconvolve(rs_amp, kernel, mode='same'))
		rs_amp /= np.max(rs_amp)
		rs_amp[rs_amp < self.rsmask] = 0.0
		# phase
		rs_pha = self.MakePhase()
		rs_pha[rs_amp < self.rsmask] = 0.0
		return rs_amp,rs_pha
	def GenShapeData(self):
		mesh = self.MakeMesh(self.rs_shp)
		kern = self.GaussKernel()
		# #
		rs_amp_phase = np.zeros((self.n, 2, self.rs_shp[-3], self.rs_shp[-2], self.rs_shp[-1]), dtype=np.float32)
		fs_amp = np.zeros((self.n, 1, self.rs_shp[-3] * 2, self.rs_shp[-2] * 2, self.rs_shp[-1] * 2), dtype=np.float32)
		# #
		shp = fs_amp.shape
		# #
		X2 = shp[-3]//2
		X4 = shp[-3]//4
		Y2 = shp[-2]//2
		Y4 = shp[-2]//4
		Z2 = shp[-1]//2
		Z4 = shp[-1]//4
		# #
		def CalcThread(idxrange):
			for i in range(idxrange[0], idxrange[1], 1):
				rs_amp_phase[i,0,:,:,:], rs_amp_phase[i,1,:,:,:] = self.SingleParticle(mesh, kern)
				rs_complex = np.zeros((self.rs_shp[-3] * 2, self.rs_shp[-2] * 2, self.rs_shp[-1] * 2), dtype=np.csingle)
				rs_complex[X2-X4:X2+X4,Y2-Y4:Y2+Y4,Z2-Z4:Z2+Z4] =  rs_amp_phase[i,0,:,:,:] * np.cos(2.0 * np.pi * rs_amp_phase[i,1,:,:,:]) + 1j*rs_amp_phase[i,0,:,:,:] * np.sin(2.0 * np.pi * rs_amp_phase[i,1,:,:,:])
				fs_amp[i,0,:,:,:] = np.abs(np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(rs_complex))))
				fs_amp[i,0,:,:,:] /= np.max(fs_amp[i,0,:,:,:])
				fs_amp[i,0,:,:,:][fs_amp[i,0,:,:,:] < self.fsmask] = 0.0
		# #
		xs = []
		blk = self.n//self.nthreads
		for i in range(self.nthreads):
			xs.append([blk*i, blk*(i+1)])
		xs[-1][1] = self.n
		threads = []
		for t in range(self.nthreads):
			thread = threading.Thread(target=CalcThread, args=(xs[t],))
			thread.start()
			threads.append(thread)
		for thread in threads:
			if thread.is_alive():
				thread.join()
		#
		self.rs_obj = rs_amp_phase
		self.fs_amp = fs_amp
	# #
	def SaveData(self, fsname="fs_amps.npy", rsname="rs_objs.npy"):
		if self.rs_obj is not None:
			np.save(rsname, self.rs_obj)
		if self.fs_amp is not None:
			np.save(fsname, self.fs_amp)

if __name__ == '__main__':
	d = GenData()
	d.SetShape([176,144,196])
	d.SetN(500)
	d.SetMorphology("hexprism")
	#d.SetMorphology("octahedron")
	#d.SetMorphology("monoclinic")
	d.GenShapeData()
	d.SaveData()
	