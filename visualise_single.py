#========================================================================
"""
Generates an RGB image of a single fits cube based on user-inputted
wavelengths for Red, Green and Blue

Usage:
		python visualise_single.py

"""
#========================================================================
# Inputs


name = 'c1_centre'
file = 'stage3/d1/Level3_ch1-short_s3d.fits'
R_wave = 5.05352
G_wave = 5.20751
B_wave = 5.3175


#========================================================================
# Imports


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider,Button
import os
import glob
from astropy.io import fits
import scipy
from PIL import Image


#========================================================================
# Functions


def find_nearest(array,value):
	array = np.asarray(array)
	idx = (np.abs(array - value)).argmin()
	return idx


def NormalizeData(data):
	return (data - (np.nanmin(data)-1)) / ((np.nanmax(data)+1) - (np.nanmin(data)-1))


#========================================================================
# Main


if not os.path.exists('rgb_images'):
	os.mkdir('rgb_images')


hdul = fits.open(file)
spec_data = hdul['SCI'].data
hdr = hdul['SCI'].header
hdul.close()

dz,dy,dx = spec_data.shape

wave = np.arange(hdr['NAXIS3'])*hdr['CDELT3']+hdr['CRVAL3']

for k in range(dz):
	for j in range(dy):
		for i in range(dx):
			if spec_data[k][j][i] == 0:
				spec_data[k][j][i] = np.nan


R_idx = find_nearest(wave,R_wave)
G_idx = find_nearest(wave,G_wave)
B_idx = find_nearest(wave,B_wave)

R = [None]*len(spec_data)
for i in range(len(spec_data)):
	R[i] = spec_data[i][15][15]

plt.plot(wave,R,'k-',linewidth=0.5)
plt.scatter(wave[R_idx],R[R_idx],linewidth=0.3,s=20,c='r')
plt.scatter(wave[G_idx],R[G_idx],linewidth=0.3,s=20,c='g')
plt.scatter(wave[B_idx],R[B_idx],linewidth=0.3,s=20,c='b')
plt.xlabel('Wavelength ($\mu m$)')
plt.ylabel('Surface Brightness (MJy sr$^{-1}$)')
plt.tight_layout()
plt.grid()
plt.savefig('rgb_images/' + name + '_spectrum.png')
plt.show()


R_frame = spec_data[R_idx]
G_frame = spec_data[G_idx]
B_frame = spec_data[B_idx]


R_frame = NormalizeData(R_frame)
G_frame = NormalizeData(G_frame)
B_frame = NormalizeData(B_frame)


for j in range(dy):
	for i in range(dx):
		if np.isnan(R_frame[j][i]) == True:
			G_frame[j][i] = np.nan
			B_frame[j][i] = np.nan
		if np.isnan(G_frame[j][i]) == True:
			R_frame[j][i] = np.nan
			B_frame[j][i] = np.nan
		if np.isnan(B_frame[j][i]) == True:
			R_frame[j][i] = np.nan
			G_frame[j][i] = np.nan 


rgbArray = np.zeros((dy,dx,3), 'uint8')
rgbArray[..., 0] = R_frame*256
rgbArray[..., 1] = G_frame*256
rgbArray[..., 2] = B_frame*256


img = Image.fromarray(rgbArray)


plt.imshow(img,origin='lower')
plt.axis('off')
plt.savefig('rgb_images/' + name + '.png')
plt.show()


#========================================================================


print('End of script\n')
