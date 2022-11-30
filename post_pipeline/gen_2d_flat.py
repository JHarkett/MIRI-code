#===========================================================
"""
Generates a striped flat field within a new directory:
flat_files

Usage
		python gen_2d_flat.oy

"""
#===========================================================
# Inputs


file = 'Level3_ch2-medium_s3d.fits'
tile_dirs = ['july/centre', 'july/east', 'aug/centre', 'aug/west']


#===========================================================
# Imports


import numpy as np
from astropy.io import fits
import glob
import os


#===========================================================
# Functions


def NormalizeData(data):
	return np.divide(data, np.nanmedian(data))


#===========================================================
# Main


if not os.path.exists('flat_files'):
	os.mkdir('flat_files')


band_pre = file.replace('Level3_','')
band = band_pre.replace('_s3d.fits','')

print(band)

if not os.path.exists('flat_files/' + band):
	os.mkdir('flat_files/' + band)

length_t = len(tile_dirs)

dithers = []

for kk in range(length_t):
	for ii in range(10):
		if os.path.exists(tile_dirs[kk] + '/stage3/d{}'.format(ii+1)):
			dithers.append(tile_dirs[kk] + '/stage3/d{}'.format(ii+1))

length_d = len(dithers)

print('{} directories detected\n'.format(length_d))

spec_norm_final = []

for kk in range(length_d):
	if os.path.exists(dithers[kk] + '/' + file):
		hdul = fits.open(dithers[kk] + '/' + file)
		spec_data = hdul['SCI'].data
		hdul.close()

		dz,dy,dx = spec_data.shape

		spec_norm = [None]*dz
		spec_norm_ave = np.zeros((dz,dy,1))

		for k in range(dz):
			for j in range(dy):
				for i in range(dx):
					if spec_data[k][j][i] == 0:
						spec_data[k][j][i] = np.nan
			spec_norm[k] = NormalizeData(spec_data[k])

		for k in range(dz):
			for j in range(dy):
				spec_norm_ave[k][j] = np.nanmedian(spec_norm[k][j])

		spec_norm_final.append(spec_norm_ave)

print('{} files used for flat-file generation\n'.format(len(spec_norm_final)))

flat_result = np.nanmedian(spec_norm_final, axis=0)

flat_result2 = np.zeros((dz,dy,dx))

for k in range(dz):
	for j in range(dy):
		for i in range(dx):
			flat_result2[k][j][i] = flat_result[k][j][0]
			if np.isnan(flat_result2[k][j][i]) == True:
				flat_result2[k][j][i] = 1

hdu2 = fits.PrimaryHDU(flat_result2)
hdu2.writeto('flat_files/' + band + '/flat_all_epoch.fits',overwrite=True)


#===========================================================


print('End of script\n')
