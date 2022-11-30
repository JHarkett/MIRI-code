#=====================================================================
"""
A combination of build_calibration.py and apply_calibration.py
Will build the wave_cal file and then apply that to the the data

To just do the application step (if the wave_cal file has already
been generated), set build to False

Applies the extracted calibration to each spaxel
Interpolates to ensure all spaxels are on the same scale


Usage
		python apply_calibration.py


"""
#=====================================================================
# Inputs


# Name format of the calibrated data directory
cal_fmt = '_interp_cal'

# To build the wave_cal file, set to true
build = True

# To change the wave_grid of the resulting data cube,
# set to True
# If False, wave_step is ignored
change_wave_grid = False
wave_step = 0.0003

remove_pix = False
x_remove = [3,28]
y_remove = [8,7]


#=====================================================================
# Imports


import numpy as np
from astropy.io import fits
import os
import shutil
import glob
import matplotlib.pyplot as plt
import scipy.interpolate
import numpy.ma as ma
from scipy.optimize import curve_fit
import math
from scipy.signal import savgol_filter
import sys


#=====================================================================
# Functions


def find_nearest(array,value):
	array = np.asarray(array)
	idx = (np.abs(array - value)).argmin()
	return idx


#=====================================================================
# Main


with open('cal_file.txt','r') as f:
	cal_data = f.readlines()
f.close()

for kk in range(len(cal_data)):
	if cal_data[kk][0:28] == 'Uncalibrated data directory:':
		uncal_dir = cal_data[kk+1].replace('\n','')

	if cal_data[kk][0:7] == 'Epochs:':
		epochs = []
		ii = kk+1
		while not cal_data[ii][0:1] == '\n':
			ii = ii+1
			epochs.append(cal_data[ii-1].replace('\n',''))

	if cal_data[kk][0:17] == 'Tile directories:':
		tile_dirs = []
		ii = kk+1
		while not cal_data[ii][0:1] == '\n':
			ii = ii+1
			tile_dirs.append(cal_data[ii-1].replace('\n',''))

	if cal_data[kk][0:5] == 'Band:':
		band = cal_data[kk+1].replace('\n','')

	if cal_data[kk][0:19] == 'Minimum wave value:':
		wave_min_val = float(cal_data[kk+1])

	if cal_data[kk][0:19] == 'Maximum wave value:':
		wave_max_val = float(cal_data[kk+1])

if build == True:
	print('Building cube\n')

	if not os.path.exists('wave_cal_data'):
		os.mkdir('wave_cal_data')

	hdul_map = fits.open('spec_data_' + band + '/retrieval_map.fits')
	map = hdul_map[0].data
	hdul_map.close()

	dy,dx = map.shape

	nem_wave = np.loadtxt('spec_data_' + band + '/nemesis_model.txt')
	dz = len(nem_wave)

	band_corr = np.zeros((dz,dy,dx))

	for j in range(dy):
		for i in range(dx):
			pix = map[j][i]
			ret_data = np.loadtxt('spec_data/{}/correction.txt'.format(pix),usecols = (1))

			for k in range(dz):
				band_corr[k][j][i] = ret_data[k]
	hdu_cal = fits.PrimaryHDU(band_corr)
	hdu_cal.writeto('wave_cal_data/' + band + '_correction.fits')

else:
	hdul_corr = fits.open('wave_cal_data/' + band + '_correction.fits')
	band_corr = hdul_corr[0].data
	hdul_corr.close()

cal_dir = uncal_dir + cal_fmt

if not os.path.exists(cal_dir):
	os.mkdir(cal_dir)

for ii in range(len(epochs)):
	if not os.path.exists(cal_dir + '/' + epochs[ii]):
		os.mkdir(cal_dir + '/' + epochs[ii])

length_t = len(tile_dirs)

file = 'Level3_' + band + '_s3d.fits'
file_nav = 'Level3_' + band + '_s3d_nav.fits'

dz, dy, dx = band_corr.shape

dithers = []

for kk in range(length_t):
	if not os.path.exists(cal_dir + '/' + tile_dirs[kk]):
		os.mkdir(cal_dir + '/' + tile_dirs[kk])

	if not os.path.exists(cal_dir + '/' + tile_dirs[kk] + '/stage3'):
		os.mkdir(cal_dir + '/' + tile_dirs[kk] + '/stage3')

	if not os.path.exists(cal_dir + '/' + tile_dirs[kk] + '/desaturation'):
		os.mkdir(cal_dir + '/' + tile_dirs[kk] + '/desaturation')

	for ii in range(3):
		if not os.path.exists(cal_dir + '/' + tile_dirs[kk] + '/desaturation/group{}'.format(ii+1)):
			os.mkdir(cal_dir + '/' + tile_dirs[kk] + '/desaturation/group{}'.format(ii+1))

		if not os.path.exists(cal_dir + '/' + tile_dirs[kk] + '/desaturation/group{}/stage3'.format(ii+1)):
			os.mkdir(cal_dir + '/' + tile_dirs[kk] + '/desaturation/group{}/stage3'.format(ii+1))

		for kkn in range(10):
			if os.path.exists(uncal_dir + '/' + tile_dirs[kk] + '/desaturation/group{aa}/stage3/d{bb}'.format(aa=ii+1,bb=kkn+1)):
				dithers.append(uncal_dir + '/' + tile_dirs[kk] + '/desaturation/group{aa}/stage3/d{bb}'.format(aa=ii+1,bb=kkn+1))

	for kkn in range(10):
		if os.path.exists(uncal_dir + '/' + tile_dirs[kk] + '/stage3/d{}'.format(kkn+1)):
			dithers.append(uncal_dir + '/' + tile_dirs[kk] + '/stage3/d{}'.format(kkn+1))

length_d = len(dithers)

dithers_nav = []
file_list = []
file_list_nav = []
spec_norm_final = []

for kk in range(length_d):
	if os.path.exists(dithers[kk] + '_nav'):
		dithers_nav.append(dithers[kk] + '_nav')

	file_test = dithers[kk] + '/' + file
	if os.path.exists(file_test):
		file_list.append(file_test)

for kk in range(len(dithers_nav)):
	file_test_nav = dithers_nav[kk] + '/' + file_nav
	if os.path.exists(file_test_nav):
		file_list_nav.append(file_test_nav)

length_f = len(file_list)
print('\n{} un-nav files detected'.format(length_f))
print('{} nav files detected\n'.format(len(file_list_nav)))

if not length_f == len(file_list_nav):
	print('WARNING\nFound different numbers of un-navigated and navigated files\n')

hdul_test = fits.open(file_list[0])
hdr_test = hdul_test['SCI'].header
wave = np.arange(hdr_test['NAXIS3'])*hdr_test['CDELT3']+hdr_test['CRVAL3']
hdul_test.close()

if change_wave_grid == True:
	wave_interp = np.arange(np.amin(wave),np.amax(wave)+wave_step,wave_step)
	wave = wave_interp

dz_wave = len(wave)

nem_wave = np.loadtxt('spec_data_' + band + '/nemesis_model.txt',usecols=(0))

arat1 = np.where(wave < wave_min_val)
arat2 = np.where(wave > wave_max_val)

hdul_dq = fits.open('spec_data_' + band + '/dq_comp.fits')
dq = hdul_dq[0].data
hdul_dq.close()

for kk in range(length_f):
	work_dir_p = file_list[kk].replace(file,'')
	work_dir = work_dir_p.replace(uncal_dir,cal_dir)

	work_dir_p_nav = file_list_nav[kk].replace(file_nav,'')
	work_dir_nav = work_dir_p_nav.replace(uncal_dir,cal_dir)

	if not os.path.exists(work_dir):
		os.mkdir(work_dir)

	if not os.path.exists(work_dir_nav):
		os.mkdir(work_dir_nav)

	hdul_file = fits.open(file_list[kk])
	spec_data = hdul_file['SCI'].data
	hdr_spec = hdul_file['SCI'].header
	wave_spec = np.arange(hdr_spec['NAXIS3'])*hdr_spec['CDELT3']+hdr_spec['CRVAL3']

	spec_data = spec_data[find_nearest(wave_spec,np.amin(nem_wave)):find_nearest(wave_spec,np.amax(nem_wave))+1]
	wave_spec = wave_spec[find_nearest(wave_spec,np.amin(nem_wave)):find_nearest(wave_spec,np.amax(nem_wave))+1]

	if not len(wave) == len(spec_data):
		print('WARNING\nSpectral and wavelength data are different lengths\n')

	spec_data_cal = np.zeros((dz_wave,dy,dx))

	for j in range(dy):
		for i in range(dx):
			wave_cal = np.add(wave_spec, band_corr[:,j,i])

			f_c2A = scipy.interpolate.interp1d(wave_cal,spec_data[:,j,i],bounds_error=False,fill_value='extrapolate')
			spec_data_cal[:,j,i] = f_c2A(wave)

			if dq[j][i] == 0:
				spec_data_cal[:,j,i] = [0]*dz_wave

			if remove_pix == True:
				for iii in range(len(x_remove)):
					spec_data_cal[:,y_remove[iii]-1,x_remove[iii]-1] = [0]*dz_wave

			arat3 = np.where(spec_data_cal[:,j,i] < 0)
			spec_data_cal[arat3,j,i] = 0

			spec_data_cal[arat1,j,i] = -10000
			spec_data_cal[arat2,j,i] = -10000

	hdul_file['SCI'].data = spec_data_cal
	if change_wave_grid == True:
		hdul_file['SCI'].header['NAXIS3'] = dz_wave
		hdul_file['SCI'].header['CDELT3'] = wave_step
		hdul_file['SCI'].header['CRVAL3'] = wave[0]
	hdul_file.writeto(file_list[kk].replace(uncal_dir,cal_dir), overwrite=True)

	if not len(file_list_nav) == 0:
		hdul_nav_file = fits.open(file_list_nav[kk])
		hdul_nav_file['SCI'].data = spec_data_cal
		if change_wave_grid == True:
			hdul_file['SCI'].header['NAXIS3'] = dz_wave
			hdul_file['SCI'].header['CDELT3'] = wave_step
			hdul_file['SCI'].header['CRVAL3'] = wave[0]
		hdul_nav_file.writeto(file_list_nav[kk].replace(uncal_dir,cal_dir), overwrite=True)


#=====================================================================


print('End of script\n')
