#=====================================================================
"""

Extracts the spaxels for a particular band, applies doppler shifting to them,
determines a correction for these spectra and saves in a fits file of the
same shape as the original data fits file

Output: the physical calibration values

Usage:
		python extract_calibration.py

"""
#=====================================================================
# Inputs


uncal_dir = 'nov_1246'

epochs = ['july','aug']
tile_dirs = ['july/centre', 'july/east', 'aug/centre', 'aug/west']
band = 'ch2-medium'

wave_min_val = 3.0
wave_max_val = 10.75

col_threshold = 10


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


#=====================================================================
# Functions


def find_nearest(array,value):
	array = np.asarray(array)
	idx = (np.abs(array - value)).argmin()
	return idx


#=====================================================================
# Main


if os.path.exists('cal_file.txt'):
        os.remove('cal_file.txt')

f = open('cal_file.txt','a')
f.write('Uncalibrated data directory:\n' + uncal_dir + '\n')
f.write('\nEpochs:\n')
for kk in range(len(epochs)):
	f.write(epochs[kk] + '\n')
f.write('\nTile directories:\n')
for kk in range(len(tile_dirs)):
	f.write(tile_dirs[kk] + '\n')
f.write('\nBand:\n' + band + '\n')
f.write('\nMinimum wave value:\n{aa}\n\nMaximum wave value:\n{bb}'.format(aa=wave_min_val,bb=wave_max_val))
f.close()


file = 'Level3_' + band + '_s3d.fits'

if os.path.exists('spec_data_' + band):
	shutil.rmtree('spec_data_' + band)

os.mkdir('spec_data_' + band)

length_t = len(tile_dirs)

dithers = []

for kk in range(length_t):
	for k in range(10):
		if os.path.exists('cal_cubes/' + tile_dirs[kk] + '/d{}'.format(k)):
			dithers.append('cal_cubes/' + tile_dirs[kk] + '/d{}'.format(k))

length_d = len(dithers)

print('\n{} directories detected'.format(length_d))

file_list = []
spec_norm_final = []

for kk in range(length_d):
	file_test = dithers[kk] + '/' + file
	if os.path.exists(file_test):
		file_list.append(file_test)

length_f = len(file_list)
print('{} files detected\n'.format(length_f))

hdul_meas = fits.open(file_list[0])
spec_data = hdul_meas['SCI'].data
dz_nouse,dy,dx = spec_data.shape
hdr_meas = hdul_meas['SCI'].header
wave = np.arange(hdr_meas['NAXIS3'])*hdr_meas['CDELT3']+hdr_meas['CRVAL3']
hdul_meas.close()

wave =  wave[find_nearest(wave,wave_min_val):find_nearest(wave,wave_max_val)+1]
dz = len(wave)

nem_data = np.loadtxt('mre_files/' + band + '.txt')
wave_nem = nem_data[:,0]
flux_nem = nem_data[:,1]

c = 2.99792458e+8
wave_nem_um = wave_nem * (1e-6)

flux_nem = flux_nem * (1e+4) # W m-2 sr-1 m-1
flux_nem = (flux_nem * wave_nem_um**2) / c # W m-2 sr-1 Hz-1
flux_nem = flux_nem / (1e-26) # Jy/sr-1
spec_nem = flux_nem / (1e+6) # MJy/sr-1

f_c2A = scipy.interpolate.interp1d(wave_nem,spec_nem,bounds_error=False,fill_value=None)
spec_nem = f_c2A(wave)

aa = [None]*2
aa[0] = wave
aa[1] = spec_nem
bb = np.transpose(aa)
np.savetxt('spec_data_' + band + '/nemesis_model.txt',bb,fmt='%.8e',delimiter='	',header='Wave (µm)	Surf brightness (MJy sr-1)')

spec_data = [None]*length_f
data_noblank = [None]*length_f

for kk in range(length_f):
	hdul_file = fits.open(file_list[kk])
	spec_data[kk] = hdul_file['SCI'].data
	dq = hdul_file['DQ'].data
	hdul_file.close()

	spec_data[kk] = spec_data[kk][find_nearest(wave,wave_min_val):find_nearest(wave,wave_max_val)+1]

	data_noblank[kk] = [None]*dz

	for k in range(dz):
		data_noblank[kk][k] = np.reshape(spec_data[kk][k],(dx*dy)) 
		data_noblank[kk][k][np.isnan(data_noblank[kk][k])] = 0

	data_noblank[kk] = np.transpose(data_noblank[kk])

data_noblank = np.array(data_noblank)

id = np.arange(1,(dy*dx)+1)
d = np.reshape(id,(dy,dx))
hdu2 = fits.PrimaryHDU(d)
hdu2.writeto('spec_data_' + band + '/retrieval_map.fits',overwrite=True)

dq_comp = []

for kk in range(len(id)):
	print(id[kk])

	os.mkdir('spec_data_' + band + '/{}'.format(id[kk]))

	length_spect = len(data_noblank[:,kk,:])

	data_save = []

	for ii in range(length_spect):
		check = [False]*len(data_noblank[ii,kk,:])
		check = np.array(check)

		idx_ch = np.where(data_noblank[ii,kk,:] == 0)
		check[idx_ch] = True

		check_sum = sum(check)

		if check_sum < col_threshold:
			data_save.append(data_noblank[ii,kk,:])


	data_save = np.array(data_save)
	length_save = len(data_save)

	if length_save == 0:
		dq_comp.append(0)
	else:
		dq_comp.append(1)

	a = [None]*(length_save + 2)

	a[0] = wave
	a[1] = spec_nem

	head = 'Wave (µm)	Spec nem (MJy sr-1)'

	for ii in range(length_save):
		head = head + '	spec {} (MJy sr-1)'.format(ii+1)

		a[ii+2] = data_save[ii]

	a = np.array(a)
	b = np.transpose(a)

	np.savetxt('spec_data_' + band + '/{}/spectra.txt'.format(id[kk]),b,fmt='%.8e',delimiter='	',header=head)

	shutil.copy('correct_spaxel.py','spec_data_' + band + '/{}/correct_spaxel.py'.format(id[kk]))
	shutil.copy('submit_caljob','spec_data_' + band + '/{}/submit_caljob'.format(id[kk]))

	os.chdir('spec_data_' + band + '/{}'.format(id[kk]))
	os.system('qsub submit_caljob')
	os.chdir('../..')


dq_comp2 = np.reshape(dq_comp,(dy,dx))
hdu_dq2 = fits.PrimaryHDU(dq_comp2)
hdu_dq2.writeto('spec_data_' + band + '/dq_comp.fits',overwrite=True)


#=====================================================================


print('End of script\n')
