#=================================================================
"""
Copies the files to be used to generate the calibration solution
to a new directory: cal_cubes

Usage
		python copy_cal.py

"""
#=================================================================
# Inputs


uncal_dir = 'nov_1246'
epochs = ['july','aug']
tile_dirs = ['july/centre', 'july/east', 'aug/centre', 'aug/west', 'aug/east']

sat_bands = ['ch1-short','ch2-long']
use_group = [3,3]


#=================================================================
# Imports


import numpy as np
from astropy.io import fits
import glob
import shutil
import os


#=================================================================
# Main


if os.path.exists('cal_cubes'):
	shutil.rmtree('cal_cubes')
os.mkdir('cal_cubes')

length_t = len(tile_dirs)

for kk in range(len(epochs)):
	if not os.path.exists('cal_cubes/' + epochs[kk]):
		os.mkdir('cal_cubes/' + epochs[kk])

for kk in range(length_t):
	print(tile_dirs[kk])

	if not os.path.exists('cal_cubes/' + tile_dirs[kk]):
		os.mkdir('cal_cubes/' + tile_dirs[kk])

	sstring = uncal_dir + '/' + tile_dirs[kk] + '/stage3/d*'
	dithers = glob.glob(sstring)

	for i in range(len(dithers)):
		string = dithers[i].replace('stage3/','')
		shutil.copytree(dithers[i],'cal_cubes/' + string.replace(uncal_dir,''))

		for ii in range(len(sat_bands)):
			print('Exchanging {}\n'.format(sat_bands[ii]))

			try:
				ch1st1 = dithers[i].replace('stage3/','desaturation/group{}/stage3/'.format(use_group[ii])) + '/Level3_' + sat_bands[ii] + '_*.fits'
				ch1na1 = glob.glob(ch1st1)
				ch1n1 = ch1na1[0]

				ch1st2 = 'cal_cubes/' + string.replace(uncal_dir,'') + '/Level3_' + sat_bands[ii] + '_*.fits'
				ch1na2 = glob.glob(ch1st2)
				ch1n2 = ch1na2[0]

				os.remove(ch1n2)
				shutil.copy(ch1n1,ch1n2)

			except:
				print('WARNING')
				print(dithers[i].replace('stage3/','desaturation/group{}/stage3/'.format(use_group[ii])) + '/Level3_' + sat_bands[ii] + '_*.fits')
				print('Does not exist\n')


#=================================================================


print('End of script\n')
