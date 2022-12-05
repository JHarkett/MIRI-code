#=====================================================================
"""
Calibrates the spectra using multiple spaxels from different exposures

Usage

	THIS CODE IS RUN REMOTELY FROM ANOTHER SCRIPT

"""
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


#=====================================================================
# Functions


# Finds index of the nearest array element to the value
def find_nearest(array,value):
	array = np.asarray(array)
	idx = (np.abs(array - value)).argmin()
	return idx


# Happens if there are no spectra (due to excessive 0 values)
# No correction will be applied to this pixel
def no_correction(data):
	wave = data[:,0]
	len_w = len(wave)

	corr_comp = [0]*len_w

	a = [None]*2
	a[0] = wave
	a[1] = corr_comp

	b = np.transpose(a)

	np.savetxt('correction.txt',b,fmt='%.8e',delimiter='	',header='Wave (µm)	Correction (MJy sr-1)')


# If spectra are present: determines the required correction
def correction(data, cols, rows, band):
	marker = True

	wave = data[:,0]
	spec_nem = data[:,1]

	n_spectra = cols - 2

	spec_right_use = [None]*n_spectra
	wave_cal = [None]*n_spectra
	wave_use = [None]*n_spectra
	a_use = [None]*n_spectra
	b_use = [None]*n_spectra
	spec_right = [None]*n_spectra
	wave_section = [None]*n_spectra
	spec_right_section = [None]*n_spectra
	spec_nem_section = [None]*n_spectra

	if not os.path.exists('data'):
		os.mkdir('data')

	for kkk in range(n_spectra):
		spec_right[kkk] = data[:,kkk+2]

		w_start = []
		w_end = []

		w_use = np.amin(wave)
		w_maxx = np.amax(wave)

		if 'ch1-short' in band or 'ch1-medium' in band:
			print('Win width: 0.2')
			win_width = 0.2
			gap = 0.1
		else:
			print('Win width: 0.1')
			win_width = 0.1
			gap = 0.05

		while w_use < (w_maxx - win_width):
			w_start.append(w_use)
			w_end.append(w_use + win_width)
			w_use = w_use + gap

		w_start.append(w_use)
		w_end.append(w_maxx)

		length_b = len(w_start)

		wave_section[kkk] = [None]*length_b
		spec_right_section[kkk] = [None]*length_b
		spec_nem_section[kkk] = [None]*length_b


		for iii in range(length_b):

			idx_min = find_nearest(wave,w_start[iii])
			idx_max = find_nearest(wave,w_end[iii])+1

			wave_section[kkk][iii] = wave[idx_min:idx_max]
			spec_right_section[kkk][iii] = spec_right[kkk][idx_min:idx_max]
			spec_nem_section[kkk][iii] = spec_nem[idx_min:idx_max]

			factor = np.divide(np.nanmedian(spec_right_section[kkk][iii]),np.nanmedian(spec_nem_section[kkk][iii]))
			spec_nem_section[kkk][iii] = np.multiply(spec_nem_section[kkk][iii],factor)


		#---------------------------------------------------
		# Start of the calibration
		marker = False

		print('Start of calibration\n')

		if not os.path.exists('data/spec{}'.format(kkk+1)):
			os.mkdir('data/spec{}'.format(kkk+1))

		spec_right_use[kkk] = [None]*length_b
		wave_cal[kkk] = [None]*length_b
		wave_use[kkk] = [None]*length_b
		a_use[kkk] = [None]*length_b
		b_use[kkk] = [None]*length_b

		for iii in range(length_b):
			a_prior = 1.0
			b_prior = 0.0

			usea = 10
			useb = 10

			da = [0.01,0.005,0.001,0.0005,0.0001,0.00005,0.00001,0.000005,0.000001,0.0000005,0.0000001]
			db = [0.01,0.005,0.001,0.0005,0.0001,0.00005,0.00001,0.000005,0.000001,0.0000005,0.0000001]

			for kk in range(len(da)):

				a_try = np.arange(a_prior-(da[kk]*usea),a_prior+(da[kk]*usea),da[kk])
				b_try = np.arange(b_prior-(db[kk]*useb),b_prior+(db[kk]*useb),db[kk])

				lengtha = len(a_try)
				lengthb = len(b_try)

				pb = [None]*lengtha
				pb_max = [None]*lengtha

				for j in range(lengtha):
					pb[j] = [None]*lengthb

					for i in range(lengthb):

						wave_try = (a_try[j]*wave_section[kkk][iii]) + b_try[i]

						f_c2A = scipy.interpolate.interp1d(wave_try,spec_right_section[kkk][iii],bounds_error=False,fill_value=0)
						spec_right_new2 = f_c2A(wave_section[kkk][iii])

						spec_right_new2 = np.array(spec_right_new2)
						spec_nem_new = np.array(spec_nem_section[kkk][iii])

						current = np.where(spec_right_new2 == 0)
						current2 = np.where(np.isnan(spec_nem_new) == True)

						spec_right_new2[current] = np.nan
						spec_nem_new[current] = np.nan
						spec_right_new2[current2] = np.nan
						spec_nem_new[current2] = np.nan

						spec_right_new2 = ma.masked_invalid(spec_right_new2)
						spec_nem_new = ma.masked_invalid(spec_nem_new)

						p_arr = ma.corrcoef(spec_nem_new,spec_right_new2)
						pb[j][i] = p_arr[0][1]

						if ma.is_masked(pb[j][i]) == True:
							pb[j][i] = 0

						test_mas = sum(spec_right_new2.mask)/len(spec_right_new2)
						if test_mas > 0.5:
							pb[j][i] = 0

					pb_max[j] = np.nanmax(pb[j])

				idx_amax = find_nearest(pb_max,np.nanmax(pb_max))
				a_prior = a_try[idx_amax]
				pa_use = pb_max[idx_amax]

				idx_bmax = find_nearest(pb[idx_amax],np.nanmax(pb[idx_amax]))
				b_prior = b_try[idx_bmax]
				pb_use = pb[idx_amax][idx_bmax]

			a_use[kkk][iii] = a_prior
			b_use[kkk][iii] = b_prior

			wave_use[kkk][iii] = a_use[kkk][iii]*(wave_section[kkk][iii]) + b_use[kkk][iii]
			wave_cal[kkk][iii] = wave_use[kkk][iii] - wave_section[kkk][iii]
			f_c2A = scipy.interpolate.interp1d(wave_use[kkk][iii],spec_right_section[kkk][iii],bounds_error=False,fill_value='extrapolate')
			spec_right_use[kkk][iii] = f_c2A(wave_section[kkk][iii])

			np.savetxt('data/spec{}/'.format(kkk+1) + 'spec_right_data_{}.txt'.format(iii),spec_right_use[kkk][iii],fmt='%.8f',delimiter='	')
			np.savetxt('data/spec{}/'.format(kkk+1) + 'wave_cal_data_{}.txt'.format(iii),wave_cal[kkk][iii],fmt='%.8f',delimiter='	')
			np.savetxt('data/spec{}/'.format(kkk+1) + 'wave_use_data_{}.txt'.format(iii),wave_use[kkk][iii],fmt='%.8f',delimiter='	')

		np.savetxt('data/spec{}/a_data.txt'.format(kkk+1), a_use[kkk],fmt='%.8f',delimiter='	')
		np.savetxt('data/spec{}/b_data.txt'.format(kkk+1),b_use[kkk],fmt='%.8f',delimiter='	')


		# End of the calibration
		#---------------------------------------------------


	# Figure generation

	if marker == True:
		a_use = [None]*n_spectra
		b_use = [None]*n_spectra
		spec_right_use = [None]*n_spectra
		wave_cal = [None]*n_spectra
		wave_use = [None]*n_spectra

		for kkk in range(n_spectra):

			a_use[kkk] = np.loadtxt('data/spec{}/a_data.txt'.format(kkk+1))
			b_use[kkk] = np.loadtxt('data/spec{}/b_data.txt'.format(kkk+1))

			spec_right_use[kkk] = [None]*length_b
			wave_cal[kkk] = [None]*length_b
			wave_use[kkk] = [None]*length_b

			for iii in range(length_b):
				spec_right_use[kkk][iii] = np.loadtxt('data/spec{}/'.format(kkk+1) + 'spec_right_data_{}.txt'.format(iii))
				wave_cal[kkk][iii] = np.loadtxt('data/spec{}/'.format(kkk+1) + 'wave_cal_data_{}.txt'.format(iii))
				wave_use[kkk][iii] = np.loadtxt('data/spec{}/'.format(kkk+1) + 'wave_use_data_{}.txt'.format(iii))

	wave_section2 = [None]*n_spectra
	wave_cal2 = [None]*n_spectra
	wave_section_comp = [None]*n_spectra
	wave_cal_comp = [None]*n_spectra
	spec_right_use = [None]*n_spectra
	wave_add = [None]*n_spectra
	idx_midi = [None]*n_spectra

	for kkk in range(n_spectra):
		wave_section2[kkk] = [None]*length_b
		wave_cal2[kkk] = [None]*length_b
		wave_section_comp[kkk] = [None]*length_b
		wave_cal_comp[kkk] = [None]*length_b
		spec_right_use[kkk] = [None]*length_b
		wave_add[kkk] = [None]*length_b
		idx_midi[kkk] = [None]*length_b

		for iii in range(length_b):
			idx_middle = int(len(wave_section[kkk][iii])/2)
			wave_section2[kkk][iii] = wave_section[kkk][iii][idx_middle]
			wave_cal2[kkk][iii] = wave_cal[kkk][iii][idx_middle]

		wave_section2[kkk] = np.array(wave_section2[kkk])
		wave_cal2[kkk] = np.array(wave_cal2[kkk])


		#---------------------------------------------------
		# Savgol filter


		win_add = len(wave_cal[kkk])
		if win_add % 2 == 0:
			win_add = win_add-1

		wave_section_comp[kkk] = wave_section2[kkk]
		wave_cal_comp[kkk] = wave_cal2[kkk]

		wave_add[kkk] = savgol_filter(wave_cal2[kkk], window_length = win_add, polyorder=3)

		f_c2A = scipy.interpolate.interp1d(wave_section2[kkk],wave_add[kkk],bounds_error=False,fill_value='extrapolate')
		wave_add2 = f_c2A(wave)


		#---------------------------------------------------
		# The correction


		wave_use = np.add(wave, wave_add2)
		f_c2A = scipy.interpolate.interp1d(wave_use,spec_right[kkk],bounds_error=False,fill_value='extrapolate')
		spec_right_use[kkk] = f_c2A(wave)

		idx_midi[kkk] = int(len(wave)/2)


	#---------------------------------------------------
	# Composite correction


	wave_section_comp = np.array(wave_section_comp)
	wave_cal_comp = np.array(wave_cal_comp)

	wave_section_comp = np.median(wave_section_comp,axis=0)
	wave_cal_comp = np.median(wave_cal_comp,axis=0)


	#---------------------------------------------------
	# Outlier detection and removal


	sd = np.std(wave_cal_comp)
	median_wa = np.median(wave_cal_comp)

	arral1 = np.where(wave_cal_comp > (median_wa + (2*sd)))
	arral1 = arral1[0]
	arral2 = np.where(wave_cal_comp < (median_wa - (2*sd)))
	arral2 = arral2[0]

	if not len(arral1) == 0:
		for kk in range(len(arral1)):
			if arral1[kk] > 0 and arral1[kk] < (len(wave_cal_comp)-1):
				print('Calibrating')
				print('Old Value: {}'.format(wave_cal_comp[arral1[kk]]))
				wave_cal_comp[arral1[kk]] = np.median([wave_cal_comp[arral1[kk]-1], wave_cal_comp[arral1[kk]+1]])
				print('New Value: {}'.format(wave_cal_comp[arral1[kk]]))
			if arral1[kk] == 0:
				print('Else: 0')
				wave_cal_comp[arral1[kk]] = wave_cal_comp[arral1[kk] + 1]
			if arral1[kk] == len(wave_cal_comp) - 1:
				print('Else: last')
				wave_cal_comp[arral1[kk]] = wave_cal_comp[arral1[kk] - 1]

	if not len(arral2) == 0:
		for kk in range(len(arral2)):
			if arral2[kk] > 0 and arral2[kk] < (len(wave_cal_comp)-1):
				print('Calibrating')
				print('Old Value: {}'.format(wave_cal_comp[arral2[kk]]))
				wave_cal_comp[arral2[kk]] = np.median([wave_cal_comp[arral2[kk]-1], wave_cal_comp[arral2[kk]+1]])
				print('New Value: {}'.format(wave_cal_comp[arral2[kk]]))
			if arral2[kk] == 0:
				print('Else: 0')
				wave_cal_comp[arral2[kk]] = wave_cal_comp[arral2[kk] + 1]
			if arral2[kk] == len(wave_cal_comp) - 1:
				print('Else: last')
				wave_cal_comp[arral2[kk]] = wave_cal_comp[arral2[kk] - 1]


	#---------------------------------------------------


	wave_add_comp = savgol_filter(wave_cal_comp, window_length = win_add, polyorder=3)

	f_c2A = scipy.interpolate.interp1d(wave_section_comp,wave_add_comp,bounds_error=False,fill_value='extrapolate')
	wave_add_comp_use = f_c2A(wave)

	a = [None]*2
	a[0] = wave
	a[1] = wave_add_comp_use

	b = np.transpose(a)

	np.savetxt('correction.txt',b,fmt='%.8e',delimiter='	',header='Wave (µm)	Correction (MJy sr-1)')


#=====================================================================
# Main


with open('../../cal_file.txt','r') as f:
	cal_data = f.readlines()
f.close()

for kk in range(len(cal_data)):
	if cal_data[kk][0:5] == 'Band:':
		band = cal_data[kk+1]

data = np.loadtxt('spectra.txt')

rows, cols = data.shape

if cols == 2:
	no_correction(data)

else:
	correction(data, cols, rows, band)


#=====================================================================
