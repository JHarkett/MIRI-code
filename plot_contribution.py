#=============================================================
"""
Plots contribution functions for forward model results that have
utilized ONE variable only

Uses data from kk.dat

Saves plots in a new directory: contribution_plots

Usage:

	Set show below to True or False

	python -W ignore plot_contribution.py


"""
#=============================================================
# Inputs

# Show plots?
# True = yes, False = No
# The plots will be saved regardless
show = False


#=============================================================
# Imports


import numpy as np
import matplotlib.pyplot as plt
import os
import glob


#=============================================================
# Main


sstring = 'core_*'
directories = glob.glob(sstring)
length = len(directories)


if not os.path.exists('contribution_plots'):
	os.mkdir('contribution_plots')


for i in range(6):
	print('core_{}'.format(i+1))

	with open('core_{}/kk.dat'.format(i+1)) as f:
		data = f.readlines()
	f.close()

	par_array = data[0].split()
	n_press = int(par_array[0])
	n_wave = int(par_array[1])


	# Contribution (Jacobian) function
	c_data = np.loadtxt('core_{}/kk.dat'.format(i+1),skiprows=1)
	c_line = np.reshape(c_data,(n_press * n_wave))
	c_array = np.reshape(c_line,(n_wave,n_press))

	len_arr2 = len(c_array)

	c_maxw = [None]*len_arr2
	pressure_max = [None]*len_arr2

	for j in range(len_arr2):
		c_maxw[j] = np.argmax(c_array[j])

	c_array_t = np.transpose(c_array)

	# Pressure data
	p_data = np.loadtxt('core_{}/nemesis.prf'.format(i+1),skiprows=15,usecols=(1))
	p_data *= 1013.25

	# Wavelengths
	w_data = np.loadtxt('core_{}/nemesis.mre'.format(i+1),skiprows=5,usecols=(1),max_rows=n_wave)


	plt.pcolor(w_data,p_data,c_array_t,cmap='inferno')
	plt.colorbar()
	plt.ylim(max(p_data),min(p_data))
	plt.yscale('log')
	plt.xlabel('Wavelength ($\mu m$)')
	plt.ylabel('Pressure ($atm$)')
	plt.savefig('contribution_plots/core_{}_contribution.png'.format(i+1))
	if show == True:
		plt.show()
	plt.clf()


	new_w_array = [None]*120
	new_p_array = [None]*120
	for j in range(120):
		new_w_array[j] = w_data		
		new_p_array[j] = [p_data[j]]*n_wave


	for j in range(len_arr2):
		pressure_max[j] = p_data[c_maxw[j]]


	plt.plot(w_data,pressure_max,'k-',linewidth=0.5)
	plt.xlabel('Wavelength ($\mu m$)')
	plt.ylabel('Pressure($atm$)')
	plt.ylim(max(pressure_max),min(pressure_max))
	plt.grid()
	plt.yscale('log')
	plt.savefig('contribution_plots/core_{}_max_pressure.png'.format(i+1))
	if show == True:
		plt.show()
	plt.clf()


#=============================================================


print('End of script\n')
