"""
Generates an spx file for NEMESIS use based on user inputs

Usage
	python -W ignore gen_spx.py [file]

Where [file] is the name of the file containing the radiance in units of
W/cm2/sr/um
Format of the file should be
	Wavelength (um)     Radiance

"""
#===========================================================================
#Inputs


#In degrees
longitude = 294.638088
latitude = -12.057455
emission = 54.777573

# Percentage
uncertainty = 10
offset = 0.00299

wmin = 7.490
wmax = 11.578


#===========================================================================
#Imports


import numpy as np
import sys
name = sys.argv[1]


#===========================================================================
#Functions


#Finds nearest array wavelength value to specified range
#Returns index of that wavelength
def find_nearest(array,value):
	array = np.asarray(array)
	idx = (np.abs(array - value)).argmin()
	return idx


#===========================================================================
#Main

data = np.loadtxt(name)
wave = data[:,0]
rad = data[:,1]

idx_start = find_nearest(wave,wmin)
idx_stop = find_nearest(wave,wmax)
wave_trimmed = wave[idx_start:idx_stop]
rad_trimmed = rad[idx_start:idx_stop]

wave_trimmed += offset

length = len(rad_trimmed)
delta = uncertainty/100

ave = np.mean(rad_trimmed)
drad = ave*delta
deltrad = [drad]*length

c = [None]*3
c[0] = wave_trimmed
c[1] = rad_trimmed
c[2] = deltrad

d = np.transpose(c)

np.savetxt('jwst_mrs.spx',d,fmt='%.5e',delimiter='	',header='{aa}      {ba}	0.0000      {ca}      0.0000	 1.0000'.format(aa=latitude,ba=longitude,ca=emission),comments='0.0000 {da} {ea} 1\n        {fa}\n       1\n'.format(da=latitude,ea=longitude,fa=length))


#============================================================================


print('End of script\n')
