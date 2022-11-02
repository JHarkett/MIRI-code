"""
For a specified position, will plot JWST pipeline output spectrum

Usage:
	line 35: input west longitude
	line 36: input pgr latitude
	line 37: input location that spectra figure should be saved to (script will
		create the directory if it does not exist)
	line 38: input name of resulting file
	line 43: input location of MIRI files to plot
	line 46: input first MIRI band to include in plot
	line 47: input last MIRI band to include in plot
	line 50: input minimum surface brightness (if [y/n]=n below), fluxes below
		this will be set to NaN
	line 53: specify whether to show location of chosen pixel on a map for each
		band (used for development of code - leave as False)
	line 56: specify whether to generate a txt file of fluxes/brightness T for
		each band

	python -W ignore plot_pixseperate_v3.py [y/n]

Where [y/n] specifies whether to convert the spectrum to brightness T units or not
y for yes
n for no


"""
#=============================================================================
#Inputs

import sys
input = sys.argv[1]

#Specify which position to plot
lon = 290.1020
lat = -15.5101
spec_dir = 'spectra/pixel_1/'
file_name = 'pix_1.png'

name = spec_dir + file_name

#Specify where output spectra are
out_dir = 'stage3/d1_nav/'

#Specify bands to plot spectrum for (0-12)
wave_start = 0
wave_stop = 6

# Min surface brightness to allow (sets anything below to NaN)
sb_min = 1e+8

# Choose to display chosen position on map (does not affect the spectrum plot)
show_plots = False

# Generates txt files of the spectra if allowed
gen_txt = False


#==============================================================================
#Imports and useful constants


from astropy.io import fits
import numpy as np
import glob
import matplotlib.pyplot as plt
import os

sre = pow(((0.1/3600.0)*(np.pi/180)),2)

c = 2.99792458e8
h=6.626e-34
k=1.3806e-23
c1=2*h*c*c
c2=(h*c)/k

if not os.path.exists('spectra'):
	os.mkdir('spectra')

#Creates save directory if it does not exist
if not os.path.exists(spec_dir):
	os.mkdir(spec_dir)


#============================================================================
#Functions


def find_nearest(array,value):
	array = np.asarray(array)
	idx = (np.abs(array - value)).argmin()
	return idx


#JWST output flux (Jy) and wavelengths for entire wave range
def jwst_output(out_dir,lon,lat):
	sstring = out_dir + '*_s3d_nav.fits'
	s3dfiles = sorted(glob.glob(sstring))
	length = len(s3dfiles)

	s3d_wave=[]
	s3d_flux=[]
	medwaves = []

	ster = [None]*length
	arc = [None]*length

	for i in range(length):
		hdu = fits.open(s3dfiles[i])
		scidata = hdu['SCI'].data
		scihdr = hdu['SCI'].header
		lon_data = hdu['LON_WEST'].data
		lat_data = hdu['LAT_PGR'].data

		ny, nx = lat_data.shape
		x, y = np.meshgrid(range(nx), range(ny))		

		lat_idx = [None]*ny
		lon_array = [None]*ny

		for jj in range(ny):
			lat_idx[jj] = find_nearest(lat_data[jj],lat)
			lon_array[jj] = lon_data[jj][lat_idx[jj]]

		y_idx = find_nearest(lon_array,lon)
		x_idx = lat_idx[y_idx]

		plt.imshow(scidata[0],cmap='inferno',origin='lower')
		cplat = plt.contour(x, y, lat_data, range(-90,90,5), colors='white', linewidths=0.3, linestyles='solid')
		cplon = plt.contour(x,y,lon_data,range(-360,360,5), colors='white',linewidths=0.3, linestyles='solid')
		plt.clabel(cplat)
		plt.clabel(cplon)

		plt.plot(x_idx,y_idx,'bo')
		plt.text(x_idx,y_idx,'({aaa},{bbb})'.format(aaa=lon,bbb=lat))

		if show_plots == True:
			plt.show()
		plt.clf()

		ster[i] = scihdr['PIXAR_SR']

		wave = np.arange(scihdr['NAXIS3'])*scihdr['CDELT3']+scihdr['CRVAL3']
		s3d_wave.append(wave)
		medwaves.append(np.median(wave))

		s3d_flux.append(scidata)

		hdu.close()

	indx = np.argsort(medwaves)
	s3d_wave = np.array(s3d_wave)
	s3d_flux = np.array(s3d_flux)

	s3d_wave = s3d_wave[indx]
	s3d_flux = s3d_flux[indx]

	flux12p = [None]*length

	for i in range(length):
		flux12 = s3d_flux[i]
		length3 = len(flux12)

		flux12p[i] = [None]*length3

		for j in range(length3):
			flux12p[i][j] = flux12[j][y_idx][x_idx]
			flux12p[i][j] = flux12p[i][j]*(1e+6)  #Jy/sr

	return flux12p, s3d_wave, ster


#Converts fluxes into brightness T
def convert_TB(flux,wave,ster):
	length = len(flux)

	flux12p = [None]*length
	v = [None]*length
	radiance_out = [None]*length
	TB_out = [None]*length
	a = [None]*length

	for i in range(length):

		length3 = len(flux[i])		

		radiance_out[i] = [None]*length3
		TB_out[i] = [None]*length3
		a[i] = [None]*length3

		v[i] = 1/(wave[i]*(1e-6))

		for j in range(length3):
			radiance_out[i][j] = (flux[i][j] * 1e-26) # Radiance in W/m2/sr/m-1
			radiance_out[i][j] = radiance_out[i][j] * c # Radiance in W/m2/sr/m$

			a[i][j] = (c1*v[i][j]*v[i][j]*v[i][j])/radiance_out[i][j]
			TB_out[i][j] = (c2*v[i][j])/np.log(a[i][j]+1)
	return TB_out


#Plots and saves composite graph
def plot_all(y2,wave2,y,wave,wave_start,wave_stop,key,name,key2,spec_dir):

	for i in range(wave_start,wave_stop):
		if i == wave_start:
			plt.plot(wave[i],y[i],'k-',label='Jovian spectra',linewidth=0.5)
		else:
			plt.plot(wave[i],y[i],'k-',linewidth=0.5)

		if gen_txt == True:
			txtname = spec_dir + '/{aaaa}_{bbbb}'.format(aaaa=lon,bbbb=lat) + '_' + key2 + '_band_{}'.format(i) + '.txt'
			b = [None]*2
			b[0] = wave[i]
			b[1] = y[i]
			c = np.transpose(b)
			np.savetxt(txtname,c,fmt='%.6f',delimiter='     ')

	plt.legend()
	plt.xlabel('Wavelength ($\mu m$)')
	plt.ylabel(key)
	if key == 'Surface Brightness (Jy sr$^{-1}$)':
		plt.yscale('log')
	plt.tight_layout()
	plt.grid()
	file_name1 = name.replace('.png','')
	file_name2 = file_name1 + '_' + key2 + '.png'
	plt.savefig(file_name2)
	plt.show()


#============================================================================
#Main


substring = 'y'

flux_out, wave_out, ster = jwst_output(out_dir,lon,lat)
flux_out2, wave_out2, ster2 = jwst_output(out_dir,294.1390,-21.5)

if substring in input:
	y = convert_TB(flux_out,wave_out,ster)
	y2 = convert_TB(flux_out2,wave_out2,ster2)
	key = 'Brightness Temperature ($K$)'
	key2 = 'T'

	for kk in range(len(y)):
		y[kk] = np.array(y[kk])
		counter = np.where(y[kk] == 0)
		y[kk][counter] = np.nan

		y2[kk] = np.array(y2[kk])
		counter2 = np.where(y2[kk] == 0)
		y2[kk][counter2] = np.nan

else:
	y = flux_out
	y2 = flux_out2

	key = 'Surface Brightness (Jy sr$^{-1}$)'
	key2 = 'jysr-1'

	for kk in range(len(y)):
		y[kk] = np.array(y[kk])
		counter = np.where(y[kk] < sb_min)
		y[kk][counter] = np.nan

		y2[kk] = np.array(y2[kk])
		counter2 = np.where(y2[kk] < sb_min)
		y2[kk][counter2] = np.nan

plot_all(y2,wave_out2,y,wave_out,wave_start,wave_stop,key,name,key2,spec_dir)


#============================================================================


print('End of script\n')
