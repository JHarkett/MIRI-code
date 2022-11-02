#==============================================================
"""
Produces an RGB image of mosaic data based on user inputted
wavelengths


Usage:
		python visualise_mosaic.py


"""
#==============================================================
# Inputs


single_file_dir = 'centre/stage3/d1/Level3_ch2-short_s3d.fits'
mosaic_file_dir = 'mosaics/c2A/c2_short_all_interp.fits'
map_dir = 'mosaics/c2A'

R_wave = 8.56434
G_wave = 8.57306
B_wave = 8.62303

name = 'c2_mosaic.png'

assign_axes = True
lon_range = [305,270]
lat_range = [-30,-10]

invert = False


#==============================================================
# Imports


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider,Button
import os
import glob
from astropy.io import fits
import scipy
from PIL import Image


#==============================================================
# Functions


def find_nearest(array,value):
	array = np.asarray(array)
	idx = (np.abs(array - value)).argmin()
	return idx

def NormalizeData(data):
	return (data - (np.nanmin(data)-1)) / ((np.nanmax(data)+1) - (np.nanmin(data)-1))


#==============================================================
# Main


hdul = fits.open(single_file_dir)
hdr = hdul['SCI'].header
hdul.close()
wave = np.arange(hdr['NAXIS3'])*hdr['CDELT3']+hdr['CRVAL3']

hdul2 = fits.open(mosaic_file_dir)
spec_data = hdul2[0].data
hdul2.close()

dz,dy,dx = spec_data.shape


for k in range(dz):
	for j in range(dy):
		for i in range(dx):
			if spec_data[k][j][i] == 0:
				spec_data[k][j][i] = np.nan


R_idx = find_nearest(wave,R_wave)
G_idx = find_nearest(wave,G_wave)
B_idx = find_nearest(wave,B_wave)

print(R_idx,G_idx,B_idx)

R = [None]*len(spec_data)
for i in range(len(spec_data)):
	R[i] = spec_data[i][int(dy/2)][int(dx/2)]

plt.plot(wave,R,'k-',linewidth=0.5)
plt.scatter(wave[R_idx],R[R_idx],linewidth=0.3,s=20,c='r')
plt.scatter(wave[G_idx],R[G_idx],linewidth=0.3,s=20,c='g')
plt.scatter(wave[B_idx],R[B_idx],linewidth=0.3,s=20,c='b')
plt.xlabel('Wavelength ($\mu m$)')
plt.ylabel('Surface Brightness (MJy sr$^{-1}$)')
plt.tight_layout()
plt.grid()
plt.show()



#R_array = spec_data[R_idx-1:R_idx+1]
#G_array = spec_data[G_idx-1:G_idx+1]
#B_array = spec_data[B_idx-1:B_idx+1]

#R_frame = np.nanmedian(R_array,axis=0)
#G_frame = np.nanmedian(G_array,axis=0)
#B_frame = np.nanmedian(B_array,axis=0)

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


if invert == True:
	rgbArray = np.zeros((dy,dx,3), 'uint8')
	rgbArray[..., 0] = B_frame*256
	rgbArray[..., 1] = G_frame*256
	rgbArray[..., 2] = R_frame*256

else:
	rgbArray = np.zeros((dy,dx,3), 'uint8')
	rgbArray[..., 0] = R_frame*256
	rgbArray[..., 1] = G_frame*256
	rgbArray[..., 2] = B_frame*256


img = Image.fromarray(rgbArray)


plt.imshow(img,origin='lower')
plt.axis('off')
plt.xlim(25,dx-26)
plt.ylim(25,dy-26)
plt.savefig('rgb_images/' + name)
plt.show()


hdul_lon = fits.open(map_dir + '/longitude_grid.fits')
lon_grid = hdul_lon[0].data
hdul_lon.close()

hdul_lat = fits.open(map_dir + '/latitude_grid.fits')
lat_grid = hdul_lat[0].data
hdul_lat.close()

dy_l,dx_l = lon_grid.shape

lon_arr = [None]*dx_l
lat_arr = [None]*dy_l

for i in range(dx_l):
	lon_arr[i] = lon_grid[0][i]
for i in range(dy_l):
	lat_arr[i] = lat_grid[i][0]



x_label_array = np.arange((round(np.amax(lon_arr)/5)*5)+5,(round(np.amin(lon_arr)/5)*5)-5,-5)
lengthx = len(x_label_array)
xtick_array = [None]*lengthx
for i in range(lengthx):
	xtick_array[i] = find_nearest(lon_arr,x_label_array[i])

y_label_array = np.arange((round(np.amin(lat_arr)/5)*5)-5,(round(np.amax(lat_arr)/5)*5)+5,5)
lengthy = len(y_label_array)
ytick_array = [None]*lengthy
for i in range(lengthy):
	ytick_array[i] = find_nearest(lat_arr,y_label_array[i])


plt.imshow(img,origin='lower')
plt.xticks(xtick_array,x_label_array)
plt.yticks(ytick_array,y_label_array)

if assign_axes == True:
	plt.xlim(find_nearest(lon_arr,lon_range[0]),find_nearest(lon_arr,lon_range[1]))
	plt.ylim(find_nearest(lat_arr,lat_range[0]),find_nearest(lat_arr,lat_range[1]))
	name1 = 'map_zoom_'

else:
	plt.xlim(1,dx_l-2)
	plt.ylim(1,dy_l-2)
	name1 = 'map_'

plt.xlabel('System III West Longitude ($^\circ$)')
plt.ylabel('Planetographic Latitude ($^\circ$)')
plt.savefig('rgb_images/' + name1 + name)
plt.show()


#==============================================================


print('End of script\n')
