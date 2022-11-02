#===============================================================
"""
Generates the interpolated spectral and mu files required for
the mosaic, also generates longitude and latitude maps for the
purposes of generating mercator-style projections


Usage:
		python build_mosaic_1.py



"""
#===============================================================
# Inputs


files1_path = 'centre/stage3/d*_nav/Level3_ch2-short_s3d_nav.fits'
files2_path = 'east/stage3/d*_nav/Level3_ch2-short_s3d_nav.fits'
files3_path = 'west/stage3/d*_nav/Level3_ch2-short_s3d_nav.fits'
main_dir = 'mosaics/ch2A_stratosphere'


#===============================================================
# Imports


import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import shutil
import scipy.interpolate
from astropy.io import fits


#===============================================================
# Main


files1 = glob.glob(files1_path)
files2 = glob.glob(files2_path)
files3 = glob.glob(files3_path)
files = np.concatenate((files1,files2,files3))

if not os.path.exists(main_dir):
	os.mkdir(main_dir)

lengthf = len(files)

spec_data = [None]*lengthf
lat_min = [None]*lengthf
lat_max = [None]*lengthf
lon_min = [None]*lengthf
lon_max = [None]*lengthf
lat_data = [None]*lengthf
lon_data = [None]*lengthf
mu_data = [None]*lengthf
dz = [None]*lengthf
wavelist = [None]*lengthf

for kk in range(lengthf):
	hdul1 = fits.open(files[kk])
	spec_data[kk] = hdul1['SCI'].data
	lat_data[kk] = hdul1['LAT_PGR'].data
	lon_data[kk] = hdul1['LON_WEST'].data
	mu_data[kk] = hdul1['MU'].data
	hdr1 = hdul1[0].header
	hdr = hdul1['SCI'].header
	wavelist[kk] = np.arange(hdr['NAXIS3'])*hdr['CDELT3'] + hdr['CRVAL3']
	hdul1.close()

	dz[kk],dy,dx = spec_data[kk].shape

	for k in range(dz[kk]):
		for j in range(dy):
			for i in range(dx):
				if spec_data[kk][k][j][i] == 0:
					spec_data[kk][k][j][i] = np.nan


	lat_min[kk] = np.amin(lat_data[kk])
	lat_max[kk] = np.amax(lat_data[kk])
	lon_min[kk] = np.amin(lon_data[kk])
	lon_max[kk] = np.max(lon_data[kk])

lat_min_total = int(np.amin(lat_min))-20.0
lat_max_total = int(np.amax(lat_max))+20.0
lon_min_total = int(np.amin(lon_min))-20.0
lon_max_total = int(np.amax(lon_max))+20.0


lat_array = np.arange(lat_min_total,lat_max_total+0.1,0.5)
lon_array = np.arange(lon_max_total,lon_min_total-0.1,-0.5)

lon_grid,lat_grid = np.meshgrid(lon_array,lat_array)

hdu111 = fits.PrimaryHDU(lon_grid)
hdu111.writeto(main_dir + '/longitude_grid.fits',overwrite=True)

hdu222 = fits.PrimaryHDU(lat_grid)
hdu222.writeto(main_dir + '/latitude_grid.fits',overwrite=True)


spec_data_out = [None]*lengthf
mu_data_out = [None]*lengthf

for kk in range(lengthf):
	spec_data_out[kk] = [None]*dz[kk]
	print(kk)
	for k in range(dz[kk]):
		points = np.asarray([(long, lat) for long, lat in zip(lon_data[kk].ravel(), lat_data[kk].ravel())])
		good_points = np.isfinite(points[:, 0]) & np.isfinite(points[:, 1])
		spec_data_out[kk][k] = scipy.interpolate.griddata(points[good_points],spec_data[kk][k].ravel()[good_points],(lon_grid, lat_grid))

	points = np.asarray([(long, lat) for long, lat in zip(lon_data[kk].ravel(), lat_data[kk].ravel())])
	good_points = np.isfinite(points[:, 0]) & np.isfinite(points[:, 1])
	mu_data_out[kk] = scipy.interpolate.griddata(points[good_points],mu_data[kk].ravel()[good_points],(lon_grid, lat_grid))

	hdu_spec = fits.PrimaryHDU(spec_data_out[kk])
	hdu_spec.writeto(main_dir + '/interspec_{}.fits'.format(kk+1),overwrite=True)
	hdu_mu = fits.PrimaryHDU(mu_data_out[kk])
	hdu_mu.writeto(main_dir + '/mu_{}.fits'.format(kk+1),overwrite=True)

	np.savetxt(main_dir + '/wavelengths_{}.txt'.format(kk+1),wavelist[kk],fmt='%.6f')


#===============================================================


print('End of script\n')
