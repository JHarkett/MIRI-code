#=====================================================================
"""
Uses the outputs from build_mosaic_1.py to construct the mosaic


Usage:
		python build_mosaic_2.py


"""
#=====================================================================
# Inputs


files1_dir = 'west/stage3/d*_nav/Level3_ch1-short_s3d_nav.fits'
files2_dir = 'centre/stage3_nup/d*_nav/Level3_ch1-short_s3d_nav.fits'
files3_dir = 'east/stage3/d*_nav/Level3_ch1-short_s3d_nav.fits'
main_dir = 'mosaics/c1A_hires'
result_name = 'c1_short_all_interp.fits'
pow = 0.8


#=====================================================================
# Imports


import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import shutil
import scipy.interpolate
from astropy.io import fits


#=====================================================================
# Main


files1 = glob.glob(files1_dir)
files2 = glob.glob(files2_dir)
files3 = glob.glob(files3_dir)
files = np.concatenate((files1,files2,files3))

lengthf = len(files)

print('{} files detected\n'.format(lengthf))

hdul1 = fits.open(main_dir + '/interspec_1.fits')
spec1 = hdul1[0].data
hdul1.close()
hdul1a = fits.open(main_dir + '/mu_1.fits')
mu1 = hdul1a[0].data
hdul1a.close()
hdul2 = fits.open(main_dir + '/interspec_2.fits')
spec2 = hdul2[0].data
hdul2.close()
hdul2a = fits.open(main_dir + '/mu_2.fits')
mu2 = hdul2a[0].data
hdul2a.close()
hdul3 = fits.open(main_dir + '/interspec_3.fits')
spec3 = hdul3[0].data
hdul3.close()
hdul3a = fits.open(main_dir + '/mu_3.fits')
mu3 = hdul3a[0].data
hdul3a.close()
hdul4 = fits.open(main_dir + '/interspec_4.fits')
spec4 = hdul4[0].data
hdul4.close()
hdul4a = fits.open(main_dir + '/mu_4.fits')
mu4 = hdul4a[0].data
hdul4a.close()
hdul5 = fits.open(main_dir + '/interspec_5.fits')
spec5 = hdul5[0].data
hdul5.close()
hdul5a = fits.open(main_dir + '/mu_5.fits')
mu5 = hdul5a[0].data
hdul5a.close()
hdul6 = fits.open(main_dir + '/interspec_6.fits')
spec6 = hdul6[0].data
hdul6.close()
hdul6a = fits.open(main_dir + '/mu_6.fits')
mu6 = hdul6a[0].data
hdul6a.close()
hdul7 = fits.open(main_dir + '/interspec_7.fits')
spec7 = hdul7[0].data
hdul7.close()
hdul7a = fits.open(main_dir + '/mu_7.fits')
mu7 = hdul7a[0].data
hdul7a.close()
hdul8 = fits.open(main_dir + '/interspec_8.fits')
spec8 = hdul8[0].data
hdul8.close()
hdul8a = fits.open(main_dir + '/mu_8.fits')
mu8 = hdul8a[0].data
hdul8a.close()
hdul9 = fits.open(main_dir + '/interspec_9.fits')
spec9 = hdul9[0].data
hdul9.close()
hdul9a = fits.open(main_dir + '/mu_9.fits')
mu9 = hdul9a[0].data
hdul9a.close()
hdul10 = fits.open(main_dir + '/interspec_10.fits')
spec10 = hdul10[0].data
hdul10.close()
hdul10a = fits.open(main_dir + '/mu_10.fits')
mu10 = hdul10a[0].data
hdul10a.close()

"""
spec1 = spec_data_out[0]
spec2 = spec_data_out[1]
spec3 = spec_data_out[2]
spec4 = spec_data_out[3]
spec5 = spec_data_out[4]
spec6 = spec_data_out[5]
spec7 = spec_data_out[6]
spec8 = spec_data_out[7]
"""

dz1,dy1,dx1 = spec1.shape
spec_result = np.zeros((dz1,dy1,dx1))

spec1 = spec1/(mu1**pow)
spec2 = spec2/(mu2**pow)
spec3 = spec3/(mu3**pow)
spec4 = spec4/(mu4**pow)
spec5 = spec5/(mu5**pow)
spec6 = spec6/(mu6**pow)
spec7 = spec7/(mu7**pow)
spec8 = spec8/(mu8**pow)
spec9 = spec9/(mu9**pow)
spec10 = spec10/(mu10**pow)


spec_result = np.nanmedian([spec1,spec2,spec3,spec4,spec5,spec6,spec7,spec8,spec9,spec10],axis=0)


hdu4 = fits.PrimaryHDU(spec_result)
hdu4.writeto(main_dir + '/' + result_name,overwrite=True)


#=====================================================================


print('End of script\n')
