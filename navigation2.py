#=====================================================================
"""
--------Information--------

User inputs JWST observation parameters below
Outputs navigated images with longitude, latitude and emission angle for each pixel


--------Setup--------

create the conda environment: jwst_navs with all the required modules

	conda create -n jwst_navs
	conda activate jwst_navs
	pip install jwst
	pip install spiceypy
	pip install pysiaf


------Usage------

Input observation parameters below in inputs

Ensure file: jupkerns.tm and the directory: kernels are in the same
directory as this script, then:	

	conda activate jwst_navs
	python -W ignore navigation2.py [directory where data is located]

	eg:
		python -W ignore navigation2.py data

the code will navigate any fits file ending with
's3d.fits'


"""
#=====================================================================
# Inputs


body = 'JUPITER'
bodfrm = 'IAU_JUPITER'
obs = 'JWST'
inst = 'MIRI'
reffrm = 'J2000'
abcorr = 'LT' # just do light time correction for astrometric RA, Dec


#=====================================================================
# Imports


import math
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path # for point in polygon
from astropy.wcs import WCS
from astropy.io import fits
from astropy import units as u
from astropy.coordinates import SkyCoord, Distance, SpectralCoord
import pysiaf
import spiceypy as spice
from spiceypy.utils.support_types import SpiceyError
dpr = spice.dpr()
import sys
import glob


#=====================================================================
# Functions


# Convert et to utc
def et2mjd(et):
	jds = spice.et2utc(et, 'J', 8) # precision 8 gives ~1ms resolution
	jd = float(jds[2:])
	mjd = jd - 2400000.5
	return mjd


#=====================================================================
# Main


spice.furnsh('jupkerns.tm')

input_dir = sys.argv[1]
sstring = input_dir + '/*s3d.fits'

files = sorted(glob.glob(sstring))
nf = len(files)

if not os.path.exists(input_dir + '_nav'):
	os.mkdir(input_dir + '_nav')


class WcsSpiceNav:

	def __init__(self, filename, body, metakern):
		dpr = spice.dpr()
		self.body = body
		self.bodfrm = 'IAU_'+body
		hdulist = fits.open(filename)
		self.hdr0 = hdulist[0].header
		self.hdr1 = hdulist[1].header
		utcbeg = hdr0['DATE-BEG']
		utcend = hdr0['DATE-END']
		etbeg = spice.utc2et(utcbeg)
		expdur = hdr0['EFFEXPTM']
		etmid = etbeg + expdur/2
		utcmid = spice.et2utc(etmid, 'ISOC', 2)
		# **** The following line is to avoid bug in astropy WCS code ****
		self.hdr1['MJD-OBS'] = self.hdr1['MJD-AVG']

		self.wcs = WCS(self.hdr1, fobj=hdulist)
		self.obs = self.hdr0['TELESCOP']
		# print('Observer is '+self.obs)

		shape = (hdulist[1].data).shape
		self.ndim = len(shape)
		if self.ndim==3:
			# Need to get a spectral/3rd-dim "Quantity" with appropriate units
			# print('Getting spectral value for plane 0.')
			sky, spect = self.wcs.pixel_to_world(0, 0, 0)
			self.spectval = spect
		# print('image shape is (nb, ny, nx): ', self.nb, self.ny, self.nx)
		hdulist.close()

		spice.furnsh(metakern)
		# print('Converting {:s} to ET.'.format(time_obs))
		self.et = etmid
		# print('ET seconds past J2000: {:.2f}'.format(self.et))
		[self.bodpos, self.ltime] = spice.spkpos(self.body, self.et,'J2000', 'LT', self.obs)
		self.dist = spice.vnorm(self.bodpos)
		au = 1.4959787066e8 # 1 AU in km (from Allen's Astrophys. Quant.)
		self.dist_au = self.dist / au
		[self.sunpos, ltime] = spice.spkpos('SUN', self.et-self.ltime,'J2000', 'LT', body)
		self.sundist = spice.vnorm(self.sunpos)
		self.sun_au = self.sundist / au
		# print('dist, sundist: ', self.dist, self.sundist)

		# get sub-observer and sub-solar lon and lat
		spoint, trgepc, srfvec = spice.subpnt('INTERCEPT/ELLIPSOID', body, etmid, self.bodfrm, 'LT', self.obs)
		radius, lon, lat = spice.reclat(spoint)
		self.lon_subobs = lon * dpr
		self.lat_subobs = lat * dpr
		spoint, trgepc, srfvec = spice.subslr('INTERCEPT/ELLIPSOID', body, etmid, self.bodfrm, 'LT', self.obs)
		radius, lon, lat = spice.reclat(spoint)
		self.lon_subsol = lon * dpr
		self.lat_subsol = lat * dpr


		# Matrix to rotate from body-centered to J2000
		self.pform = spice.pxform('IAU_'+body, 'J2000', self.et-self.ltime)
		# print(self.pform)
		[dim, radii] = spice.bodvrd(body, 'RADII', 3)
		self.re = radii[0]
		self.rp = radii[2]
		self.f = (self.re - self.rp) / self.re

		# the following are for tweaking navigations
		self.ra_tome = 0.0 # Target Observed Minus Ephemeris RA (deg)
		self.dec_tome = 0.0 # Target Observed Minus Ephemeris Dec (deg)

		# the following are for UW spice-free navigation
		xc, yc, mu, muz, az = self.body2image(self.lon_subobs, self.lat_subobs)
		self.xcent = xc
		self.ycent = yc
		xpole, ypole, mu, muz, az = self.body2image(0.0, 90.0) # pole
		pole_cw = (360 - (math.atan2(-(xpole-xc),ypole-yc) * dpr)) % 360 # deg CW from up
		self.mpole = pole_cw
		cna = math.cos(pole_cw/dpr)
		sna = math.sin(pole_cw/dpr)
		m0to1 = [[cna, -sna, 0], [sna, cna, 0], [0,   0,   1]]
		calf = math.cos(math.pi/2 - self.lat_subobs/dpr)
		salf = math.sin(math.pi/2 - self.lat_subobs/dpr)
		m1to2 = [[1,  0,  0], [0, calf, -salf], [0, salf, calf]]
		m0to2 = np.matmul(m1to2, m0to1)
		m2to3 = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
		m0to3 = np.matmul(m2to3, m0to2)
		self.ps = math.sqrt(hdr1['PIXAR_A2']) # pixel scale arcsec/pixel

	def image2body(self, x, y):
		if self.ndim==3:
			sky, spect = self.wcs.pixel_to_world(x, y, 0)
		else:
			sky = self.wcs.pixel_to_world(x, y)
		ra = sky.ra.deg - self.ra_tome # offset from observation WCS to ephemeris
		dec = sky.dec.deg - self.dec_tome
		rect = spice.radrec(1.0, ra/dpr, dec/dpr)
		try:
			[point, int_et, obs_vec] = spice.sincpt('Ellipsoid', self.body, self.et, 'IAU_'+self.body, 'LT', self.obs, 'J2000', rect)
		except SpiceyError as exc: # assume exceptions are for intercept point
		# not on planet
			return [-999.9, -99.9, -99.9, -9.99, -9.99, 999.9, -999.9]
		[radius, lon, lat] = spice.reclat(point)
		[pglon, pglat, alt] = spice.recpgr(self.body, point, self.re, self.f)
		epoch, otvec, phase, incid, emiss, vis, illumd = spice.illumf('ELLIPSOID', self.body, 'SUN', self.et, 'IAU_'+self.body, 'LT', self.obs, point)
		cos_g = np.cos(phase)
		cos_i = np.cos(incid)
		cos_e = np.cos(emiss)
		sin_i = np.sin(incid)
		sin_e = np.sin(emiss)
		mu = cos_e
		muz = cos_i
		cos_az = (cos_g - cos_i * cos_e) / (sin_i * sin_e)
		az = 180.0 - np.arccos(cos_az) * dpr

		return lon*dpr, lat*dpr, pglat*dpr, mu, muz, az, pglon*dpr
    
	def body2image(self, elon, latc):
		surfpoint = spice.latsrf('ELLIPSOID', self.body, self.et-self.ltime,'IAU_'+self.body, [[elon/dpr, latc/dpr]])
		surfpoint_j2000 = spice.mxv(self.pform, surfpoint[0])
		# print('surfpoint_j2000: ', surfpoint_j2000)
		vec_samp = spice.mxv(self.pform, surfpoint[0]) + self.bodpos
		(dist, ra, dec) = spice.recrad(vec_samp)
		ra = ra + self.ra_tome # offset from ephemeris to observation WCS
		dec = dec + self.dec_tome
		sky = SkyCoord(ra*dpr*u.deg, dec*dpr*u.deg)
		# print(sky)
		if self.ndim==3:
			[x, y, z] = self.wcs.world_to_pixel(sky, self.spectval)
		else:
			[x, y] = self.wcs.world_to_pixel(sky)
		# Get illumination and viewing angles for this sample point.
		epoch, otvec, phase, incid, emiss, vis, illumd = spice.illumf('ELLIPSOID', self.body, 'SUN', self.et, 'IAU_'+self.body, 'LT', self.obs, surfpoint[0])
		cos_g = np.cos(phase)
		cos_i = np.cos(incid)
		cos_e = np.cos(emiss)
		sin_i = np.sin(incid)
		sin_e = np.sin(emiss)
		mu = cos_e
		muz = cos_i
		cos_az = (cos_g - cos_i * cos_e) / (sin_i * sin_e)
		az = 180.0 - np.arccos(cos_az) * dpr

		return x.item(), y.item(), mu, muz, az

	def imtopla(self, xim, yim): # UW spice-free version of image2body
		x1 = math.cos(self.mpole/dpr) * (xim - self.xcent) - math.sin(self.mpole/dpr) * (yim - self.ycent)
		y1 = math.sin(self.mpole/dpr) * (xim - self.xcent) + math.cos(self.mpole/dpr) * (yim - self.ycent)
		q = self.ps / 3600 / dpr
		ra = self.dist
		olat = self.lat_subobs
		olon = self.lon_subobs
		slat = self.lat_subsol
		slon = self.lon_subsol
		# print('olat, olon, slat, slon: ', olat, olon, slat, slon)
		re = self.re
		rp = self.rp
		a = 1 + (re/rp)**2 * math.tan(olat/dpr)**2
		b = 2 * ra * q * y1 * (re/rp)**2 * math.sin(olat/dpr) / math.cos(olat/dpr)**2
		c = (x1*ra*q)**2 + (re/rp)**2 * (ra*q*y1)**2 / math.cos(olat/dpr)**2 - re**2
		# print('a, b, c: ', a, b, c)
		radical = b**2 - 4 * a * c
		# print('radical: ', radical)
		if radical < 0:
			return -999.9, -99.9, -99.9, -9.99, -9.99, 999.9
		x3s1 = (-b + math.sqrt(radical)) / (2 * a)
		x3s2 = (-b - math.sqrt(radical)) / (2 * a)
		# print('x3s1, x3s2: ', x3s1, x3s2)
		z3s1 = (ra*q*y1 + x3s1 * math.sin(olat/dpr)) / math.cos(olat/dpr)
		z3s2 = (ra*q*y1 + x3s2 * math.sin(olat/dpr)) / math.cos(olat/dpr)
		# print('z3s1, z3s2: ', z3s1, z3s2)
		odotr1 = x3s1 * math.cos(olat/dpr) + z3s1 * math.sin(olat/dpr)
		odotr2 = x3s2 * math.cos(olat/dpr) + z3s2 * math.sin(olat/dpr)
		# print('odotr1, odotr2: ', odotr1, odotr2)
		if odotr1 > 0:
			x3 = x3s1
			z3 = z3s1
			odotr = odotr1
		else:
			x3 = x3s2
			z3 = z3s2
			odotr = odotr2
		y3 = x1 * ra * q
		lone = math.atan2(y3, x3) * dpr + olon
		r = math.sqrt(x3**2 + y3**2 + z3**2)
		latc = math.asin(z3/r) * dpr
		latg = math.atan((re**2/rp**2) * math.tan(latc/dpr)) * dpr
		norm = np.array([math.cos(latg/dpr) * math.cos((lone-olon)/dpr), math.cos(latg/dpr) * math.sin((lone-olon)/dpr), math.sin(latg/dpr)])
		obs = np.array([math.cos(olat/dpr), 0, math.sin(olat/dpr)])
		sun = np.array([math.cos(slat/dpr) * math.cos((slon-olon)/dpr), math.cos(slat/dpr) * math.sin((slon-olon)/dpr), math.sin(slat/dpr)])
		mu = np.matmul(norm, np.transpose(obs)) # surface normal dot vector to obs
		muz = np.matmul(norm, np.transpose(sun)) # surface normal dot vector to sun
		snx = np.cross(-1*sun, norm)
		onx = np.cross(obs, norm)
		snxmag = math.sqrt(np.vdot(snx, snx))
		onxmag = math.sqrt(np.vdot(onx, onx))
		az = math.acos(np.vdot(snx, onx) / (snxmag * onxmag)) * dpr
		return lone, latc, latg, mu, muz, az


etbegs = np.zeros(nf)
etmids = np.zeros(nf)
utcmids = []
navs = []
for i,f in enumerate(files):
		print('file:', f)
		hdulist = fits.open(f)
		hdr0 = hdulist[0].header
		hdr1 = hdulist[1].header
		utcbeg = hdr0['DATE-BEG']
		etbeg = spice.utc2et(utcbeg)
		expdur = hdr0['EFFEXPTM']
		etmid = etbeg + expdur/2
		utcmid = spice.et2utc(etmid, 'ISOC', 2)
		etbegs[i] = etbeg
		etmids[i] = etmid
		utcmids.append(utcmid)
		(pos, lt) = spice.spkpos(body, etmids[i], reffrm, abcorr, obs)
		(dist, ra, dec) = spice.recrad(pos) # dist in km
		ra_bod = ra * dpr
		dec_bod = dec * dpr
		ra_hdr = hdr0['TARG_RA']
		dec_hdr = hdr0['TARG_DEC']
		ra_diff = ra_hdr - ra_bod
		dec_diff = dec_hdr - dec_hdr
		#print('Header-computed RA, Dec (arcsec): {:.4f} {:.4f}'.format(ra_diff*3600, dec_diff*3600))
		#print('No pointing tweak applied at this time')
		nav = WcsSpiceNav(f, body, 'jupkerns.tm')
		navs.append(nav)
		hdulist.close()



# This is just code to test transforms
slat, slon = nav.lat_subsol, nav.lon_subsol
slon, slat = -80.0, -30.0
x, y, mu, muz, az = nav.body2image(slon, slat)
lone, latc, latg, mu, muz, az, pglon = nav.image2body(x, y)
#print(slat, slon)
#print(x, y)
#print(latc, lone)
lone2, latc2, latg2, mu2, muz2, az2 = nav.imtopla(x, y)
#print(latc2, lone2)
#print('mu1, muz1, az1:', mu, muz, az)
#print('mu2, muz2, az2:', mu2, muz2, az2)


nav.__dict__.keys()


for i,f in enumerate(files):
	hdul = fits.open(f)
	hdr0 = hdul[0].header
	hdr1 = hdul[1].header
	cube = hdul[1].data # in MJy/sr
	hdul.close()
	nav = navs[i] # already created
    
	dims = len(cube.shape)
	if dims == 2:
		ny, nx = cube.shape
	elif dims==3:
		nb, ny, nx = cube.shape
	print('nx, ny, nb: ', nx, ny, nb)
	# The following header keywords are for UW non-spice navigation code
	xc, yc, mu, muz, az = nav.body2image(nav.lon_subobs, nav.lat_subobs)
	hdr1['XCENT'] = xc, 'Target center, 0-based'
	hdr1['YCENT'] = yc
	xpole, ypole, mu, muz, az = nav.body2image(0.0, 90.0) # pole
	pole_cw = (360 - (math.atan2(-(xpole-xc),ypole-yc) * dpr)) % 360 # deg CW from up
	hdr1['MPOLE'] = pole_cw, 'Pole angle in image, CW from up (deg)'
	hdr1['RE_KM'] = nav.re
	hdr1['RP_KM'] = nav.rp
	hdr1['RANGE'] = nav.dist, 'Observer-Target (km)'
	hdr1['SUNDIST'] = nav.sundist, 'Sun-Target (km)'
	hdr1['SUN_AU'] = nav.sun_au, 'Sun-Target (AU)'
	hdr1['SO_LAT'] = nav.lat_subobs, 'Subobs lat, centric deg'
	hdr1['SO_LON'] = nav.lon_subobs, 'Subobs lon, east deg'
	hdr1['SS_LAT'] = nav.lat_subsol, 'Subsol lat, centric deg'
	hdr1['SS_LON'] = nav.lon_subsol, 'Subsol lon, east deg'
	ps = math.sqrt(hdr1['PIXAR_A2'])
	hdr1['PSCALE1'] = ps, 'arcseconds / pixel' # assume square
	re_pix = nav.re / nav.dist / (ps / 3600 / dpr)

	lat_plane = np.zeros((ny,nx)) # planetocentric latitude
	lon_plane = np.zeros((ny,nx)) # east longitude
	pglat_plane = np.zeros((ny,nx))
	pglon_plane = np.zeros((ny,nx))
	mu_plane = np.zeros((ny,nx)) - 9.99 # cos emission, set to invalid
	muz_plane = np.zeros((ny,nx)) # cos incidence
	az_plane = np.zeros((ny,nx)) # 180 - (angle between projections of i and e onto surface normal)
	for y in range(ny):
		if y % 100 == 0:
			print('line ', y)
		for x in range(nx):
			if math.sqrt((y-yc)**2 + (x-xc)**2) < re_pix*1.05:
				lon, lat, pglat, mu, muz, az, pglon = nav.image2body(x, y)
				if mu>=-1 and mu<=1: # How to test for pixel on planet disk.
					lon_plane[y,x] = lon
					lat_plane[y,x] = lat
					pglat_plane[y,x] = pglat
					pglon_plane[y,x] = pglon
					mu_plane[y,x] = mu
					muz_plane[y,x] = muz
					az_plane[y,x] = az

	primary_hdu = fits.PrimaryHDU(header=hdr0)
	image_hdu = fits.ImageHDU(cube, header=hdr1, name='SCI')
	lat_hdu = fits.ImageHDU(lat_plane, name='LAT_CENT')
	lon_hdu = fits.ImageHDU(lon_plane, name='LON_EAST')

	lat_pgr_hdu = fits.ImageHDU(pglat_plane, name='LAT_PGR')
	lon_west_hdu = fits.ImageHDU(pglon_plane, name='LON_WEST')

	mu_hdu = fits.ImageHDU(mu_plane, name='MU')
	muz_hdu = fits.ImageHDU(muz_plane, name='MU_ZERO')
	az_hdu = fits.ImageHDU(az_plane, name='AZIMUTH')
	hduout = fits.HDUList([primary_hdu, image_hdu, lat_hdu, lat_pgr_hdu, lon_hdu, lon_west_hdu,  mu_hdu, muz_hdu, az_hdu])
	base, ext = f.split('.')
	base2 = base.replace(input_dir + '/','')

	outname = input_dir + '_nav/' + base2 +'_nav.'+ext
	hduout.writeto(outname, overwrite=True)



#=====================================================================


print('End of script')
