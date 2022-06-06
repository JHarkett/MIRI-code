#==========================================================================================
#Inputs

#MIRISim variables: set to be the same as the JWST data you will be collecting
#These variables do have to have values even if they are not being used

integrations = 8 #Number of integrations
groups = 4 #Number of groups
exposures = 1 #Number of exposures
mode = 'FAST' #Can be slow or fast
channels = 'BOTH' #Can be SW, LW or BOTH

dither = True #Turn off dithering by changing to False
number_dithers = 4 #Number of dithers
start_index = 33 #Start index for dither pattern

#Number of CPUs you want to use in script (multiprocessing)
#The more you use, the faster the script runs at the expense of your other
#computer processes being slower
#Set usage as none to run 1 process at a time (very slow)
#Or set as quarter, half or all to use a quarter, half or all of your computers
#CPUs

usage='none'

save0 = True

#========================================================================================
#Notes
"""
Runs mirisim module on NEMESIS data to output mimic-MIRI data

Usage:

export MIRISIM_ROOT="$HOME/mirisim"
export PYSYN_CDBS="$HOME/mirisim/cdbs"
export CDP_DIR="$MIRISIM_ROOT/CDP"
conda activate mirisim

python -W ignore runmsim_v6.py [input file]

Where [input file] can be a directory: NEMESISCubes, containing 2 fits files
for each of JWST MIRI IFUs 12 dispersers. One is the NEMESIS forward model data
cube in units of Jy/arcsec^2 and the other is a file containing the wavelengths.
These 24 images will be interpolated into a single cube before being run through
MIRISim. The input can also be the already generated .fits cube.

The code will detect what input has been provided and will either run the
function to generate the .fits cube or skip straight to the MIRISim function. 

Version 2:
	- Fixed issues that caused MIRISim to be unstable in version 1
	- Re-organised script for ease of editing

Version 3:
	- Altered MIRISim exposure time inputs to improve data quality
	- Added process to input file without editing script
	- Created function to copy all det_images into one directory named stage0
	- Included explanation at end of script describing output directories

Version 4:
	- Included multiprocessing to reduce running time of script
	- Fixed numerous issues associated with running multiprocessing
	- Added comments and instructions for ease of use

Version 5:
	- Generalised script to run on data that is not Uranus

Version 6:
	- Fixed bugs in version 5
	- Simplified script for ease of use
"""

#===========================================================================
#Configuration

#Setting up multiprocessing

import multiprocessing
multiprocessing.set_start_method('fork')
from multiprocessing import Pool
import os

if (usage == 'none'):
	if os.path.exists('stpipe-log.cfg'):
		os.remove('stpipe-log.cfg')
else:
	print('[*]',file=open('stpipe-log.cfg',"w"))
	print('handler = file:pipeline.log',file=open('stpipe-log.cfg',"a"))
	print('level = INFO',file=open('stpipe-log.cfg',"a"))

num_cores = multiprocessing.cpu_count()
if usage == 'quarter':
	maxp = num_cores // 4 or 1
elif usage == 'half':
	maxp = num_cores // 2 or 1
elif usage == 'all':
	maxp = num_cores
else:
	maxp = 1

print(''+str(maxp)+' CPUs will be used in this run\n')

#Importing mirisim modules

from mirisim.config_parser import SimConfig, SimulatorConfig, SceneConfig
from mirisim.skysim import Background, sed, Point, Galaxy, kinetics
from mirisim.skysim import wrap_pysynphot as wS
from mirisim import MiriSimulation
from mirisim.skysim import externalsources

#Importing other modules

import sys
import numpy as np
import glob
import shutil
import time
from astropy.io import fits
import astropy.units as u
import matplotlib.pyplot as plt
from matplotlib import colors,cm


#===========================================================================
#Functions

#Runs multiprocessing
def runmany(step,array):
	if __name__ == '__main__':
		p = Pool(maxp)
		res = p.map(step, array)
		p.close()
		p.join()


#Create NEMESIS cube
def build_cube(input):

	path = os.getcwd()
	os.chdir(input)

	wmin=4.88
	wmax=28.34
	dwave=0.001 # Selected to provide a round number for the MIRI dispersion.

	openarray = glob.glob('*_chan1_radJy.fits')
	opendata = openarray[0]

	hdu2_list = fits.open(opendata)
	hdu2 = hdu2_list[0]
	image2 = hdu2.data

	nz,nx,ny = image2.shape
	hdu2_list.close()

	dw=0.001

	npts=int(np.ceil((wmax-wmin)/dwave))
	print(npts)

	outcube=np.empty(shape=(npts,nx,ny))

	for ix in range(nx):
		for iy in range(ny):
			for ichan in range(1,13):

				if ichan==1:
					ConfigPath = 'MRS_1SHORT'
					disperser = 'SHORT'
					detector = 'SW'
					u_raw_fitsi = '*_chan1_radJy.fits'
					wav_fitsi = '*_chan1_wave.fits'
				if ichan==2:
					ConfigPath = 'MRS_1MEDIUM'
					disperser = 'MEDIUM'
					detector = 'SW'
					u_raw_fitsi = '*_chan2_radJy.fits'
					wav_fitsi = '*_chan2_wave.fits'
				if ichan==3:
					ConfigPath = 'MRS_1LONG'
					disperser = 'LONG'
					detector = 'SW'
					u_raw_fitsi = '*_chan3_radJy.fits'
					wav_fitsi = '*_chan3_wave.fits'
				if ichan==4:
					ConfigPath = 'MRS_2SHORT'
					disperser = 'SHORT'
					detector = 'SW'
					u_raw_fitsi = '*_chan4_radJy.fits'
					wav_fitsi = '*_chan4_wave.fits'
				if ichan==5:
					ConfigPath = 'MRS_2MEDIUM'
					disperser = 'MEDIUM'
					detector = 'SW'
					u_raw_fitsi = '*_chan5_radJy.fits'
					wav_fitsi = '*_chan5_wave.fits'
				if ichan==6:
					ConfigPath = 'MRS_2LONG'
					disperser = 'LONG'
					detector = 'SW'
					u_raw_fitsi ='*_chan6_radJy.fits'
					wav_fitsi = '*_chan6_wave.fits'
				if ichan==7:
					ConfigPath = 'MRS_3SHORT'
					disperser = 'SHORT'
					detector = 'LW'
					u_raw_fitsi = '*_chan7_radJy.fits'
					wav_fitsi = '*_chan7_wave.fits'
				if ichan==8:
					ConfigPath = 'MRS_3MEDIUM'
					disperser = 'MEDIUM'
					detector = 'LW'
					u_raw_fitsi = '*_chan8_radJy.fits'
					wav_fitsi = '*_chan8_wave.fits'
				if ichan==9:
					ConfigPath = 'MRS_3LONG'
					disperser = 'LONG'
					detector = 'LW'
					u_raw_fitsi = '*_chan9_radJy.fits'
					wav_fitsi = '*_chan9_wave.fits'
				if ichan==10:
					ConfigPath = 'MRS_4SHORT'
					disperser = 'SHORT'
					detector = 'LW'
					u_raw_fitsi = '*_chan10_radJy.fits'
					wav_fitsi = '*_chan10_wave.fits'
				if ichan==11:
					ConfigPath = 'MRS_4MEDIUM'
					disperser = 'MEDIUM'
					detector = 'LW'
					u_raw_fitsi = '*_chan11_radJy.fits'
					wav_fitsi = '*_chan11_wave.fits'
				if ichan==12:
					ConfigPath = 'MRS_4LONG'
					disperser = 'LONG'
					detector = 'LW'
					u_raw_fitsi = '*_chan12_radJy.fits'
					wav_fitsi = '*_chan12_wave.fits'
				#Finds name of the required files
				u_raw_fits_array = glob.glob(u_raw_fitsi)
				wav_fits_array = glob.glob(wav_fitsi)

				#glob returns the filenames as 1 by 1 arrays
				#Below is conversion to just a string name
				u_raw_fits = u_raw_fits_array[0]
				wav_fits = wav_fits_array[0]				

				hdu_list = fits.open(u_raw_fits)
				hdu = hdu_list[0]
				data=hdu.data
				whdu_list = fits.open(wav_fits)
				whdu = whdu_list[0]

				wavegrid=whdu.data
				wavegrid=wavegrid.flatten()
				radiance=data[:,ix,iy]

				if ichan==1:
					waveout=wavegrid
					radout=radiance

				if ichan>=2:
					waveout=np.concatenate((waveout,wavegrid))
					radout=np.concatenate((radout,radiance))

			s = np.argsort(waveout)
			waveout=waveout[s]
			radout=radout[s]

			from scipy import interpolate
			f = interpolate.interp1d(waveout,radout,fill_value='extrapolate')
			vmin=4.88
			dwave=0.001
			wavnew=np.arange(vmin, 28.34, dwave)
			radnew=f(wavnew)

			outcube[:,ix,iy]=radnew

			print(ix,iy,np.max(radnew))

	crpix = (1, 1, 1)
	crval = (10., 10., wmin)
	dwave = dw             # micron (bw / 1300 ~
	cdelt = (-0.1,0.1, dwave)

	hdu.header['CRVAL1'] = crval[0]
	hdu.header['CRPIX1'] = crpix[0]
	hdu.header['CDELT1'] = cdelt[0]
	hdu.header['CUNIT1'] = u.arcsec.to_string()
	hdu.header['CTYPE1'] = 'RA---TAN'

	hdu.header['CRVAL2'] = crval[1]
	hdu.header['CRPIX2'] = crpix[1]
	hdu.header['CDELT2'] = cdelt[1]
	hdu.header['CUNIT2'] = u.arcsec.to_string()
	hdu.header['CTYPE2'] = 'DEC--TAN'

	hdu.header['CRVAL3'] = crval[2]
	hdu.header['CRPIX3'] = crpix[2]
	hdu.header['CDELT3'] = cdelt[2]
	hdu.header['CUNIT3'] = 'um'
	hdu.header['CTYPE3'] = 'WAVE    '

	hdu.header['UNITS'] = 'uJy / arcsec2'  #u.jansky.to_string()

	hdu.data = np.multiply(outcube,1e+6)

	os.chdir(path)
	#Combines all the files into 1 and saves as MIRI_combined.fits
	run_file_name = 'MIRI_combined.fits'
	if os.path.isfile(run_file_name):
		os.remove(run_file_name)
	hdu.writeto(run_file_name)

	print('Modified FITS file created\n')
	print('Name:\n',run_file_name)

	return run_file_name


#Run MIRISim
def run_sim(array):
	newdisperser = array[0]
	data = array[1]

	input = externalsources.Skycube(data)
	scene_config = SceneConfig.makeScene(name='FITScube', loglevel=1, targets = [input])

	newConfigPath = 'MRS_1SHORT'

	if newdisperser=='MEDIUM':
		time.sleep(2)
	if newdisperser=='LONG':
		time.sleep(4)

	#Defines the simulator setup

	os.system('rm scene_{}.ini'.format(newdisperser))
	scene_config.write('scene_{}.ini'.format(newdisperser))	
	#Configuring simulation
	sim_config = SimConfig.makeSim(
		name = 'mrs_simulation_{}'.format(newdisperser),    # name given to simulation
		scene = 'scene_{}.ini'.format(newdisperser), # name of scene file to input
		rel_obsdate = 0.0,          # relative observation date (0 = launch, 1 = end of 5 yrs)
		POP = 'MRS',                # Component on which to center (Imager or MRS)
		ConfigPath = newConfigPath,  # Configure the Optical path (MRS sub-band)
		Dither = dither,             # Dither
		StartInd = start_index,               # start index for dither pattern [NOT USED HERE]
		NDither = number_dithers,                # number of dither positions [NOT USED HERE]
		DitherPat = 'mrs_recommended_dither.dat', # dither pattern to use [NOT USED HERE]
		disperser = newdisperser,        # Which disperser to use (SHORT/MEDIUM/LONG)
		detector = channels,            # Specify Channel (SW = channels 1,2, LW= channels 3,4)
		mrs_mode = mode,          # MRS read mode (default is SLOW. ~ 24s)
		mrs_exposures = exposures,          # number of exposures
		mrs_integrations = integrations,       # number of integrations
		mrs_frames = groups,             # number of groups (for MIRI, # Groups = # Frames)
		ima_exposures = 4,          # [NOT USED HERE]
		ima_integrations = integrations,       # [NOT USED HERE]
		ima_frames = groups,             # [NOT USED HERE]
		ima_mode = 'FAST',          # [NOT USED HERE]
		filter = 'F770W',          # [NOT USED HERE]
		readDetect = 'FULL'         # [NOT USED HERE]
	)

	os.system('rm MRS_simulation_{}.ini'.format(newdisperser))
	sim_config.write('MRS_simulation_{}.ini'.format(newdisperser))

	#Running simulation

	print('Running simulation for {}...\n'.format(newdisperser))

	#simulator_config = SimulatorConfig.from_default()

	simulator_config = SimulatorConfig.makeSimulator(
	take_webbPsf=False,
	add_extended=False,
	include_refpix=True,
	include_poisson=True,
	include_readnoise=False,
	include_badpix=True,
	include_dark=True,
	include_flat=True,
	include_gain=True,
	include_nonlinearity=True,
	include_drifts=True,
	include_latency=True,
	cosmic_ray_mode='SOLAR_MIN')

	mysim = MiriSimulation(sim_config,scene_config,simulator_config)
	mysim.run()

	print('Finished running: {}'.format(newdisperser))	

	os.system('rm scene_{}.ini'.format(newdisperser))
	os.system('rm MRS_simulation_{}.ini'.format(newdisperser))


#============================================================================
#Main


input = sys.argv[1]
setting = ['SHORT','MEDIUM','LONG']
substring = '.fits'
length = len(setting)


if substring in input:
	print('Combined .fits file provided. Will run MIRISim\n')
	filename = [None]*length
	array = [None]*length
	for i in range(length):
		filename[i] = input
		array[i] = [setting[i], filename[i]]
	runmany(run_sim,array)

else:
	print('NEMESIS cube provided. Will generate combined .fits cube\n')
	file = build_cube(input)
	filename = [None]*length
	array = [None]*length
	for i in range(length):
		filename[i] = file
		array[i] = [setting[i], filename[i]]
	runmany(run_sim,array)


outputdir = sorted(glob.glob('*_*_mirisim'))


if save0 == True:
	print('Saving det_images to stage0\n')

	if os.path.exists('stage0'):
		shutil.rmtree('stage0')
		os.mkdir('stage0')
	else:
		os.mkdir('stage0')

	measure = len(outputdir)
	list34 = os.listdir('{}/det_images'.format(outputdir[0]))
	measure2 = len(list34)

	path = [None]*measure
	number = measure*measure2
	file_list = [None]*number

	for i in range(len(outputdir)):
		path[i] = '{}/det_images'.format(outputdir[i])
		file_list[i] = os.listdir(path[i])
		os.chdir(path[i])
		for j in range(measure2):
			shutil.copy(file_list[i][j],'../../stage0')
		os.chdir('../..')

	list = os.listdir('stage0')
	file_num = len(list)
	print('det_images successfully saved')
	print('There are {} files in stage0\n'.format(file_num))

else:
	print('det_images not saved to stage0\n')

os.rename(outputdir[0],'SHORT_data')
os.rename(outputdir[1],'MEDIUM_data')
os.rename(outputdir[2],'LONG_data')


#===========================================================================

print('End of script\n')
