#====================================================================================
#Inputs


# Set degree of multiprocessing
usage='half'

# Set to true to run residual fringe step between stages 2 and 3 (takes 30 mins)
run_fringe = True

# Number of available dither positions
n_dither = 4


#====================================================================================
#Information

"""
------------Information------------

Script to process Extended-source MIRI MRS IFU data via stage 1, 2 and 3
Does not combine dithers

------------Pre-pipeline setup------------

It is advised to install and run the JWST pipeline module within an environment, to create a new environment type
	conda create -n [name] python

To activate this environment, type
	conda activate [name]

Install the latest version of the jwst python module (1.8.3 as of 02/11/22) (also installs several other packages):
	pip install --upgrade jwst

This will take 5 mins to run.

Upgrade the packages; gwcs, asdf, specutils, imexam to the latest version:
	pip install --upgrade [package name]

Install latest version of photutils
	pip install --upgrade photutils


------------Usage------------


	line 155: (optional) change the settings for the 3 stages

	set inputs above

	conda activate jwst
	export CRDS_PATH="$HOME/crds_cache"
	export CRDS_SERVER_URL="https://jwst-crds.stsci.edu"

	python -W ignore jwstpipeline_singledither_v2.py [directory]

		(directory = location of stage0 data)


"""

#==========================================================================
#Package configuration

#Set up multiprocessing

import multiprocessing
multiprocessing.set_start_method('fork')
from multiprocessing import Pool
import os
import sys

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

#Import other useful packages

import glob, sys, os, time, shutil, warnings
from astropy.io import fits
from astropy.io import ascii
from astropy.visualization import (LinearStretch, LogStretch, ImageNormalize, ZScaleInterval)
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc

import jwst
if jwst.__version__ != '1.8.0':
	warnings.warn(f'You are running version {jwst.__version__} of the jwst module\n the most recent is version 1.7.2 (As of 05/10/22)\n consider upgrading\n')

from jwst.pipeline import Detector1Pipeline
from jwst.pipeline import Spec2Pipeline
from jwst.pipeline import Spec3Pipeline

from jwst.residual_fringe import ResidualFringeStep
from jwst import datamodels
from jwst.associations import asn_from_list as afl
from jwst.associations.lib.rules_level2_base import DMSLevel2bBase
from jwst.associations.lib.rules_level3_base import DMS_Level3_Base


from jwst.datamodels import dqflags


#=========================================================================
#Functions


#Defines and creates necessary directories
def directory_check(directory1,path):
	os.chdir(directory1)
	os.chdir('..')
	det1_directory = 'stage1'
	spec2_directory = 'stage2'
	spec3_directory = 'stage3'
	if not os.path.exists(det1_directory):
		os.makedirs(det1_directory)
		print('Created stage 1 directory\n')
	if not os.path.exists(spec2_directory):
		os.makedirs(spec2_directory)
		print('Created stage 2 directory\n')
	if not os.path.exists(spec3_directory):
		os.makedirs(spec3_directory)
		print('Created stage 3 directory\n')


	os.chdir(path)
	return det1_directory, spec2_directory, spec3_directory


#Runs multiprocessing
def runmany(step,filenames):
	if __name__ == '__main__':
		p = Pool(maxp)
		res = p.map(step, filenames)
		p.close()
		p.join()


#Stage 1 pipeline
def rundet1(filenames):
	det1 = Detector1Pipeline()
	det1.output_dir = '../stage1'
	det1.dq_init.skip = False
	det1.saturation.skip = False
	det1.superbias.skip = False
	det1.refpix.skip = False
	det1.linearity.skip = False
	det1.persistence.skip = False
	det1.dark_current.skip = False
	det1.jump.skip = False
	det1.save_results = True
	det1(filenames)


#Stage 2 pipeline
def runspec2(filename):
	spec2 = Spec2Pipeline()
	spec2.output_dir = '../stage2'
	spec2.assign_wcs.skip = False
	spec2.bkg_subtract.skip = True
	spec2.flat_field.skip = False
	spec2.srctype.skip = False
	spec2.straylight.skip = False
	spec2.fringe.skip = False
	spec2.photom.skip = False
	spec2.cube_build.skip = True
	spec2.extract_1d.skip = True
	spec2.save_results = True
	spec2(filename)


#Write a Lv13 association file from an input list (3a)
def writel3asn(files,asnfile,prodname,**kwargs):
	asn = afl.asn_from_list(files,rule=DMS_Level3_Base,product_name=prodname)
	if ('bg' in kwargs):
		for bgfile in kwargs['bg']:
			asn['products'][0]['members'].append({'expname': bgfile, 'exptype':'background'})
	_, serialized = asn.dump()
	with open(asnfile, 'w') as outfile:
		outfile.write(serialized)		


#Splits stage 2 _cal.fits files into their bands
def sort_calfiles(d, files):
	nfiles = len(files)
	channel = []
	band = []
	dither = []
	for file in files:
		hdr = (fits.open(file))[0].header
		channel.append(hdr['CHANNEL'])
		band.append(hdr['BAND'])
		dither.append(hdr['PATT_NUM'])
	channel = np.array(channel)
	band = np.array(band)
	dither = np.array(dither)

	indx = np.where((channel == '12')&(band == 'SHORT')&(dither == int(d)))
	files12A = files[indx]

	indx = np.where((channel == '12')&(band == 'MEDIUM')&(dither == int(d)))
	files12B = files[indx]

	indx = np.where((channel == '12')&(band == 'LONG')&(dither == int(d)))
	files12C = files[indx]

	indx = np.where((channel == '34')&(band == 'SHORT')&(dither == int(d)))
	files34A = files[indx]

	indx = np.where((channel == '34')&(band == 'MEDIUM')&(dither == int(d)))
	files34B = files[indx]

	indx = np.where((channel == '34')&(band == 'LONG')&(dither == int(d)))
	files34C = files[indx]

	return files12A,files12B,files12C,files34A,files34B,files34C


# Define the residual fringe step
def runrf(filename):
	print(filename)
	rf1 = ResidualFringeStep() # Instantiate the pipeline

	rf1.save_results = True
	rf1.output_dir = spec2_dir
	rf1(filename) # Run the pipeline on an input list of files


#Stage 3 pipeline
def runspec3(filename):
	print(d)
	crds_config = Spec3Pipeline.get_config_from_reference('l3asn-12A.json')
	spec3 = Spec3Pipeline.from_config_section(crds_config)
	spec3.output_dir = '../stage3/d{}'.format(d)
	spec3.save_results = True
	spec3.assign_mtwcs.skip = False
	spec3.master_background.skip = True
	spec3.outlier_detection.skip = False
	spec3.mrs_imatch.skip = False
	spec3.cube_build.skip = False
	spec3.extract_1d.skip = True
	spec3.cube_build.output_type = 'band'
	spec3.cube_build.coord_system = 'ifualign'
	spec3(filename)


#==========================================================================
#Main


#Locating/creating directories
mirisim_dir = sys.argv[1]
retval = os.getcwd()
det1_dir, spec2_dir, spec3_dir = directory_check(mirisim_dir,retval)


#stage 1
print('Running stage 1 pipeline...\n')
os.chdir(mirisim_dir)
sstring = 'det*exp1.fits'
simfiles = sorted(glob.glob(sstring))
print('Found ' + str(len(simfiles)) + ' input files to process\n')
runmany(rundet1,simfiles)
os.chdir(retval)
print('Finished stage 1 pipeline\n')


#Stage 2
print('Running stage 2 pipeline...\n')
os.chdir(det1_dir)
sstring = 'det*rate.fits'
ratefiles = sorted(glob.glob(sstring))
print('Found ' + str(len(ratefiles)) + ' input files to process\n')

for ii in range(0, len(ratefiles)):
	hdu = fits.open(ratefiles[ii])
	hdu[1].header['SRCTYPE']='EXTENDED'
	hdu.writeto(ratefiles[ii],overwrite=True)
	hdu.close()

runmany(runspec2,ratefiles)
os.chdir(retval)
print('Finished stage 2 pipeline\n')


if run_fringe == True:
	print('Running Residual fringe step\n')
	calfiles = sorted(glob.glob(spec2_dir + '/' + '*cal.fits'))

	nfile = len(calfiles)
	print('Found ' +str(nfile) + ' files in first step\n')

	for ii in range(0,nfile):
		runrf(calfiles[ii])
	print('Finished Residual fringe step\n')


#Stage 3
for kk in range(n_dither):
	print('Calibrating dither: {}'.format(kk+1))
	d = kk+1

	if not os.path.exists(spec3_dir + '/d{}'.format(d)):
		os.makedirs(spec3_dir + '/d{}'.format(d))

	print('Running stage 3 pipeline\n')
	os.chdir(spec2_dir)

	if run_fringe == True:
		sstring = 'det*residual_fringe.fits'
	else:
		sstring = 'det*cal.fits'

	calfiles = np.array(sorted(glob.glob(sstring)))
	sortfiles = sort_calfiles(d, calfiles)

	print('Found ' + str(len(calfiles)) + ' input files to process\n')

	asnlist = []
	names=['12A','12B','12C','34A','34B','34C']
	for ii in range(0,len(sortfiles)):
		thesefiles = sortfiles[ii]
		ninband = len(thesefiles)
		if (ninband > 0):
			filename = 'l3asn-' + names[ii] + '.json'
			asnlist.append(filename)
			writel3asn(thesefiles,filename,'Level3')

	runmany(runspec3,asnlist)
	os.chdir(retval)
	print('Finished stage 3 pipeline\n')


#===========================================================================


print('End of script\n')
