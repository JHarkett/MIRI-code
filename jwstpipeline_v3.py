#====================================================================================
#Inputs

#Set degree of multiprocessing
usage='half'

#====================================================================================
#Information

"""
Script to process Extended-source MIRI MRS IFU data via stage 1, 2 and 3

Note photutils 1.2.0 is incompatable with jwst 1.3.3
Use photutils 1.1.0 instead

Usage:

conda activate astroconda
export CRDS_PATH="$HOME/crds_cache"
export CRDS_SERVER_URL="https://jwst-crds.stsci.edu"

python -W ignore jwstpipeline_v3.py [directory]

	(directory = location of stage0 data)

Version 2:
	- Changed name of script for ease of use
	- Re-organised script to enable editing
	- Created function to input stage0 directory on command line
	- Updated code to use latest version of jwst module (1.3.3 as of 10/11/21)
	- Included background subtraction in stage 2
	- Stage 3 master background included, will be skipped if stage 2 background
	  is applied

Version 3:
	- Updated script to process extended-source data

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
if jwst.__version__ != '1.3.3':
	warnings.warn(f'You are running version {jwst.__version__} of the jwst module\n the most recent is version 1.3.3 (As of 19/10/21)\n consider upgrading\n')

from jwst.pipeline import Detector1Pipeline
from jwst.pipeline import Spec2Pipeline
from jwst.pipeline import Spec3Pipeline

from jwst import datamodels
import jwst.ramp_fitting.utils as utils
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
	det1.refpix.skip = True
	det1.save_results = True
	det1(filenames)


#Stage 2 pipeline
def runspec2(filename):
	spec2 = Spec2Pipeline()
	spec2.output_dir = '../stage2'
	spec2.assign_wcs.skip = False
	spec2.bkg_subtract.skip = False
	spec2.flat_field.skip = False
	spec2.srctype.skip = False
	spec2.straylight.skip = True
	spec2.fringe.skip = False
	spec2.photom.skip = False
	spec2.cube_build.skip = False
	spec2.extract_1d.skip = False
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
def sort_calfiles(files):
	nfiles = len(files)
	channel = []
	band = []
	for file in files:
		hdr = (fits.open(file))[0].header
		channel.append(hdr['CHANNEL'])
		band.append(hdr['BAND'])
	channel = np.array(channel)
	band = np.array(band)
	
	indx = np.where((channel == '12')&(band == 'SHORT'))
	files12A = files[indx]
	indx = np.where((channel == '12')&(band == 'MEDIUM'))
	files12B = files[indx]
	indx = np.where((channel == '12')&(band == 'LONG'))
	files12C = files[indx]
	indx = np.where((channel == '34')&(band == 'SHORT'))
	files34A = files[indx]
	indx = np.where((channel == '34')&(band == 'MEDIUM'))
	files34B = files[indx]
	indx = np.where((channel == '34')&(band == 'LONG'))
	files34C = files[indx]
	return files12A,files12B,files12C,files34A,files34B,files34C


#Stage 3a pipeline
def runspec3(filename):
	crds_config = Spec3Pipeline.get_config_from_reference('l3asn-12A.json')
	spec3 = Spec3Pipeline.from_config_section(crds_config)
	spec3.output_dir = '../stage3'
	spec3.save_results = True
	spec3.master_background.skip = False
	spec3.outlier_detection.skip = False
	spec3.mrs_imatch.skip = False
	spec3.cube_build.skip = False
	spec3.extract_1d.skip = False
	spec3(filename)


#Uber-cube pipeline
def runspec3_all(filename):
	crds_config = Spec3Pipeline.get_config_from_reference('l3asn-12A.json')
	spec3 = Spec3Pipeline.from_config_section(crds_config)
	spec3.output_dir = '../stage3'
	spec3.save_results = True
	spec3.master_background.skip = False
	spec3.outlier_detection.skip = False
	spec3.mrs_imatch.skip = False
	spec3.cube_build.skip = False
	spec3.extract_1d.skip = False
	spec3.cube_build.output_file = 'allcube'
	spec3.cube_build.output_type = 'multi'
	spec3(filename)


#==========================================================================
#Main


#Locating/creating directories
mirisim_dir = sys.argv[1]
retval = os.getcwd()
det1_dir, spec2_dir, spec3_dir = directory_check(mirisim_dir,retval)

"""
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


#Stage 3a
print('Running stage 3a pipeline\n')
os.chdir(spec2_dir)
sstring = 'det*cal.fits'
calfiles = np.array(sorted(glob.glob(sstring)))
sortfiles = sort_calfiles(calfiles)
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
print('Finished stage 3a pipeline\n')
"""

#Stage 3c pipeline
print('Running stage 3c pipeline...\n')
os.chdir(spec2_dir)
sstring = 'det*cal.fits'
calfiles = np.array(sorted(glob.glob(sstring)))
writel3asn(calfiles,'l3asn.json','Level3')
print('Found ' + str(len(calfiles)) + ' input files to process\n')
runspec3_all('l3asn.json')
os.chdir(retval)
print('End of stage 3c pipeline\n')



#===========================================================================


print('End of script\n')
