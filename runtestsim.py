#==========================================================================
#Inputs

#Set to be the same as runmsim_v5.py

Disperser = 'SHORT'
configpath = 'MRS_1SHORT'
Detecter = 'SW' #Can be SW, LW or BOTH
mode = 'FAST'
exposures = 1
dither = False
startind = 1 #start index for dither pattern
number_dither = 4
integrations = 3
ngroups = 5


#=========================================================================
#Information
"""
Runs mirisim module on simplified scene to generate required cache files
in CDP directory

Usage:

export MIRISIM_ROOT="$HOME/mirisim"
export PYSYN_CDBS="$HOME/mirisim/cdbs"
export CDP_DIR="$MIRISIM_ROOT/CDP"
conda activate mirisim

python -W ignore runtestsim.py

"""

#===========================================================================
#Config

#Import mirisim modules

from mirisim.config_parser import SimConfig, SimulatorConfig, SceneConfig
from mirisim.skysim import Background, sed, Point, Galaxy, kinetics
from mirisim.skysim import wrap_pysynphot as wS
from mirisim import MiriSimulation

#Import other modules

import numpy as np
import glob
import os
from astropy.io import fits
import matplotlib.pyplot as plt
from matplotlib import colors,cm


#============================================================================
#Scene


point1 = Point(Cen = (-0.5,0.5))
Blackbody = sed.BBSed(Temp = 800., wref = 10., flux = 1e5)
point1.set_SED(Blackbody)

point2 = Point(Cen = (0.5,0.5))
PYSPsedDict = {'family':'bkmodels','sedname':'bk_b0005','flux':1.e5, 'wref':10.}
sedP2 = wS.PYSPSed(**PYSPsedDict)
point2.set_SED(sedP2)

galaxy = Galaxy(Cen = (0.,-0.5),n=1.,re=1.,q=0.1,pa=90)
PowerLaw = sed.PLSed(alpha=1.0,flux=5e5,wref=10.)
galaxy.set_SED(PowerLaw)
VMAPpars = {'vrot': 200., 'Cen': (0., -0.5), 'pa': 90., 'q': 0.1, 'c': 0}
VelocityMap = kinetics.FlatDisk(**VMAPpars)
galaxy.set_velomap(VelocityMap)
losVeloDist = kinetics.Losvd(sigma=200.,h3=0.,h4=0.)
galaxy.set_LOSVD(losVeloDist)

bg = Background(level='low',gradient=5., pa=45.)

scene = bg + point1 + point2 + galaxy

#============================================================================
#Saving data

#Saves scene as a .ini file
#Do not comment this
scene_config = SceneConfig.makeScene(loglevel=0,background=bg,targets = [point1,point2,galaxy])
os.system('rm MRS_example_scene.ini')
scene_config.write('MRS_example_scene.ini')


#Saves scene as a .fits file
#Can uncomment this if a fits file is required
#FOV = np.array([[-4.,4.],[-4.,4.]])   # field of view [xmin,xmax],[ymin,ymax] (in arcsec)
#SpatialSampling = 0.1               # spatial sampling (in arcsec)
#WavelengthRange = [5,15]            # wavelength range to process (in microns)
#WavelengthSampling = 0.05 

#scene.writecube(cubefits = 'MRS_example_scene.fits',FOV = FOV, time = 0.0,spatsampling = SpatialSampling,wrange = WavelengthRange,wsampling = WavelengthSampling,overwrite = True)  


#============================================================================
#Configuring simulation

sim_config = SimConfig.makeSim(
    name = 'mrs_simulation',    # name given to simulation
    scene = 'MRS_example_scene.ini', # name of scene file to input
    rel_obsdate = 0.0,          # relative observation date (0 = launch, 1 = end of 5 yrs)
    POP = 'MRS',                # Component on which to center (Imager or MRS)
    ConfigPath = configpath,  # Configure the Optical path (MRS sub-band)
    Dither = dither,             # Don't Dither
    StartInd = startind,               # start index for dither pattern [NOT USED HERE]
    NDither = number_dither,                # number of dither positions [NOT USED HERE]
    DitherPat = 'mrs_recommended_dither.dat', # dither pattern to use [NOT USED HERE]
    disperser = Disperser,        # Which disperser to use (SHORT/MEDIUM/LONG)
    detector = Detecter,            # Specify Channel (SW = channels 1,2, LW= channels 3,4)
    mrs_mode = mode,          # MRS read mode (default is SLOW. ~ 24s)
    mrs_exposures = exposures,          # number of exposures
    mrs_integrations = integrations,       # number of integrations
    mrs_frames = ngroups,             # number of groups (for MIRI, # Groups = # Frames)
    ima_exposures = 0,          # [NOT USED HERE]
    ima_integrations = 0,       # [NOT USED HERE]
    ima_frames = 0,             # [NOT USED HERE]
    ima_mode = 'FAST',          # [NOT USED HERE]
    filter = 'F1130W',          # [NOT USED HERE]
    readDetect = 'FULL'         # [NOT USED HERE]
)

os.system('rm MRS_simulation.ini')
sim_config.write('MRS_simulation.ini')


#============================================================================
#Running simulation

print('Running simulation...\n')
simulator_config = SimulatorConfig.from_default()
mysim = MiriSimulation(sim_config,scene_config,simulator_config)
mysim.run()
print('Finished simulation\n')


#===========================================================================

print('End of script\n')
