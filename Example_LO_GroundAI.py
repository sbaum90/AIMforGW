############################################################################
# this example script runs the AIMforGW.ParamEstimator_LO_GroundAI code 
# for a single point and stores the output as a pickle object
############################################################################

######################################################
# load python packages
import pickle
import numpy as np
import time
import os
import sys
import healpy as hp
import multiprocessing as mp

######################################################
# load parameter estimation code
import ParamEstimator_LO_GroundAI as AI_PE

######################################################
# set parameters

#######################
# multiprocessing 
Ncores = 4 # number of cores used for paralellized computation

#######################
# GW source
source_Mc = 25. # detector-frame chirp mass [Msol]])
source_dL = 100. # luminosity distance [Mpc]
source_phi0 = 0. # phase of the GW signal at fGWmin
source_tc = 0. # time of merger, measured from the solar equinox [seconds]
source_iota = np.pi/4 # angle between sky localization and orbital momentum of the binary [rad]
source_psi = np.pi/3. # polarization angle [rad]
source_RA = np.pi*60.0/180 # source right ascension [rad]
source_DEC = np.pi*(30.0-23.4)/180 # source declination [rad]

#######################
# detector
# inputs for detector
detector_fGWmin = 0.45 # smallest frequency considered [Hz]
detector_fGWmax = 12. # smallest frequency considered [Hz]
detector_t0 = 0. # reference time at which to define the location of the detector [seconds after vernal equinox]
#  detector network: Homestake+Zaoshan+Renstroem
detector_RA0 = np.pi/180.*np.array([256.2, 113.0, 20.1]) # right ascension of detector at t0 [rad]. For a detector network, put these as np.arrays with one entry for each detector
detector_DEC0 = np.pi/180.*np.array([44.4, 28.0, 64.9]) # declination of detector at t0 [rad]. For a detector network, put these as np.arrays with one entry for each detector

#######################
# set up path to store results
timestamp_str = time.strftime("%Y%m%d_%H%M")
fpath_out = "ExampleOutput/results_Example_LO_GroundAI_"+timestamp_str+"/"

######################################################
# run
######################################################

#######################
# print a message to std out
print("###########################################")
print("starting to run")
print("results will be saved in folder '"+fpath_out+"'")
print("###########################################")

#######################
# start timer
test_t0 = time.time()
# create output directory
os.mkdir(fpath_out)

#######################
# write info file
fo = open(fpath_out+'/info.txt', 'w')
fo.write('# RUNNING WITH AIMforGW.ParamEstimator_LO_GroundAI\n')

fo.write('# multiprocessing setup\n')
fo.write(str(Ncores)+' # Ncores\n')

fo.write('# inputs for GW source\n')
fo.write(str(source_Mc)+' # detector-frame chirp mass [Msol]]\n')
fo.write(str(source_dL)+' # source_dL, luminosity distance [Mpc]\n')
fo.write(str(source_phi0)+' # source_phi0, phase of the GW signal at fGWmin\n')
fo.write(str(source_tc)+' # source_tc, time of merger, measured from the solar equinox [seconds]\n')
fo.write(str(source_iota)+' # source_iota, angle between sky localization and orbital momentum of the binary [rad]\n')
fo.write(str(source_psi)+' # source_psi, polarization angle [rad]\n')
fo.write(str(source_RA)+' # source right ascension [rad]\n')
fo.write(str(source_DEC)+' # source declination [rad]\n')

fo.write('# inputs for detector\n')
fo.write(str(detector_fGWmin)+' # detector_fGWmin, smallest frequency considered [Hz]\n')
fo.write(str(detector_fGWmax)+' # detector_fGWmax, smallest frequency considered [Hz]\n')
fo.write(str(detector_t0)+' # detector_t0, reference time at which to define the location of the detector [seconds after vernal equinox]\n')
for i in range(len(detector_RA0)):
   fo.write(str(detector_RA0[i])+' # detector_RA0, right ascension of detector '+str(i)+' at t0 [rad]\n')
   fo.write(str(detector_DEC0[i])+' # detector_DEC0, declination of detector '+str(i)+' at t0 [rad]\n')

fo.close()

#######################
# create an instance of the GW_event class
event = AI_PE.GW_event(
    source_Mc,
    source_iota,
    source_phi0,
    source_tc,
    source_dL,
    source_RA,
    source_DEC,
    source_psi,
    detector_fGWmin,
    detector_fGWmax,
    detector_t0,
    detector_RA0,
    detector_DEC0
    )
# compute Fisher Matrix
event.get_SNR_FisherMatrix(Ncores_mp=Ncores)
# compute covariance matrices without priors
event.get_CoVaMat()
event.get_CoVaMat_dimless()
event.get_angular_resolution()
# compute covariance matrices with priors
event.get_CoVaMat_priors()
event.get_CoVaMat_priors_dimless()
event.get_angular_resolution_priors()
# save event
with open(fpath_out+'/data.pkl', 'wb') as outp:
    pickle.dump(event, outp, pickle.HIGHEST_PROTOCOL)
del event

#######################
# write message to std out
print("###########################################")
print(
    "runtime:",
    str(int((time.time() - test_t0)/60**2)),
    "h"
    )
print("###########################################")