############################################################################
# this example script runs the AIMforGW.ParamEstimator_SpaceAI code 
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
import ParamEstimator_SpaceAI as AI_PE

######################################################
# set parameters

#######################
# multiprocessing 
Ncores = 4 # number of cores used for paralellized computation

#######################
# GW source
source_Mc = 25. # detector-frame chirp mass [Msol]])
source_q = 1.15 # mass ratio (m_1/m_2)
source_dL = 100. # luminosity distance [Mpc]
source_phi0 = 0. # phase of the GW signal at fGWmin
source_tc = 0. # time of merger, measured from the solar equinox [seconds]
source_iota = np.pi/4 # angle between sky localization and orbital momentum of the binary [rad]
source_psi = np.pi/3. # polarization angle [rad]
source_RA = np.pi*60.0/180 # source right ascension [rad]
source_DEC = np.pi*(30.0-23.4)/180 # source declination [rad]

#######################
# detector
detector_fGWmin = 0.03 # smallest frequency considered [Hz]
detector_fGWmax = 4. # smallest frequency considered [Hz]
detector_max_measurement_time = 1.*3.155814954e7 # if detector_max_measurement_time [seconds] is longer than the lifetime of the source in the band, only the last detector_max_measurement_time piece of the signal in the band is considered
detector_orbit_R_cb = 8.44e6 # radius of the orbit of the center of the baseline around Earth [m]
detector_orbit_R_sat = 2e7 # radius of the orbit of the satellite around Earth [m], for period
detector_orbit_t0 = 3.155814954e7/4 # reference time for fixing orbit of satellite around Earth [seconds]
detector_orbit_RA0 = np.pi/2 # right ascension of the satellite at t0 [rad]
detector_orbit_DEC0 = -23.4*np.pi/180  # declination of the satellite at t0 [rad]

#######################
# post-newtonian order
PN_order_phase = 3.5
PN_order_amplitude = 3.0

#######################
# set up path to store results
timestamp_str = time.strftime("%Y%m%d_%H%M")
fpath_out = "ExampleOutput/results_Example_SpaceAI_"+timestamp_str+"/"

######################################################
# run
######################################################
# get some derived parameters
detector_orbit_period = AI_PE.antennaFuns_satellites.period_satellite_earth(detector_orbit_R_sat)

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
fo.write('# RUNNING WITH AIMforGW.ParamEstimator_SpaceAI\n')

fo.write('# multiprocessing setup\n')
fo.write(str(Ncores)+' # Ncores\n')

fo.write('# inputs for GW source\n')
fo.write(str(source_Mc)+' # detector-frame chirp mass [Msol]]\n')
fo.write(str(source_q)+' # source_q, mass ratio (m_1/m_2)\n')
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
fo.write(str(detector_max_measurement_time)+' # detector_max_measurement_time, max length of signal considered [seconds]\n')
fo.write(str(detector_orbit_R_cb)+' # detector_orbit_R_cb, radius of the orbit of the center of the baseline around Earth [m]\n')
fo.write(str(detector_orbit_R_sat)+' # detector_orbit_R_sat, radius of the orbit of the satellite around Earth [m], for period\n')
fo.write(str(detector_orbit_t0)+' # detector_orbit_t0, reference time for fixing orbit of satellite around Earth [seconds]\n')
fo.write(str(detector_orbit_RA0)+' # detector_orbit_RA0, right ascension of the satellite at t0 [rad]\n')
fo.write(str(detector_orbit_DEC0)+' # detector_orbit_DEC0, declination of the satellite at t0 [rad]\n')

fo.write('# inputs computation\n')
fo.write(str(PN_order_phase)+' # PN_order_phase\n')
fo.write(str(PN_order_amplitude)+' # PN_order_amplitude\n')
fo.close()

#######################
# create an instance of the GW_event class
event = AI_PE.GW_event(
    source_Mc,
    source_q,
    source_iota,
    source_phi0,
    source_tc,
    source_dL,
    source_RA,
    source_DEC,
    source_psi,
    detector_fGWmin,
    detector_fGWmax,
    detector_orbit_R_cb,
    detector_orbit_period,
    detector_orbit_t0,
    detector_orbit_RA0,
    detector_orbit_DEC0,
    PN_order_phase,
    PN_order_amplitude,
    max_measurement_time = detector_max_measurement_time
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