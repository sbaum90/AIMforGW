import numpy as np
import time

from scipy import interpolate
from scipy import fft

import multiprocessing as mp


###################################################
# import the noise curve data
noise_curve_data = np.loadtxt('NoiseCurve_Space.dat').T
noise_fun_PSD_Hz = interpolate.interp1d(
   noise_curve_data[0], 
   noise_curve_data[1]**2, 
   bounds_error=False, 
   fill_value = np.inf
   )


###################################################
# import modules
import helper_funs
import waveform_PN
import antennaFuns_satellites


###################################################
# some functions to put it all together
def get_htilde(
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
   detector_orbit_R,
   detector_orbit_period,
   detector_orbit_t0,
   detector_orbit_RA0,
   detector_orbit_DEC0,
   PN_order_phase,
   PN_order_waveform,
   win_func,
   tD_edges,
   clock_tD
   ):
   """
   Returns the frequency domain waveform
   GW source inputs:
      source_Mc -- detector-frame chirp mass [Msol]
      source_q -- mass ratio q=m_1/m_2
      source_iota -- angle between sky localization and orbital momentum 
                     of the binary [rad]
      source_phi0 -- phase of the GW signal at fGWmin
      source_tc -- time of merger, measured from the solar equinox [seconds]
      source_dL -- luminosity distance [Mpc]
      source_RA --  right ascension [rad]
      source_DEC -- declination [rad]
      source_psi -- polarization angle [rad]
   Detector inputs:
      detector_fGWmin -- smallest frequency considered [Hz]
      detector_fGWmax -- smallest frequency considered [Hz]
      detector_orbit_R -- radius of the orbit of the satellites around 
                          Earth [m]
      detector_orbit_period -- period of the detector around Earth [seconds]
      detector_orbit_t0 -- reference time for fixing orbit of satellite 
                           around Earth [seconds]
      detector_orbit_RA0 -- right ascension of the satellite at t0 [rad]
      detector_orbit_DEC0 -- declination of the satellite at t0 [rad]
   Parameters for computation:
      PN_order_phase -- order of PN expansion of phase
      PN_order_waveform -- order of PN expansion of waveform
      win_func -- array with the window function for the signal
      tD_edges -- tuple with detector time start and end of the signal
      clock_tD -- reference time for clock location
   """
   # compute mass params
   source_m = source_Mc*(1.+source_q)**1.2/source_q**0.6
   source_mu = source_Mc*source_q**0.4/(1.+source_q)**0.8
   source_nu = source_q/(1.+source_q)**2
   source_Delta = (source_q-1.)/(source_q+1.)
   # get theta0 integration constant
   source_theta_intc = waveform_PN.fix_theta0(
      source_nu, 
      source_m, 
      PN_order_phase, 
      detector_fGWmin, 
      detector_fGWmax, 
      phi0 = source_phi0
      )
   # compute detector-frame time vector
   time_vec_detector = np.linspace(
      tD_edges[0], 
      tD_edges[1], 
      num = win_func.size+1,
      dtype = np.longdouble
      )[:-1]
   # get helicocentric time vector and antenna functions
   time_vec_heliocentric, Fp, Fc = antennaFuns_satellites.time_delay_antenna_funs(
      time_vec_detector,
      source_RA, 
      source_DEC, 
      source_psi, 
      detector_orbit_R, 
      detector_orbit_t0, 
      detector_orbit_RA0, 
      detector_orbit_DEC0, 
      clock_tD,
      detector_period = detector_orbit_period
      )
   # get time-domain polarization waveform
   hp, hc = waveform_PN.get_hp_hc(
      time_vec_heliocentric,
      source_tc,
      source_nu,
      source_mu, 
      source_m,
      source_Delta,
      source_theta_intc,
      source_iota,
      source_dL,
      PN_order_phase,
      PN_order_waveform
      )
   # get the frequency domain waveform
   h = hp*Fp + hc*Fc
   h[np.where(np.isnan(h))[0]] = 0.
   htilde = (
      (time_vec_detector.max()-time_vec_detector.min())/win_func.size
      * fft.rfft(win_func*h)
      )
   return htilde


def SNR_FisherMatrix(
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
   detector_orbit_R,
   detector_orbit_period,
   detector_orbit_t0,
   detector_orbit_RA0,
   detector_orbit_DEC0,
   PN_order_phase,
   PN_order_waveform,
   Delta_Mc_rel,
   Delta_q_rel,
   Delta_iota_abs,
   Delta_phi0_abs,
   Delta_tc_abs,
   Delta_dL_rel,
   Delta_angleSL_abs,
   Delta_psi_abs,
   win_func,
   tD_edges,
   clock_tD
   ):
   """
   Big ugly wrapper function which returns the 
   SNR^2 and (upper triangle of the) FisherMatrix of the GW signal
   GW source inputs:
      source_Mc -- detector-frame chirp mass [Msol]
      source_q -- mass ratio q=m_1/m_2
      source_iota -- angle between sky localization and orbital momentum 
                     of the binary [rad]
      source_phi0 -- phase of the GW signal at fGWmin
      source_tc -- time of merger, measured from the solar equinox [seconds]
      source_dL -- luminosity distance [Mpc]
      source_RA --  right ascension [rad]
      source_DEC -- declination [rad]
      source_psi -- polarization angle [rad]
   Detector inputs:
      detector_fGWmin -- smallest frequency considered [Hz]
      detector_fGWmax -- smallest frequency considered [Hz]
      detector_orbit_R -- radius of the orbit of the satellites around 
                          Earth [m]
      detector_orbit_period -- period of the detector around Earth [seconds]
      detector_orbit_t0 -- reference time for fixing orbit of satellite 
                           around Earth [seconds]
      detector_orbit_RA0 -- right ascension of the satellite at t0 [rad]
      detector_orbit_DEC0 -- declination of the satellite at t0 [rad]
   Parameters for computation:
      PN_order_phase -- order of PN expansion of phase
      PN_order_waveform -- order of PN expansion of waveform
      Delta_Mc_rel -- relative change of chirp mass parameter for finite
                      difference computation of the Fisher matrix
      Delta_q_rel -- relative change of q parameter for the finite
                     difference computation of the Fisher Matrix
      Delta_iota_abs -- absolute change of iota parameter for finite
                        difference computation of the Fisher matrix
      Delta_phi0_abs -- ansolute change of phi0 parameter for finite
                        difference computation of the Fisher matrix [rad]
      Delta_tc_abs -- absolute change of t_c [seconds] parameter for finite
                      difference computation of the Fisher matrix
      Delta_dL_rel -- relative change of dL parameter for finite
                      difference computation of the Fisher matrix
      Delta_angleSL_abs -- absolute change in sky localization angle 
                           parameters for finite difference computation 
                           of Fisher matrix [rad]
      Delta_psi_abs -- absolute change of psi parameter for finite
                       difference computation of the Fisher matrix [rad]
      win_func -- array with the window function for the signal
      tD_edges -- tuple with detector time start and end of the signal
      clock_tD -- reference time for clock location
   """
   # make arrays of source parameters
   source_params = np.array([
      source_Mc,
      source_q,
      source_iota,
      source_phi0,
      source_tc,
      source_dL,
      source_RA,
      source_DEC,
      source_psi
      ])
   # make array of modified parameters for finite difference computation
   source_params_mod = np.array([
      source_Mc*(1.+Delta_Mc_rel),
      source_q*(1.+Delta_q_rel),
      source_iota+Delta_iota_abs,
      source_phi0+Delta_phi0_abs,
      source_tc+Delta_tc_abs,
      source_dL*(1.+Delta_dL_rel),
      source_RA+Delta_angleSL_abs,
      source_DEC+Delta_angleSL_abs,
      source_psi+Delta_psi_abs
      ])
   # make a new get_htilde with fixed detector and computational params
   func_htilde = lambda params: get_htilde(
      params[0],
      params[1],
      params[2],
      params[3],
      params[4],
      params[5],
      params[6],
      params[7],
      params[8],
      detector_fGWmin,
      detector_fGWmax,
      detector_orbit_R,
      detector_orbit_period,
      detector_orbit_t0,
      detector_orbit_RA0,
      detector_orbit_DEC0,
      PN_order_phase,
      PN_order_waveform,
      win_func,
      tD_edges,
      clock_tD
      )
   # get the reference signal
   htilde_ref = func_htilde(source_params)
   # get signals for varied parameters
   htilde_mod_list = []
   for i in range(len(source_params)):
      temp_params = np.copy(source_params)
      temp_params[i] = source_params_mod[i]
      htilde_mod_list.append(func_htilde(temp_params))
   # get frequencies
   htilde_freqs_ref = fft.rfftfreq(
      win_func.size,
      d = (tD_edges[1]-tD_edges[0])/win_func.size
      )
   # get noise curve
   noise_curve = noise_fun_PSD_Hz(htilde_freqs_ref)
   # get SNR
   SNR2 = helper_funs.inner_product_SNR(
      htilde_ref, 
      noise_curve, 
      htilde_freqs_ref
      )
   # get entries of Fisher matrix
   # create output array for Fisher matrix
   FisherMat = np.zeros((
      len(source_params),
      len(source_params)
      ))
   # fill upper triangle of Fisher matrix
   for i in range(FisherMat.shape[0]):
      for j in range(i,FisherMat.shape[1]):
         FisherMat[i,j] = helper_funs.inner_product_FisherM(
            htilde_ref,
            htilde_mod_list[i],
            htilde_mod_list[j],
            source_params_mod[i]-source_params[i],
            source_params_mod[j]-source_params[j],
            noise_curve,
            htilde_freqs_ref
            )
   return SNR2, FisherMat


###################################################
# and the GW_event class which wraps everything
class GW_event:
   def __init__(
      self,
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
      detector_orbit_R,
      detector_orbit_period,
      detector_orbit_t0,
      detector_orbit_RA0,
      detector_orbit_DEC0,
      PN_order_phase,
      PN_order_waveform,
      max_signal_length = 1e7,
      max_measurement_time = 3.155814954e7,
      window_eps = np.nan,
      Delta_Mc_rel = 1e-9,
      Delta_q_rel = 1e-4,
      Delta_iota_abs = 1e-4,
      Delta_phi0_abs = 1e-6,
      Delta_tc_abs = 1e-6,
      Delta_dL_rel = 1e-9,
      Delta_angleSL_abs = 1e-9,
      Delta_psi_abs = 1e-5,
      t_buffer = 1500.,
      Delta_autoadjust = True
      ):
      """
      GW source inputs:
         source_Mc -- detector-frame chirp mass [Msol]
         source_q -- mass ratio q=m_1/m_2
         source_iota -- angle between sky localization and orbital momentum 
                        of the binary [rad]
         source_phi0 -- phase of the GW signal at fGWmin
         source_tc -- time of merger, measured from the solar equinox [seconds]
         source_dL -- luminosity distance [Mpc]
         source_RA --  right ascension [rad]
         source_DEC -- declination [rad]
         source_psi -- polarization angle [rad]
      Detector inputs:
         detector_fGWmin -- smallest frequency considered [Hz]
         detector_fGWmax -- smallest frequency considered [Hz]
         detector_orbit_R -- radius of the orbit of the satellites around 
                             Earth [m]
         detector_orbit_period -- period of the detector around Earth [seconds]
         detector_orbit_t0 -- reference time for fixing orbit of satellite 
                              around Earth [seconds]
         detector_orbit_RA0 -- right ascension of the satellite at t0 [rad]
         detector_orbit_DEC0 -- declination of the satellite at t0 [rad]
      Parameters for computation:
         PN_order_phase -- order of PN expansion of phase
         PN_order_waveform -- order of PN expansion of waveform
         max_signal_length -- max number of points in a single 
                              time-domain signal chunk
         max_measurement_time -- if the lifetime of the signal in the band
                                 is larger than max_measurement_time, only the
                                 part of the signal within max_measurement_time 
                                 at frequencies below detector_fGWmax will be 
                                 considered. 
                                 Units of max_measurement_time are [seconds]
         window_eps -- width parameter for width of window function
                       if window_eps is left at the default value of np.nan,
                       it is automatically set to max([1e-5, 100/length of signal])
         Delta_Mc_rel -- relative change of chirp mass parameter for finite
                           difference computation of the Fisher matrix
         Delta_q_rel -- relative change of q parameter for the finite
                        difference computation of the Fisher Matrix
         Delta_iota_abs -- absolute change of iota parameter for finite
                           difference computation of the Fisher matrix [rad]
         Delta_phi0_abs -- ansolute change of phi0 parameter for finite
                           difference computation of the Fisher matrix [rad]
         Delta_tc_abs -- absolute change of t_c parameter [seconds] for finite
                         difference computation of the Fisher matrix
         Delta_dL_rel -- relative change of dL parameter for finite
                         difference computation of the Fisher matrix
         Delta_angleSL_abs -- absolute change in sky localization angle 
                              parameters for finite difference computation 
                              of Fisher matrix [rad]
         Delta_psi_abs -- absolute change of psi parameter for finite
                          difference computation of the Fisher matrix [rad]
         t_buffer -- buffer for time range in which signal is computed to 
                     make sure that the time-shift from the Doppler does 
                     not clip the edges of the signal [s]
         Delta_autoadjust -- boolean to turn on/off the automatic adjustment
                             of Delta_Mc_rel and Delta_q_rel with the lifetime
                             of the source
      """
      self.source_Mc = source_Mc
      self.source_q = source_q
      self.source_iota = source_iota
      self.source_phi0 = source_phi0
      self.source_tc = source_tc
      self.source_dL = source_dL
      self.source_RA = source_RA
      self.source_DEC = source_DEC
      self.source_psi = source_psi
      self.detector_fGWmin = detector_fGWmin
      self.detector_fGWmax = detector_fGWmax
      self.detector_orbit_R = detector_orbit_R
      self.detector_orbit_period = detector_orbit_period
      self.detector_orbit_t0 = detector_orbit_t0
      self.detector_orbit_RA0 = detector_orbit_RA0
      self.detector_orbit_DEC0 = detector_orbit_DEC0
      self.PN_order_phase = PN_order_phase
      self.PN_order_waveform = PN_order_waveform
      self.max_signal_length = max_signal_length
      self.max_measurement_time = max_measurement_time
      self.window_eps = window_eps
      self.Delta_Mc_rel = Delta_Mc_rel
      self.Delta_q_rel = Delta_q_rel
      self.Delta_iota_abs = Delta_iota_abs
      self.Delta_phi0_abs = Delta_phi0_abs
      self.Delta_tc_abs = Delta_tc_abs
      self.Delta_dL_rel = Delta_dL_rel
      self.Delta_angleSL_abs = Delta_angleSL_abs
      self.Delta_psi_abs = Delta_psi_abs
      self.t_buffer = t_buffer
      self.Delta_autoadjust = Delta_autoadjust
   def get_SNR_FisherMatrix(self, Ncores_mp=1):
      """
      Appends the class with SNR^2 and FisherMatrix of the GW signal
      inputs:
         Ncores_mp -- number of cores for parallelized computation
      """
      #compute mass params
      self.source_m = self.source_Mc*(1.+self.source_q)**1.2/self.source_q**0.6
      self.source_mu = self.source_Mc*self.source_q**0.4/(1.+self.source_q)**0.8
      self.source_nu = self.source_q/(1.+self.source_q)**2
      self.source_Delta = (self.source_q-1.)/(self.source_q+1.)
      # get an estimate for the time the signal enters/leaves the band
      # compute PN time variable at fGW = fGWmin
      source_theta_in = waveform_PN.find_theta_fGW(
         self.source_nu, 
         self.source_m, 
         self.PN_order_phase, 
         self.detector_fGWmin
         )
      # compute PN time variable at fGW = fGWmax
      source_theta_out = waveform_PN.find_theta_fGW(
         self.source_nu, 
         self.source_m, 
         self.PN_order_phase, 
         self.detector_fGWmax
         )
      # compute corresponding times in the detector frame
      tD_in = (
         self.source_tc 
         - waveform_PN.tau(
            source_theta_in,
            self.source_nu,
            self.source_m
            )
         )
      if np.isnan(source_theta_out):
         tD_out = self.source_tc
      else:
         tD_out = (
            self.source_tc 
            - waveform_PN.tau(
               source_theta_out,
               self.source_nu,
               self.source_m
               )
            )
      # estimated lifetime of the source in the band
      self.source_lifetime = tD_out - tD_in
      if self.source_lifetime > self.max_measurement_time:
         tD_in = tD_out - self.max_measurement_time
      # slap on buffer for the time-shift from the Doppler 
      tD_in -= self.t_buffer
      tD_out += self.t_buffer
      # compute into how many chunks of data the signal should be divided
      Nruns = int(
         np.ceil(
            (tD_out - tD_in)
            * 2.*self.detector_fGWmax
            / self.max_signal_length
            )
         )
      # if computation parallelized, set Nruns to a multiple of Ncores_mp
      if Ncores_mp > 1:
         Nruns = int(np.ceil(Nruns/Ncores_mp)*Ncores_mp)
      # make list of time slice edges
      tD_edges = np.linspace(tD_in, tD_out, num=Nruns+1)
      Npoints_per_run = int(
         np.ceil(
            2.*(tD_edges[1] - tD_edges[0])*self.detector_fGWmax
            )
         )
      # get window function
      if np.isnan(self.window_eps):
         self.window_eps = np.max([1e-5, 1e2/Npoints_per_run])
      win_func = helper_funs.window_planck(Npoints_per_run, self.window_eps)
      # autoadjust Delta_Mc_rel and Delta_q_rel
      if self.Delta_autoadjust:
         self.Delta_Mc_rel *= 3e5/np.min(
            [self.source_lifetime, self.max_measurement_time]
            )
      # get the time and location of the clock
      self.clock_tD = np.double(0.5*(tD_in+tD_out))
      self.clock_r = antennaFuns_satellites.orbit_earth_sun(np.array(self.clock_tD))
      # make list of run_params
      run_params_list = []
      for i in range(Nruns):
         run_params_list.append((
            self.source_Mc,
            self.source_q,
            self.source_iota,
            self.source_phi0,
            self.source_tc,
            self.source_dL,
            self.source_RA,
            self.source_DEC,
            self.source_psi,
            self.detector_fGWmin,
            self.detector_fGWmax,
            self.detector_orbit_R,
            self.detector_orbit_period,
            self.detector_orbit_t0,
            self.detector_orbit_RA0,
            self.detector_orbit_DEC0,
            self.PN_order_phase,
            self.PN_order_waveform,
            self.Delta_Mc_rel,
            self.Delta_q_rel,
            self.Delta_iota_abs,
            self.Delta_phi0_abs,
            self.Delta_tc_abs,
            self.Delta_dL_rel,
            self.Delta_angleSL_abs,
            self.Delta_psi_abs,
            win_func,
            (tD_edges[i], tD_edges[i+1]),
            self.clock_tD
            ))
      # start time (used to print time elapsed)
      starttime = time.time() 
      print("Starting calculation of FisherMat using",
         str(int(Ncores_mp)),
         "cores"
         )
      print(str(Nruns),
         "time-domain chunks will be computed...")
      if Ncores_mp > 1:
         pool = mp.Pool(Ncores_mp)
         p = pool.starmap_async(SNR_FisherMatrix, run_params_list)
         helper_funs.track_mp_progress(p, run_params_list, starttime)
         temp_results = p.get()
         pool.close()
         pool.join()
      else:
         temp_results = []
         for i in range(len(run_params_list)):
            temp_results.append(SNR_FisherMatrix(*run_params_list[i]))
            print(
               str(i+1),
               "of",
               str(len(run_params_list)), 
               "time-domain chunks computed; runtime:",
               str(int((time.time() - starttime)/60)),
               "min"
               )
      # collect results
      self.SNR2 = 0.
      self.FisherMat = np.zeros((9,9))
      for i in range(len(temp_results)):
         self.SNR2 += temp_results[i][0]
         self.FisherMat += temp_results[i][1]
      # make Fisher matrix symmetric
      for i in range(self.FisherMat.shape[0]):
         for j in range(i+1, self.FisherMat.shape[1]):
            self.FisherMat[j,i] = self.FisherMat[i,j]
      # and add dimensionless version of Fisher matrix
      self.FisherMat_dimless = np.copy(self.FisherMat)
      source_params_dimful = np.ones(self.FisherMat.shape[0])
      source_params_dimful[0] = self.source_Mc
      source_params_dimful[5] = self.source_dL
      for i in range(self.FisherMat.shape[0]):
         self.FisherMat_dimless[:,i] *= source_params_dimful[i]
         self.FisherMat_dimless[i,:] *= source_params_dimful[i]
      # key for Fisher/Covariance matrix entries
      self.FisherMat_key = {
         0 : 'source_Mc [Msol]',
         1 : 'source_q',
         2 : 'source_iota [rad]',
         3 : 'source_phi0 [rad]',
         4 : 'source_tc [s]',
         5 : 'source_dL [Mpc]',
         6 : 'source_RA [rad]',
         7 : 'source_DEC [rad]',
         8 : 'source_psi [rad]',
         }
      # matrix to be added to Fisher matrix to include
      # Gaussian priors with std (2)pi for angles.
      self.PriorMat = np.diag([
         0., 
         0., 
         1./np.pi**2, 
         1./(4.*np.pi**2),
         0.,
         0.,
         1./(4.*np.pi**2),
         1./np.pi**2, 
         1./(4.*np.pi**2)
         ])
      return
   def get_CoVaMat(self):
      """
      Appends class by covariance matrix.
      If Fisher Matrix not computed, executing this
      function will run the FisherMatrix computation
      """
      try:
         self.FisherMat
      except:
         self.get_SNR_FisherMatrix()
      self.CoVaMat = np.linalg.inv(np.double(self.FisherMat))
      return
   def get_CoVaMat_dimless(self):
      """
      Appends class by dimensionless covariance matrix.
      If Fisher Matrix not computed, executing this
      function will run the FisherMatrix computation
      """
      try:
         self.FisherMat_dimless
      except:
         self.get_SNR_FisherMatrix()
      self.CoVaMat_dimless = np.linalg.inv(
         np.double(
            self.FisherMat_dimless
            )
         )
      return
   def get_CoVaMat_priors(self):
      """
      Appends class by covariance matrix
      including Gaussian priors with std (2)pi for angles.
      If Fisher Matrix not computed, executing this
      function will run the FisherMatrix computation
      """
      try:
         self.FisherMat
      except:
         self.get_SNR_FisherMatrix()
      self.CoVaMat_priors = np.linalg.inv(
         np.double(
            self.FisherMat + self.PriorMat
            )
         )
      return
   def get_CoVaMat_priors_dimless(self):
      """
      Appends class by dimensionless covariance matrix
      including Gaussian priors with std (2)pi for angles.
      If Fisher Matrix not computed, executing this
      function will run the FisherMatrix computation
      """
      try:
         self.FisherMat_dimless
      except:
         self.get_SNR_FisherMatrix()
      self.CoVaMat_priors_dimless = np.linalg.inv(
         np.double(
            self.FisherMat_dimless + self.PriorMat
            )
         )
      return
   def get_angular_resolution(self):
      """
      Appends class by angular resolution in [sr]
      If Fisher CoVaMat (and FisherMatrix) is not computed,
      executing this function will run the CoVaMat 
      (and FisherMatrix) computation
      """
      try:
         self.CoVaMat
      except:
         self.get_CoVaMat()
      self.angular_resolution = (
         2.*np.pi
         * np.sin(np.pi/2. - self.source_DEC)
         * np.sqrt(
            self.CoVaMat[6,6]*self.CoVaMat[7,7]
            - self.CoVaMat[6,7]**2
            )
         )
      return
   def get_angular_resolution_priors(self):
      """
      Appends class by angular resolution in [sr]
      including Gaussian priors with std (2)pi for angles.
      If Fisher CoVaMat (and FisherMatrix) is not computed,
      executing this function will run the CoVaMat 
      (and FisherMatrix) computation
      """
      try:
         self.CoVaMat_priors
      except:
         self.get_CoVaMat_priors()
      self.angular_resolution_priors = (
         2.*np.pi
         * np.sin(np.pi/2. - self.source_DEC)
         * np.sqrt(
            self.CoVaMat_priors[6,6]*self.CoVaMat_priors[7,7]
            - self.CoVaMat_priors[6,7]**2
            )
         )
      return
