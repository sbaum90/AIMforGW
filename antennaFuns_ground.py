###################################################
# Antenna Functions for ground-based detector,
# assuming vertical baselines on Earth's surface
###################################################

import numpy as np
import sys

# some constant and unit conversions
G_m3_Msol_s2 = 1.32716e20 # gravitational constant in [m^3/M_sol/s^2]
c_m_s = 2.99792458e8 # speed of light in [m/s]


def orbit_earth_sun(t):
   """
   returns the position of the Earth in [m] in a heliocentric
   equatorial coordinate system as a function of time in [s]
   Note that t=0 corresponds to the vernal equinox (Mar 20)
   """
   phi_ecl = 23.4/180*np.pi # angle of the ecliptical 
   om = 2.*np.pi/3.155814954e7 # angular frequency of the orbit of Earth around the Sun in [1/s]
   R = 1.4959802296e11 # average distance from the Earth to the Sun in [m]
   phase = om*t-np.pi
   RotMat = np.array([
         [1., 0., 0.],
         [0., np.cos(phi_ecl), np.sin(phi_ecl)],
         [0., -np.sin(phi_ecl), np.cos(phi_ecl)]
         ])
   if t.ndim == 0:
      LocVec = np.array([np.cos(phase), np.sin(phase), 0.])
      return R*np.dot(RotMat, LocVec)
   elif t.ndim == 1:
      LocVec = np.array([np.cos(phase), np.sin(phase), np.zeros(phase.size)])
      out = R*np.dot(RotMat, LocVec)
      return out.T
   else:
      print("The format of t you entered in orbit_earth_sun is not valid")
      print("Abort code")
      sys.exit()


def baseline_direction(t, t0, RA0, DEC0):
   """
   returns the (unit length) direction of the baseline
   for a vertical detector at the surface of the Earth
   in equatorial Cartesian coordinates for inputs:
      t -- time in [seconds]
      t0 -- reference time at which to define the location of the detector 
            [seconds after vernal equinox]
      RA0 -- right ascension of detector at t0 [rad]
      DEC0 -- declination of detector at t0 [rad]
   """
   om = 2.*np.pi/86164.09054 # angular frequency of the orbit of Earth in [1/2]
   phase = om*(t-t0) + RA0
   if t.ndim == 0:
      return np.array([
         np.cos(DEC0)*np.cos(phase),
         np.cos(DEC0)*np.sin(phase),
         np.sin(DEC0)
         ])
   elif t.ndim == 1:
      return np.array([
         np.cos(DEC0)*np.cos(phase),
         np.cos(DEC0)*np.sin(phase),
         np.sin(DEC0)*np.ones(phase.size)
         ]).T
   else:
      print("The format of t you entered in orbit_earth_sun is not valid")
      print("Abort code")
      sys.exit()


def loc_baseline_earth(t, t0, RA0, DEC0):
   """
   returns the position in [m] of the detector on the surface of Earth
   in Earth-centric equatorial Cartesian coordinates for inputs:
      t -- time in [seconds]
      t0 -- reference time at which to define the location of the detector 
            [seconds after vernal equinox]
      RA0 -- right ascension of detector at t0 [rad]
      DEC0 -- declination of detector at t0 [rad]
   """
   R = 6.371e6 # average radius of Earth in [m]
   return R*baseline_direction(t, t0, RA0, DEC0)
   

def time_delay_antenna_funs(
   time_vec, 
   source_RA, 
   source_DEC, 
   source_psi, 
   detector_t0, 
   detector_RA0, 
   detector_DEC0,
   clock_tD
   ):
   """
   returns a tuple of three arrays:
   - times in heliocentric system corresponding to time_vec at the detector 
      in [seconds]. Note that time=0 corresponds to the vernal equinox.
   - F_+ - antenna function for +-polarization
   - F_cross - antenna function for - polarization
   Inputs:
      time_vec -- times in [seconds] at the detector, measured from the vernal equinox
      source_RA -- right ascension of the GW source [rad]
      source_DEC -- declination of the GW source [rad]
      source_psi -- polarization angle [psi] of the source [rad]
      detector_t0 -- reference time at which to define the location of the detector 
            [seconds after vernal equinox]
      detector_RA0 -- right ascension of detector at t0 [rad]
      detector_DEC0 -- declination of detector at t0 [rad]
      clock_tD -- reference time for clock location
   """
   # compute polarization tensors in equatorial coordinate system
   nhat = np.array([
      np.cos(source_RA)*np.cos(source_DEC),
      np.sin(source_RA)*np.cos(source_DEC),
      np.sin(source_DEC)
      ])
   ihat = np.array([np.sin(source_RA), -np.cos(source_RA), 0.])
   jhat = np.cross(ihat, nhat)
   jhat /= np.linalg.norm(jhat)
   eps_tensor_p = (
      np.tensordot(ihat, ihat, axes=0) 
      - np.tensordot(jhat, jhat, axes=0)
      )
   eps_tensor_c = (
      np.tensordot(ihat, jhat, axes=0) 
      + np.tensordot(jhat, ihat, axes=0)
      )
   e_tensor_p = (
      eps_tensor_p*np.cos(2.*source_psi) 
      + eps_tensor_c*np.sin(2.*source_psi)
      )
   e_tensor_c = (
      -eps_tensor_p*np.sin(2.*source_psi) 
      + eps_tensor_c*np.cos(2.*source_psi)
      )
   # compute time in heliocentric frame and antenna funs
   if time_vec.ndim == 0:
      r = (
         orbit_earth_sun(time_vec)
         - orbit_earth_sun(clock_tD)
         + loc_baseline_earth(
            time_vec, 
            detector_t0, 
            detector_RA0, 
            detector_DEC0
            )
         )
      out_time = time_vec + np.dot(nhat, r)/c_m_s
      baselinehat = baseline_direction(
         time_vec, 
         detector_t0, 
         detector_RA0, 
         detector_DEC0
         )
      out_Fp = np.dot(baselinehat, np.dot(e_tensor_p, baselinehat))
      out_Fc = np.dot(baselinehat, np.dot(e_tensor_c, baselinehat))
   elif time_vec.ndim == 1:
      r = (
         orbit_earth_sun(time_vec)
         - orbit_earth_sun(clock_tD)
         + loc_baseline_earth(
            time_vec, 
            detector_t0, 
            detector_RA0, 
            detector_DEC0
            )
         )
      out_time = time_vec + np.dot(nhat, r.T)/c_m_s
      baselinehat = baseline_direction(
         time_vec, 
         detector_t0, 
         detector_RA0, 
         detector_DEC0
         ).T
      dummy = np.dot(e_tensor_p, baselinehat)
      out_Fp = np.array([
         np.dot(baselinehat[:,i], dummy[:,i]) for i in range(time_vec.size)
         ]) 
      dummy = np.dot(e_tensor_c, baselinehat)
      out_Fc = np.array([
         np.dot(baselinehat[:,i], dummy[:,i]) for i in range(time_vec.size)
         ]) 
   else:
      print("The format of t you entered in time_delay_antenna_funs is not valid")
      print("Abort code")
      sys.exit()
   return out_time, out_Fp, out_Fc
