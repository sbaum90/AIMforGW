###################################################
# Antenna Functions for satellite-based detector,
# assuming two satellites trailing each other in 
# their orbit around Earth
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


def period_satellite_earth(R):
   """
   returns the period in [seconds] of a satellite
   in a circular orbit with radius R (in [m]) 
   around Earth
   """
   MEarth = 3.0035e-6 # mass of the Earth in [Msol]
   return 2.*np.pi*R**1.5/np.sqrt(G_m3_Msol_s2*MEarth)


def orbit_satellite_earth(t, R, t0, RA0, DEC0, period=np.nan):
   """
   returns the position in [m] of a satellite orbiting Earth
   in Earth-centric coordinates for inputs:
      t -- time in [seconds]
      R -- radius of orbit around the center of the Earth
      t0 -- reference time at which to define orbit [seconds after vernal equinox]
      RA0 -- right ascension of satellite at t0 [rad]
      DEC0 -- Declination of orbit at t0 [rad]
      period -- period of satellite orbit. If no input, the period is
                is computed for a gravitationally bound orbit around 
                Earth
   """
   if np.isnan(period):
      period = period_satellite_earth(R)
   RotMat2 = np.array([
      [np.cos(RA0), -np.sin(RA0), 0.],
      [np.sin(RA0), np.cos(RA0), 0.],
      [0., 0., 1.]
      ])
   RotMat1 = np.array([
      [np.cos(DEC0), 0., -np.sin(DEC0)],
      [0., 1., 0.],
      [np.sin(DEC0), 0., np.cos(DEC0)]
      ])
   phase = 2.*np.pi/period*(t-t0)
   if t.ndim == 0:
      LocVec = np.array([np.cos(phase), np.sin(phase), 0.])
      return R*np.dot(RotMat2, np.dot(RotMat1, LocVec))
   elif t.ndim == 1:
      LocVec = np.array([np.cos(phase), np.sin(phase), np.zeros(phase.size)])
      out = R*np.dot(RotMat2, np.dot(RotMat1, LocVec))
      return out.T
   else:
      print("The format of t you entered in orbit_earth_sun is not valid")
      print("Abort code")
      sys.exit()


def baseline_direction(t, R, t0, RA0, DEC0, period=np.nan):
   """
   returns the direction of the baseline for a constellation of two
   trailing satellites trailing each other in the same orbit around
   Earth in equatorial coordinates.
   Inputs:
      t -- time in [seconds]
      R -- radius of orbit around the center of the Earth
      t0 -- reference time at which to define orbit [seconds]
      RA0 -- right ascension of satellite at t0 [rad]
      Dec0 -- Declination of orbit at t0 [rad]
      period -- period of satellite orbit. If no input, the period is
                is computed for a gravitationally bound orbit around 
                Earth
   """
   if np.isnan(period):
      period = period_satellite_earth(R)
   RotMat2 = np.array([
      [np.cos(RA0), -np.sin(RA0), 0.],
      [np.sin(RA0), np.cos(RA0), 0.],
      [0., 0., 1.]
      ])
   RotMat1 = np.array([
      [np.cos(DEC0), 0., -np.sin(DEC0)],
      [0., 1., 0.],
      [np.sin(DEC0), 0., np.cos(DEC0)]
      ])
   phase = 2.*np.pi/period*(t-t0)
   if t.ndim == 0:
      LocVec = np.array([-np.sin(phase), np.cos(phase), 0.])
      return np.dot(RotMat2, np.dot(RotMat1, LocVec))
   elif t.ndim == 1:
      LocVec = np.array([-np.sin(phase), np.cos(phase), np.zeros(phase.size)])
      out = np.dot(RotMat2, np.dot(RotMat1, LocVec))
      return out.T
   else:
      print("The format of t you entered in orbit_earth_sun is not valid")
      print("Abort code")
      sys.exit()


def time_delay_antenna_funs(
   time_vec, 
   source_RA, 
   source_DEC, 
   source_psi, 
   detector_R, 
   detector_t0, 
   detector_RA0, 
   detector_DEC0, 
   clock_tD,
   detector_period=np.nan
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
      detector_R -- radius of the satellites' orbit around Earth [m]
      detector_t0 -- reference time at which to define orbit of the detector [seconds]
      detector_RA0 -- right ascension of satellite at detector_t0 [rad]
      detector_DEC0 -- declination of satellite at detector_t0 [rad]
      clock_tD -- reference time for clock location
      detector period -- period of satellite orbit. If no input, the period is
         is computed for a gravitationally bound orbit around 
         Earth [seconds]
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
         + orbit_satellite_earth(
            time_vec, 
            detector_R, 
            detector_t0, 
            detector_RA0, 
            detector_DEC0, 
            period=detector_period
            )
         )
      out_time = time_vec + np.dot(nhat, r)/c_m_s
      baselinehat = baseline_direction(
         time_vec, 
         detector_R, 
         detector_t0, 
         detector_RA0, 
         detector_DEC0, 
         period=detector_period)
      out_Fp = np.dot(baselinehat, np.dot(e_tensor_p, baselinehat))
      out_Fc = np.dot(baselinehat, np.dot(e_tensor_c, baselinehat))
   elif time_vec.ndim == 1:
      r = (
         orbit_earth_sun(time_vec)
         - orbit_earth_sun(clock_tD)
         + orbit_satellite_earth(
            time_vec, 
            detector_R, 
            detector_t0, 
            detector_RA0, 
            detector_DEC0, 
            period=detector_period
            )
         )
      out_time = time_vec + np.dot(nhat, r.T)/c_m_s
      baselinehat = baseline_direction(
         time_vec, 
         detector_R, 
         detector_t0, 
         detector_RA0, 
         detector_DEC0, 
         period=detector_period).T
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
