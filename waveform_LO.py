###################################################
# leading order waveform computation following Maggiore 2008, chapter 4.1
###################################################

import numpy as np

# some constant and unit conversions
G_m3_Msol_s2 = 1.32716e20 # gravitational constant in [m^3/M_sol/s^2]
c_m_s = 2.99792458e8 # speed of light in [m/s]
Mpc_per_m = 3.2408e-23

def GWPhase(Mc, Theta0, tau):
   """
   Returns the phase of the GW signal in radians for inputs:
      Mc -- detector frame chirp mass [Msol]
      Theta0 - GW phase at coalesence
      tau -- time to coalesence [seconds] 
   Using Eq. 4.193 of Maggiore
   """
   return -2.*(5.*G_m3_Msol_s2*Mc/c_m_s**3)**(-5./8.)*tau**(5./8.)+Theta0


def fGW(Mc, tau):
   """
   Returns the frequency of the GW signal in Hz for inputs:
      Mc -- detector frame chirp mass [Msol]
      tau -- time to coalesence [seconds] 
   Using Eq. 4.195 of Maggiore
   """
   return 1./np.pi*(5./256./tau)**(3./8.)*(G_m3_Msol_s2*Mc/c_m_s**3)**(-5./8.)


def tau_fGW(Mc, fGW):
   """
   Returns the time before merger in [sec] at which the GW signal has the 
   frequency fGW for inputs:
      Mc -- detector frame chirp mass [Msol]
      fGW -- frequency of the GW signal [Hz] 
   Inverting Eq. 4.195 of Maggiore
   """
   return 5./256. * (np.pi*fGW)**(-8./3.)*(c_m_s**3/G_m3_Msol_s2/Mc)**(5./3.)


def hfac(Mc, dL, fGW):
   """
   returns common amplitude factor for GW strain, Eq. 4.194 of Maggiore
   Inputs:
      Mc -- detector frame chirp mass [Msol]
      dL -- luminosity distance in Mpc
      fGW -- GW frequency [Hz]
   """
   return 4./dL*Mpc_per_m*(G_m3_Msol_s2*Mc/c_m_s**2)**(5./3.)*(np.pi*fGW/c_m_s)**(2./3.)


def get_hp_hc(Mc, iota, dL, Theta0, tau):
   """
   return a tuple of plus-polarization, cross polarization strain signals. 
   Inputs:
      Mc -- detector frame chirp mass [Msol]
      iota -- angle between the sky-localization and the orbital momentum of the binary [rad]
      dL -- luminosity distance in Mpc
      Theta0 - GW phase at coalesence
      tau -- time to coalesence [seconds] 
   """
   fGW_vec = fGW(Mc, tau)
   phase_vec = GWPhase(Mc, Theta0, tau)
   hfac_vec = hfac(Mc, dL, fGW_vec)
   return hfac_vec*(1.+np.cos(iota)**2)/2.*np.cos(phase_vec), hfac_vec*np.cos(iota)*np.sin(phase_vec)
