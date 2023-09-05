###################################################
# Post Newtonian Waveform computation following Blanchet 1310.1528
###################################################

import numpy as np
from scipy import optimize

# some constant and unit conversions
G_m3_Msol_s2 = 1.32716e20 # gravitational constant in [m^3/M_sol/s^2]
c_m_s = 2.99792458e8 # speed of light in [m/s]
Mpc_per_m = 3.2408e-23

def theta(t, nu, m, tc=0.):
   """
   Returns dimensionless time variable for inputs:
      t -- time [s]
      nu -- symmetric mass ratio of the binary
      m -- total mass of the binary [Msol]
      tc -- time of coalesence (i.e. time where theta=0)
   """
   return nu*c_m_s**3/(5.*G_m3_Msol_s2*m)*(tc-t)


def x_PN(theta, nu, PN_order):
   """
   Returns the parameter x describing the frequency
   compute to a given PN order. Inputs:
      theta -- dimensionless time variable
      nu -- symmetric mass ratio of the binary
      PN_order -- order of PN expansion
   """
   # compute the PN coefficients
   c0 = 1.
   c10 = 743./4032. + 11./48.*nu
   c15 = np.pi*(-1./5.)
   c20 = 19583./254016. + 24401./193536.*nu + 31./288.*nu**2
   c25 = np.pi*(-11891./53760. + 109./1920.*nu)
   c30 = (
      - 10052469856691./6008596070400
      + 1./6.*np.pi**2
      + 107./420.*np.euler_gamma
      - 107./3360.*np.log(theta/256.)
      + (3147553127./780337152. - 451./3072.*np.pi**2)*nu
      - 15211./442368.*nu**2
      + 25565./331776.*nu**3
      )
   c35 = np.pi*(
      - 113868647./433520640.
      - 31821./143360.*nu
      + 294941./3870720.*nu**2
      )
   # output to desired PN order
   out = c0
   if PN_order > 0:
      out += c10*theta**(-1./4.)
   if PN_order > 1.4:
      out += c15*theta**(-3./8.)
   if PN_order > 1.9:
      out += c20*theta**(-1./2.)
   if PN_order > 2.4:
      out += c25*theta**(-5./8.)
   if PN_order > 2.9:
      out += c30*theta**(-3./4.)
   if PN_order > 3.4:
      out += c35*theta**(-7./8.)
   return theta**(-1./4.)/4.*out


def phi_PN(theta, nu, PN_order, theta0):
   """
   Returns the orbital phase phi to a given PN order. 
   Inputs:
      theta -- dimensionless time variable
      nu -- symmetric mass ratio of the binary
      PN_order -- order of PN expansion
      theta0 -- constant of integration fixed by initial conditions
   """
   # compute the PN coefficients
   c0 = 1.
   c10 = 3715./8064. + 55./96.*nu
   c15 = np.pi*(-3./4.)
   c20 = 9275495./14450688. + 284875./258048.*nu + 1855./2048.*nu**2
   c25 = np.pi*(-38645./172032. + 65./2048.*nu)*np.log(theta/theta0)
   c30 = (
      831032450749357./57682522275840.
      - 53./40.*np.pi**2
      - 107./56.*np.euler_gamma
      + 107./448.*np.log(theta/256.)
      + (-126510089885./4161798144. + 2255./2048.*np.pi**2)*nu
      + 154565./1835008.*nu**2
      - 1179625./1769472.*nu**3
      )
   c35 = np.pi*(
      188516689./173408256. 
      + 488825./516096.*nu
      - 141769./516096.*nu**2
      )
   # output to desired PN order
   out = c0
   if PN_order > 0:
      out += c10*theta**(-1./4.)
   if PN_order > 1.4:
      out += c15*theta**(-3./8.)
   if PN_order > 1.9:
      out += c20*theta**(-1./2.)
   if PN_order > 2.4:
      out += c25*theta**(-5./8.)
   if PN_order > 2.9:
      out += c30*theta**(-3./4.)
   if PN_order > 3.4:
      out += c35*theta**(-7./8.)
   return -theta**(5./8.)/nu*out


def find_theta_fGW(nu, m, PN_order, fGW):
   """
   Returns the dimensionless time variable corresponding
   to a particular GW frequency
   Inputs:
      nu -- dimensionless mass ratio of the binary
      m -- total mass of the binary [Msol]
      PN_order -- order of PN expansion
      fGW -- GW frequency [Hz]
   """
   x0 = (np.pi*G_m3_Msol_s2*m*fGW/c_m_s**3)**(2./3.)
   try:
      logth = optimize.brentq(
      lambda x: x_PN(np.exp(x), nu, PN_order) - x0, 
      np.log(6.), 50.
      )
      return np.exp(logth)
   except:
      return np.nan


def tau(theta, nu, m):
   """
   Returns time before coalesence in [s].
   Inputs:
      theta -- dimensionless time variable
      nu -- symmetric mass ratio of the binary
      m -- total mass of the binary [Msol]
   """
   return (5.*G_m3_Msol_s2*m)/(nu*c_m_s**3)*theta


def fix_theta0(nu, m, PN_order, fGW0, fGW1, phi0=0.):
   """
   returns theta0 such that when then frequency
   of the GW signal is fGW0, the phase phi is phi0
   Choose theta0 such that the log is minimized at fgW1
   Inputs:
      nu -- dimensionless mass ratio of the binary
      m -- total mass of the binary [Msol]
      PN_order -- order of PN expansion
      fGW0 -- GW frequency at which to fix the phase [Hz]
      fGW1 -- GW frequency at which to minimize the 2.5PN log
      phi0 -- desired phase at fGW0
   """
   if PN_order > 2.4:
      # get theta when the signal has frequency fGW0 and fGW1
      th_fGW0 = find_theta_fGW(nu, m, PN_order, fGW0)
      try:
         th_fGW1 = find_theta_fGW(nu, m, PN_order, fGW1)
         # get phase at th_fGW0 with guess theta0 = th_fGW1
         phi_prior_fGW0 = phi_PN(th_fGW0, nu, PN_order, th_fGW1)
         # get target phase
         target_phi = phi0 + (phi_prior_fGW0 - np.mod(phi_prior_fGW0, 2.*np.pi))
         # compute theta_0
         out = optimize.brentq(
            lambda x: phi_PN(th_fGW0, nu, PN_order, x) - target_phi, 
            th_fGW1*(0.9), 
            th_fGW1*(1.+th_fGW0**0.65)
            )
      except: # if frequency never hits fGW1
         # get phase at th_fGW0 with guess theta0 = th_fGW0
         phi_prior_fGW0 = phi_PN(th_fGW0, nu, PN_order, th_fGW0)
         # get target phase
         target_phi = phi0 + (phi_prior_fGW0 - np.mod(phi_prior_fGW0, 2.*np.pi))
         # compute theta_0
         out = optimize.brentq(
            lambda x: phi_PN(th_fGW0, nu, PN_order, x) - target_phi, 
            th_fGW0**0.6, 
            th_fGW0**1.6
            )
      return out
   else:
      return find_theta_fGW(nu, m, PN_order, fGW0)


def E_PN(x, nu, mu, m, PN_order):
   """
   returns binding energy to a given PN order in [Msol.m2/s2]
   Inputs:
      x -- PN parameter describing the frequency
      nu -- symmetric mass ratio of the binary
      mu -- reduced mass of the binary [Msol]
      m -- total mass of the binary [Msol]
      PN_order -- order of PN expansion
   """
   # PN coefficients
   c0 = 1.
   c10 = -3./4. - 1./12.*nu
   c20 = -27./8. + 19./8.*nu - 1./24.*nu**2
   c30 = (
      -675./64. 
      + (34445./576. - 205./96.*np.pi**2)*nu 
      - 155./96.*nu**2 
      - 35./5184.*nu**3
      )
   # output to desired PN order
   out = c0
   if PN_order > 0:
      out += c10*x
   if PN_order > 1.9:
      out += c20*x**2
   if PN_order > 2.9:
      out += c30*x**3
   return -mu*c_m_s**2/2.*x*out


def M_ADM(m, E):
   """
   returns the ADM mass in [Msol] for inputs:
      m -- total mass of the binary [Msol]
      E -- post-Newtownian binding energy
   """
   return m + E/c_m_s**2


def psi_PN(theta, nu, mu, m, PN_order, theta0, Omega0):
   """
   Returns the auxiliary phase psi that enters the waveforms to a given PN order. 
   Inputs:
      theta -- dimensionless time variable
      nu -- symmetric mass ratio of the binary
      mu -- reduced mass of the binary [Msol]
      m -- total mass of the binary [Msol]
      PN_order -- order of PN expansion
      theta0 -- constant of integration fixed by initial conditions
      Omega0 -- Angular frequency when the wave enters the detector
   """
   x = x_PN(theta, nu, PN_order)
   phi = phi_PN(theta, nu, PN_order, theta0)
   E = E_PN(x, nu, mu, m, PN_order)
   M = M_ADM(m, E)
   Omega = c_m_s**3/(G_m3_Msol_s2*m)*x**1.5
   return phi - 2.*G_m3_Msol_s2/c_m_s**3*M*Omega*np.log(Omega/Omega0)


def H_plus_PN00(psi, iota):
   """
   0 PN order contribution to h_+
   """
   si = np.sin(iota)
   ci = np.cos(iota)
   c2p = np.cos(2.*psi)
   #return (
   #   - (1. + ci**2)*c2p 
   #   - 1./96.*si**2*(17. + ci**2)
   #   )
   # without the DC term
   return - (1. + ci**2)*c2p 


def H_plus_PN05(psi, iota, Delta):
   """
   0.5 PN order contribution to h_+
   """
   si = np.sin(iota)
   ci = np.cos(iota)
   cp = np.cos(psi)
   c3p = np.cos(3.*psi)
   return -si*Delta*(
      cp*(5./8. + 1./8.*ci**2)
      - c3p*(9./8. + 9./8.*ci**2)
      )


def H_plus_PN10(psi, iota, Delta, nu):
   """
   1.0 PN order contribution to h_+
   """
   si = np.sin(iota)
   ci = np.cos(iota)
   c2p = np.cos(2.*psi)
   c4p = np.cos(4.*psi)
   return (
      c2p*(
         19./6. 
         + 3./2.*ci**2 
         - 1./3.*ci**4 
         + nu*(-19./6. + 11./6.*ci**2 + ci**4)
         )
      - c4p*(
         4./3.*si**2*(1. + ci**2)*(1.-3*nu)
         )
      )


def H_plus_PN15(psi, iota, Delta, nu):
   """
   1.5 PN order contribution to h_+
   """
   si = np.sin(iota)
   ci = np.cos(iota)
   cp = np.cos(psi)
   c2p = np.cos(2.*psi)
   c3p = np.cos(3.*psi)
   c5p = np.cos(5.*psi)
   return (
      si*Delta*cp*(
         19./64. 
         + 5./16.*ci**2 
         - 1./192.*ci**4 
         + nu*(-49./96. + 1./8.*ci**2 + 1./96.*ci**4)
         )
      +c2p*(
         -2.*np.pi*(1. + ci**2)
         )
      + si*Delta*c3p*(
         -657./128. 
         - 45./16.*ci**2 
         + 81./128.*ci**4
         + nu*(225./64. - 9./8.*ci**2 - 81./64.*ci**4)
         )
      + si*Delta*c5p*(
         625./384.*si**2*(1. + ci**2)*(1.-2.*nu)
         )
      )


def H_plus_PN20(psi, iota, Delta, nu):
   """
   2.0 PN order contribution to h_+
   """
   si = np.sin(iota)
   ci = np.cos(iota)
   cp = np.cos(psi)
   c2p = np.cos(2.*psi)
   c3p = np.cos(3.*psi)
   c4p = np.cos(4.*psi)
   c6p = np.cos(6.*psi)
   sp = np.sin(psi)
   s3p = np.sin(3.*psi)
   return (
      np.pi*si*Delta*cp*(
         -5./8.
         - 1./8.*ci**2
         )
      + c2p*(
         11./60.
         + 33./10.*ci**2
         + 29./24.*ci**4
         - 1./24.*ci**6
         + nu*(353./36. - 3.*ci**2 - 251./72.*ci**4 + 5./24.*ci**6)
         + nu**2*(-49./12. + 9./2.*ci**2 - 7./24.*ci**4 - 5./24.*ci**6)
         )
      + np.pi*si*Delta*c3p*(
         27./8.*(1. + ci**2)
         )
      + 2./15.*si**2*c4p*(
         59. 
         + 35.*ci**2
         - 8.*ci**4
         - 5./3.*nu*(131. + 59.*ci**2 - 24.*ci**4)
         + 5.*nu**2*(21. - 3.*ci**2 - 8.*ci**4)
         )
      + c6p*(
         -81./40.*si**4*(1. + ci**2)*(1. - 5.*nu + 5.*nu**2)
         )
      + si*Delta*sp*(
         11./40.
         + 5.*np.log(2.)/4.
         + ci**2*(7./40. + np.log(2.)/4.)
         )
      + si*Delta*s3p*(
         (-189./40. + 27./4.*np.log(3./2.))*(1. + ci**2)
         )
      )


def H_plus_PN25(psi, iota, Delta, nu):
   """
   2.5 PN order contribution to h_+
   """
   si = np.sin(iota)
   ci = np.cos(iota)
   cp = np.cos(psi)
   c2p = np.cos(2.*psi)
   c3p = np.cos(3.*psi)
   c4p = np.cos(4.*psi)
   c5p = np.cos(5.*psi)
   c7p = np.cos(7.*psi)
   s2p = np.sin(2.*psi)
   s4p = np.sin(4.*psi)
   return (
      si*Delta*cp*(
         1771./5120.
         - 1667./5120.*ci**2
         + 217./9216.*ci**4
         - 1./9216.*ci**6
         + nu*(681./256. + 13./768.*ci**2 - 35./768.*ci**4 + 1./2304*ci**6)
         + nu**2*(-3451./9216. + 673./3072.*ci**2 - 5./9216.*ci**4 - 1./3072.*ci**6)
         )
      + np.pi*c2p*(
         19./3.
         + 3.*ci**2
         - 2./3.*ci**4
         + nu*(-16./3. + 14./3.*ci**2 + 2.*ci**4)
         )
      + si*Delta*c3p*(
         3537./1024.
         - 22977./5120.*ci**2
         - 15309./5120.*ci**4
         + 729./5120.*ci**6
         + nu*(-23829./1280. + 5529./1280.*ci**2 + 7749./1280.*ci**4 - 729./1280.*ci**6)
         + nu**2*(29127./5120. - 27267./5120*ci**2 - 1647./5120.*ci**4 + 2187./5120.*ci**6)
         )
      + c4p*(
         -16.*np.pi/3.*(1. + ci**2)*si**2*(1.-3.*nu)
         )
      + si*Delta*c5p*(
         -108125./9216.
         + 40625./9216.*ci**2
         + 83125./9216.*ci**4
         - 15625./9216.*ci**6
         + nu*(8125./256. - 40625./2304.*ci**2 - 48125./2304.*ci**4 + 15625./2304.*ci**6)
         + nu**2*(-119375./9216. + 40625./3072.*ci**2 + 44375./9216.*ci**4 - 15625./3072.*ci**6)
         )
      + Delta*c7p*(
         117649./46080.*si**5*(1. + ci**2)*(1. - 4.*nu + 3.*nu**2)
         )
      + s2p*(
         -9./5.
         + 14./5.*ci**2
         + 7./5.*ci**4
         + nu*(32. + 56./5.*ci**2 - 28./5.*ci**4)
         )
      + si**2*(1. + ci**2)*s4p*(
         56./5.
         - 32.*np.log(2.)/3.
         + nu*(-1193./30. + 32.*np.log(2.))
         )
      )


def H_plus_PN30(psi, iota, Delta, nu, x):
   """
   3.0 PN order contribution to h_+
   """
   si = np.sin(iota)
   ci = np.cos(iota)
   cp = np.cos(psi)
   c2p = np.cos(2.*psi)
   c3p = np.cos(3.*psi)
   c4p = np.cos(4.*psi)
   c5p = np.cos(5.*psi)
   c6p = np.cos(6.*psi)
   c8p = np.cos(8.*psi)
   sp = np.sin(psi)
   s2p = np.sin(2.*psi)
   s3p = np.sin(3.*psi)
   s5p = np.sin(5.*psi)
   l16x = np.log(16.*x)
   return (
      np.pi*Delta*si*cp*(
         19./64.
         + 5./16.*ci**2
         - 1./192.*ci**4
         + nu*(-19./96. + 3./16.*ci**2 + 1./96.*ci**4)
         )
      + c2p*(
         -465497./11025.
         + (856./105.*np.euler_gamma - 2.*np.pi**2/3. + 428./105.*l16x)*(1. + ci**2)
         - 3561541./88200.*ci**2
         - 943./720.*ci**4
         + 169./720.*ci**6
         - 1./360.*ci**8
         + nu*(2209./360. - 41.*np.pi**2/96.*(1. + ci**2) - 2039./180.*ci**2 + 3311./720.*ci**4 - 853./720.*ci**6 + 7./360.*ci**8)
         + nu**2*(12871./540. - 1583./60.*ci**2 - 145./108.*ci**4 + 56./45.*ci**6 - 7./180.*ci**8)
         + nu**3*(-3277./810. + 19661./3240.*ci**2 - 281./144.*ci**4 - 73./720.*ci**6 + 7./360.*ci**8)
         )
      + np.pi*Delta*si*c3p*(
         -1971./128.
         - 135./16.*ci**2
         + 243./128.*ci**4
         + nu*(567./64. - 81./16.*ci**2 - 243./64.*ci**4)
         )
      + si**2*c4p*(
         -2189./210.
         + 1123./210.*ci**2
         + 56./9.*ci**4
         - 16./45.*ci**6
         + nu*(6271./90. - 1969./90.*ci**2 - 1432./45.*ci**4 + 112./45.*ci**6)
         + nu**2*(-3007./27. + 3493./135.*ci**2 + 1568./45.*ci**4 - 224./45.*ci**6)
         + nu**3*(161./6. - 1921./90.*ci**2 - 184./45.*ci**4 + 112./45.*ci**6)
         )
      + Delta*c5p*(
         3125.*np.pi/384.*si**3*(1. + ci**2)*(1. - 2.*nu)
         )
      + si**4*c6p*(
         1377./80.
         + 891./80.*ci**2
         - 729./280.*ci**4
         + nu*(-7857./80. - 891./16.*ci**2 + 729./40.*ci**4)
         + nu**2*(567./4. + 567./10.*ci**2 - 729./20.*ci**4)
         * nu**3*(-729./16. - 243./80.*ci**2 + 729./40.*ci**4) 
         )
      + c8p*(
         -1024./315.*si**6*(1. + ci**2)*(1. - 7.*nu - 14.*nu**2 - 7.*nu**3)
         )
      + Delta*si*sp*(
         -2159./40320.
         - 19.*np.log(2.)/32.
         + (-95./224. - 5.*np.log(2.)/8.)*ci**2
         + (181./13440. + np.log(2.)/96.)*ci**4
         + nu*(1369./160. + 19.*np.log(2.)/48. + (-41./48. - 3.*np.log(2.)/8.)*ci**2 + (-313./480. - np.log(2.)/48.)*ci**4)
         )
      + s2p*(
         -428.*np.pi/105.*(1. + ci**2)
         )
      + Delta*si*s3p*(
         205119./8960.
         - 1971./64.*np.log(3./2.)
         + (1917./224. - 135./8.*np.log(3./2.))*ci**2
         + (-43983./8960. + 243./64.*np.log(3./2.))*ci**4
         + nu*(-54869./960. + 567./32.*np.log(3./2.) + (-923./80. - 81./8.*np.log(3./2.))*ci**2 + (41851./2880. - 243./32.*np.log(3./2.))*ci**4)
         )
      + Delta*si**3*(1. + ci**2)*s5p*(
         -113125./5376.
         + 3125./192.*np.log(5./2.)
         + nu*(17639./320. - 3125./96.*np.log(5./2.))
         )
      )


def H_cross_PN00(psi, iota):
   """
   0 PN order contribution to h_x
   """
   ci = np.cos(iota)
   s2p = np.sin(2.*psi)
   return -2.*ci*s2p


def H_cross_PN05(psi, iota, Delta):
   """
   0.5 PN order contribution to h_x
   """
   si = np.sin(iota)
   ci = np.cos(iota)
   sp = np.sin(psi)
   s3p = np.sin(3.*psi)
   return si*ci*Delta*(-3./4.*sp + 9./4.*s3p)


def H_cross_PN10(psi, iota, Delta, nu):
   """
   1.0 PN order contribution to h_x
   """
   si = np.sin(iota)
   ci = np.cos(iota)
   s2p = np.sin(2.*psi)
   s4p = np.sin(4.*psi)
   return (
      ci*s2p*(
         17./3.
         - 4./3.*ci**2
         + nu*(-13./3. + 4.*ci**2)
         )
      + ci*si**2*s4p*(
         -8./3.*(1. - 3.*nu)
         )
      )


def H_cross_PN15(psi, iota, Delta, nu):
   """
   1.5 PN order contribution to h_x
   """
   si = np.sin(iota)
   ci = np.cos(iota)
   sp = np.sin(psi)
   s2p = np.sin(2.*psi)
   s3p = np.sin(3.*psi)
   s5p = np.sin(5.*psi)
   return (
      si*ci*Delta*sp*(
         21./32. 
         - 5./96.*ci**2
         + nu*(-23./48. + 5./48.*ci**2)
         )
      - 4.*np.pi*ci*s2p
      + si*ci*Delta*s3p*(
         -603./64.
         + 135./64.*ci**2
         + nu*(171./32. - 135./32.*ci**2)
         )
      + si*ci*Delta*s5p*(
         625./192.*(1. - 2.*nu)*si**2
         )
      )


def H_cross_PN20(psi, iota, Delta, nu):
   """
   2.0 PN order contribution to h_x
   """
   si = np.sin(iota)
   ci = np.cos(iota)
   cp = np.cos(psi)
   c3p = np.cos(3.*psi)
   sp = np.sin(psi)
   s2p = np.sin(2.*psi)
   s3p = np.sin(3.*psi)
   s4p = np.sin(4.*psi)
   s6p = np.sin(6.*psi)
   return (
      si*ci*Delta*cp*(
         -9./20.
         - 3.*np.log(2.)/2.
         )
      + si*ci*Delta*c3p*(
         189./20.
         - 27.*np.log(3./2.)/2.
         )
      - si*ci*Delta*3.*np.pi/4.*sp
      + ci*s2p*(
         17./15.
         + 113./30.*ci**2
         - 1./4.*ci**4
         + nu*(143./9. - 245./18.*ci**2 + 5./4.*ci**4)
         + nu**2*(-14./3. + 35./6.*ci**2 - 5./4.*ci**4)
         )
      + si*ci*Delta*s3p*(
         27.*np.pi/4.
         )
      + 4./15.*ci*si**2*s4p*(
         55.
         - 12.*ci**2
         - 5./3.*nu*(119. - 36.*ci**2)
         + 5.*nu**2*(17. - 12.*ci**2)
         )
      + ci*s6p*(
         -81./20.*si**4*(1. - 5.*nu + 5.*nu**2)
         )
      )


def H_cross_PN25(psi, iota, Delta, nu):
   """
   2.5 PN order contribution to h_x
   """
   si = np.sin(iota)
   ci = np.cos(iota)
   c2p = np.cos(2.*psi)
   c4p = np.cos(4.*psi)
   sp = np.sin(psi)
   s2p = np.sin(2.*psi)
   s3p = np.sin(3.*psi)
   s4p = np.sin(4.*psi)
   s5p = np.sin(5.*psi)
   s7p = np.sin(7.*psi)
   #return (
   #   6./5.*si**2*ci*nu
   #   + ci*c2p*(
   #      2.
   #      - 22./5.*ci**2
   #      + nu*(-282./5. + 94./5.*ci**2)
   #      )
   #   + ci*si**2*c4p*(
   #      -112./5.
   #      + 64.*np.log(2.)/3.
   #      + nu*(1193./15. - 64.*np.log(2.))
   #      )
   #   + si*ci*Delta*sp*(
   #      -913./7680.
   #      + 1891./11520.*ci**2
   #      - 7./4608.*ci**4
   #      + nu*(1165./384. - 235./576.*ci**2 + 7./1152.*ci**4)
   #      + nu**2*(-1301./4608. + 301/2304.*ci**2 - 7./1536.*ci**4)
   #      )
   #   + np.pi*ci*s2p*(
   #      34./3.
   #      - 8./3.*ci**2
   #      + nu*(-20./3. + 8.*ci**2)
   #      )
   #   + si*ci*Delta*s3p*(
   #      12501./2560. 
   #      - 12069./1280.*ci**2 
   #      + 1701./2560.*ci**4
   #      + nu*(-19581./640. + 7821./320.*ci**2 - 1701./640.*ci**4)
   #      + nu**2*(18903./2560. - 11403./1280.*ci**2 + 5103./2560.*ci**4)
   #      )
   #   + si**2*ci*s4p*(
   #      -32.*np.pi/3.*(1. - 3.*nu)
   #      )
   #   + Delta*si*ci*s5p*(
   #      -101875./4608.
   #      + 6875./256.*ci**2
   #      - 21875./4608.*ci**4
   #      + nu*(66875./1152. - 44375./576.*ci**2 + 21875./1152.*ci**4)
   #      + nu**2*(-100625./4608. + 83125./2304.*ci**2 - 21875./1536.*ci**4)
   #      )
   #   + Delta*si**5*ci*s7p*(
   #      117649./23040.*(1. - 4.*nu + 3.*nu**2)
   #      )
   #   )
   # without the DC term
   return (
      ci*c2p*(
         2.
         - 22./5.*ci**2
         + nu*(-282./5. + 94./5.*ci**2)
         )
      + ci*si**2*c4p*(
         -112./5.
         + 64.*np.log(2.)/3.
         + nu*(1193./15. - 64.*np.log(2.))
         )
      + si*ci*Delta*sp*(
         -913./7680.
         + 1891./11520.*ci**2
         - 7./4608.*ci**4
         + nu*(1165./384. - 235./576.*ci**2 + 7./1152.*ci**4)
         + nu**2*(-1301./4608. + 301/2304.*ci**2 - 7./1536.*ci**4)
         )
      + np.pi*ci*s2p*(
         34./3.
         - 8./3.*ci**2
         + nu*(-20./3. + 8.*ci**2)
         )
      + si*ci*Delta*s3p*(
         12501./2560. 
         - 12069./1280.*ci**2 
         + 1701./2560.*ci**4
         + nu*(-19581./640. + 7821./320.*ci**2 - 1701./640.*ci**4)
         + nu**2*(18903./2560. - 11403./1280.*ci**2 + 5103./2560.*ci**4)
         )
      + si**2*ci*s4p*(
         -32.*np.pi/3.*(1. - 3.*nu)
         )
      + Delta*si*ci*s5p*(
         -101875./4608.
         + 6875./256.*ci**2
         - 21875./4608.*ci**4
         + nu*(66875./1152. - 44375./576.*ci**2 + 21875./1152.*ci**4)
         + nu**2*(-100625./4608. + 83125./2304.*ci**2 - 21875./1536.*ci**4)
         )
      + Delta*si**5*ci*s7p*(
         117649./23040.*(1. - 4.*nu + 3.*nu**2)
         )
      )


def H_cross_PN30(psi, iota, Delta, nu, x):
   """
   3.0 PN order contribution to h_x
   """
   si = np.sin(iota)
   ci = np.cos(iota)
   cp = np.cos(psi)
   c2p = np.cos(2.*psi)
   c3p = np.cos(3.*psi)
   c5p = np.cos(5.*psi)
   sp = np.sin(psi)
   s2p = np.sin(2.*psi)
   s3p = np.sin(3.*psi)
   s4p = np.sin(4.*psi)
   s5p = np.sin(5.*psi)
   s6p = np.sin(6.*psi)
   s8p = np.sin(8.*psi)
   l16x = np.log(16.*x)
   return (
      Delta*si*ci*cp*(
         11617./20160.
         + 21.*np.log(2.)/16.
         + (-251./2240. - 5.*np.log(2.)/48.)*ci**2
         + nu*(-2419./240. - 5.*np.log(2.)/24. + (727./240. + 5.*np.log(2.)/24.)*ci**2)
         )
      + ci*c2p*(
         856.*np.pi/105.
         )
      + Delta*si*ci*c3p*(
         -36801./896.
         + 1809.*np.log(3./2.)/32.
         + (65097./4480. - 405.*np.log(3./2.)/32.)*ci**2
         + nu*(28445./288. - 405.*np.log(3./2.)/16. + (-7137./160. + 405.*np.log(3./2.)/16.)*ci**2)
         )
      + Delta*si**3*ci*c5p*(
         113125./2688.
         - 3125*np.log(5./2.)/96.
         + nu*(-17639./160. + 3125.*np.log(5./2.)/48.)
         )
      + np.pi*Delta*si*ci*sp*(
         21./32.
         - 5./96.*ci**2
         + nu*(-5./48. + 5./48.*ci**2)
         )
      + ci*s2p*(
         -3620761./44100.
         + 1712.*np.euler_gamma/105.
         - 4.*np.pi**2/3.
         + 856./105.*l16x
         - 3413./1260.*ci**2
         + 2909./2520.*ci**4
         - 1./45.*ci**6
         + nu*(743./90. - 41.*np.pi**2/48. + 3391./180.*ci**2 - 2287./360.*ci**4 + 7./45.*ci**6)
         + nu**2*(7919./270. - 5426./135.*ci**2 + 382./45.*ci**4 - 14./45.*ci**6)
         + nu**3*(-6457./1620. + 1109./180.*ci**2 - 281./120.*ci**4 + 7./45.*ci**6)
         )
      + np.pi*Delta*si*ci*s3p*(
         -1809./64.
         + 405./64.*ci**2
         + nu*(405./32. - 405./32.*ci**2)
         )
      + si**2*ci*s4p*(
         -1781./105.
         + 1208./63.*ci**2
         - 64./45.*ci**4
         + nu*(5207./45. - 536./5.*ci**2 + 448./45.*ci**4)
         + nu**2*(-24838./135. + 2224./15.*ci**2 - 896./45.*ci**4)
         + nu**3*(1703./45. - 1976./45.*ci**2 + 448./45.*ci**4)
         )
      + Delta*s5p*(
         3125.*np.pi/192.*si**3*ci*(1. - 2.*nu)
         )
      + si**4*ci*s6p*(
         9153./280.
         - 243./35.*ci**2
         + nu*(-7371./40. + 243./5.*ci**2)
         + nu**2*(1296./5. - 486./5.*ci**2)
         + nu**3*(-3159./40. + 243./5.*ci**2)
         )
      + s8p*(-2048./315.*si**6*ci*(1. - 7.*nu + 14.*nu**2 - 7.*nu**3))
      )


def h_plus_PN(nu, mu, Delta, x, psi, iota, dL, PN_order):
   """
   Returns the plus-polarization waveform (strain) to a given PN order. 
   Inputs:
      nu -- dimensionless mass ratio of the binary
      mu -- reduced mass of the binary [Msol]
      Delta -- relative mass difference
      x -- PN parameter for the frequency
      psi -- auxiliary phase of the signal
      iota -- angle between the sky-localization and the orbital momentum of the binary
      dL -- luminosity distance to the binary [Mpc]
      PN_order -- order of PN expansion
   Note that x and psi can be vectors. If so, x and psi must have the same length   
   """
   # output to desired PN order
   out = H_plus_PN00(psi, iota)
   if PN_order > 0.4:
      out += x**0.5*H_plus_PN05(psi, iota, Delta)
   if PN_order > 0.9:
      out += x*H_plus_PN10(psi, iota, Delta, nu)
   if PN_order > 1.4:
      out += x**1.5*H_plus_PN15(psi, iota, Delta, nu)
   if PN_order > 1.9:
      out += x**2*H_plus_PN20(psi, iota, Delta, nu)
   if PN_order > 2.4:
      out += x**2.5*H_plus_PN25(psi, iota, Delta, nu)
   if PN_order > 2.9:
      out += x**3*H_plus_PN30(psi, iota, Delta, nu, x)
   return 2.*G_m3_Msol_s2*mu*x/c_m_s**2/dL*Mpc_per_m*out 


def h_cross_PN(nu, mu, Delta, x, psi, iota, dL, PN_order):
   """
   Returns the plus-polarization waveform (strain) to a given PN order. 
   Inputs:\
      nu -- dimensionless mass ratio of the binary
      mu -- reduced mass of the binary [Msol]
      Delta -- relative mass difference
      x -- PN parameter for the frequency
      psi -- auxiliary phase of the signal
      iota -- angle between the sky-localization and the orbital momentum of the binary
      dL -- luminosity distance to the binary [Mpc]
      PN_order -- order of PN expansion
   Note that x and psi can be vectors. If so, x and psi must have the same length   
   """
   # output to desired PN order
   out = H_cross_PN00(psi, iota)
   if PN_order > 0.4:
      out += x**0.5*H_cross_PN05(psi, iota, Delta)
   if PN_order > 0.9:
      out += x*H_cross_PN10(psi, iota, Delta, nu)
   if PN_order > 1.4:
      out += x**1.5*H_cross_PN15(psi, iota, Delta, nu)
   if PN_order > 1.9:
      out += x**2*H_cross_PN20(psi, iota, Delta, nu)
   if PN_order > 2.4:
      out += x**2.5*H_cross_PN25(psi, iota, Delta, nu)
   if PN_order > 2.9:
      out += x**3*H_cross_PN30(psi, iota, Delta, nu, x)
   return 2.*G_m3_Msol_s2*mu*x/c_m_s**2/dL*Mpc_per_m*out 


def get_hp_hc(
   time_vec_helio, 
   tc, 
   nu, 
   mu, 
   m, 
   Delta, 
   theta0, 
   iota, 
   dL, 
   PN_order_phase,
   PN_order_waveform
   ):
   """
   wrapper function for h_cross/plus_PN, 
   returns a tuple of h_plus, h_cross polarization waveforms for inputs:
      time_vec_helio -- array of time when signal is sample in a inertial 
                        reference frame [s]
      tc -- time of coalesence
      nu -- dimensionless mass ratio of the binary
      mu -- reduced mass of the binary [Msol]
      m -- total mass of the binary [Msol]
      Delta -- relative mass difference
      theta0 -- constant of integration fixed by initial conditions
      iota -- angle between the sky-localization and the orbital momentum 
              of the binary
      dL -- luminosity distance to the binary [Mpc]
      PN_order_phase -- order of PN expansion of phase
      PN_order_waveform -- order of PN expansion of waveform
   """
   theta_vec = theta(time_vec_helio, nu, m, tc=tc)
   x_vec = x_PN(theta_vec, nu, PN_order_phase)
   Omega0 = c_m_s**3/(G_m3_Msol_s2*m)*x_vec[0]**1.5
   psi_vec = psi_PN(theta_vec, nu, mu, m, PN_order_phase, theta0, Omega0)
   hp_vec = h_plus_PN(nu, mu, Delta, x_vec, psi_vec, iota, dL, PN_order_waveform)
   hc_vec = h_cross_PN(nu, mu, Delta, x_vec, psi_vec, iota, dL, PN_order_waveform)
   return hp_vec, hc_vec
