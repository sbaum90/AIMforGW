import numpy as np
import time
import sys

###################################################
# helper functions
###################################################

def window_planck(N, eps):
   """
   returns a Planck window functions as described in arXiv:1003.2939
   inputs:
      N -- length of window function
      eps -- epsilon [0,1] controlling the turn-on of the window function
   """
   itaper = int(eps*N)
   n = np.linspace(0, N-1, int(N))
   out = np.ones(N)
   if itaper > 0:
      taper = 1./(np.exp(itaper/n[:itaper] + itaper/(n[:itaper]-itaper))+1.)
      out[:itaper] = taper
      out[-itaper:] = taper[::-1]
   return out


def inner_product_SNR(htilde, noise, freqs):
   integrand = np.real(htilde*np.conj(htilde))/noise
   return 4.*np.trapz(integrand, x=freqs)


def inner_product_FisherM(
   htilderef, 
   htilde1, 
   htilde2, 
   Delta1, 
   Delta2, 
   noise, 
   freqs
   ):
   integrand = np.real(
      (htilde1-htilderef)
      * np.conj(htilde2-htilderef)
      ) / (Delta1*Delta2*noise)
   return 4.*np.trapz(integrand, x=freqs)


def track_mp_progress(job, mp_inputs, starttime, update_interval=10*60):
   # function to track multiprocessing progress
   sleep_time = 10
   my_t0 = time.time()
   while job._number_left > 0:
      time.sleep(sleep_time)
      if time.time()-my_t0 > update_interval:
         print(
            str(int(100 * job._number_left * job._chunksize / len(mp_inputs))),
            "% of time-domain chunks remain to be computed; runtime:",
            str(int((time.time() - starttime)/60)),
            "min"
            )
         my_t0 = time.time()
         sys.stdout.flush()
