# AIMforGW
Atom Interferometer detectors in the Mid-band for Gravitational Waves: fisher-matrix parameter reconstruction forecast

If you use **AIMforGW** for your work, please cite [arXiv:2309.XXXX](https://arxiv.org/abs/2309.XXXXX).

AIMforGW is a Fisher-matrix forecast code for parameter reconstruction of gravitational wave (GW) signals from compact binaries generated during the inspiral phase with single-baseline (e.g., atom-interferometry based) detectors.

A short description of the modules can be found below. 
The basic useage is as follows:
- Select the version of the code you would like to run (`ParamEstimator_SpaceAI.py`,  `ParamEstimator_GroundAI.py`, `ParamEstimator_LO_SpaceAI.py`,  `ParamEstimator_LO_GroundAI.py`) and load the module into python3.
- Create an instance of the `GW_event` class with the GW source and detector parameters as inputs, see the description in each script
- You can then for example calculate the signal-to-noise ratio and the Fisher Matrix for that `GW_event` instance by running `GW_event.get_SNR_FisherMatrix()`.
- Similarly, the co-variance matrix and the angular resolution for the instance with and without including priors on periodic parameters (the angular variables describing the position of the GW source and the orientation of the binary's angular momentum) can be generated by calling `GW_event.get_CoVaMat()`, `GW_event.get_CoVaMat_priors()`, `GW_event.get_angular_resolution()`, and `GW_event.get_angular_resolution_priors()`.

Examples of scripts showing how to set up these input parameters, run the code, and save the output as pickle files are included, see the `Example_*.py` files. Each example can be executed by calling `python3 EXAMPLE.py`. The output generated by each of these examples can be found in the "ExamplesOutput" folder. A jupyter notebook showing how to load the output stored in these pickle files is also included, see `Example_results.ipynb`.

### Description of the Code:
The basic approach of the code is as follows:
- Generate polarization-basis time-domain waveforms using the post-newtonian waveforms (up to 3.5/3.0 PN order frequency evolution/amplitude correction) for non-spinning, non-precessing binaries.
- Generate time-dependent antenna functions as appropriate for a (network of) vertical terrestrial atom-interferometer-based detector(s) or a satellite-borne detector in geocentric orbit.
- Compute the time-domain strain in the detector(s), Fourier transform to the frequency domain.
- Compute the signal-to-noise ratio and the Fisher matrix (using single-sided first-order finite difference derivatives).

## Short description of the modules in **AIMforGW**:
# Main Modules:
- **ParamEstimator_SpaceAI.py:** Main wrapper for parameter estimation forecasts for a satellite-borne detector, using Post-Newtonian (up to 3.5/3.0 PN) inspiral waveforms.
- **ParamEstimator_GroundAI.py:** Main wrapper for parameter estimation forecasts for a network of km-size terrestrial detectors, using Post-Newtonian (up to 3.5/3.0 PN) inspiral waveforms.
- **ParamEstimator_LO_SpaceAI.py:** Same as SpaceAI_paramEstimator.py, but using leading-order (phase evolution and polarisation waveforms) waveforms.
- **ParamEstimator_LO_GroundAI.py:** Same as GroundAI_paramEstimator.py, but using leading-order (phase evolution and polarisation waveforms) waveforms.

# Auxiliary modules:
- **waveform_PN.py:** Functions for computing the inspiral waveforms in the Post-Newtonian expansion up to 3.5/3.0 PN, following Blanchet [arXiv:1310.1528](https://arxiv.org/abs/1310.1528).
- **waveform_LO.py:** Functions for computing leading-order inspiral waveforms, following [Maggiore 2008](https://oxford.universitypressscholarship.com/view/10.1093/acprof:oso/9780198570745.001.0001/acprof-9780198570745), chapter 4.1.
- **antennaFuns_satellites.py:** Functions to compute the antenna functions for a space-based AI GW detector. These antenna functions assume that the baseline is formed by two satellites orbiting Earth in a formation where one satellite trails the other on the same orbit.
- **antennaFuns_ground.py:** Functions to compute the antenna functions for ground-based AI GW detectors. These antenna functions assume that the baseline of a given detector is vertical and at the surface of the Earth.
- **helper_funs.py:** Some functions useful for the main wrappers.

# Noise Curve data:
- **NoiseCurve_Space.dat:** Noise curve for a satellite-borned detector. This is the envelope of the sensitivity in the resonant detector mode (the brown line) from Fig. 1 of [arXiv:1711.02225](https://arxiv.org/abs/1711.02225)
- **NoiseCurve_Groundkm.dat:** Noise curve for a km-version of a terrestrial GW detector. This is based on Fig. 1 of [arXiv:2104.02835](https://arxiv.org/abs/2104.02835), including the gravity gradient noise contribution

# Examples:
- **Example_SpaceAI.py**: Example script showing how to enter parameters and run the code for `ParamEstimator_SpaceAI.py`. An example of the output generated by this script (stored as a pickle object) can be found in **ExampleOutput/results_Example_SpaceAI**.
- **Example_GroundAI.py**: Example script showing how to enter parameters and run the code for `ParamEstimator_GroundAI.py`. An example of the output generated by this script (stored as a pickle object) can be found in **ExampleOutput/results_Example_GroundAI**.
- **Example_LO_SpaceAI.py**: Example script showing how to enter parameters and run the code for `ParamEstimator_LO_SpaceAI.py`. An example of the output generated by this script (stored as a pickle object) can be found in **ExampleOutput/results_Example_LO_SpaceAI**.
- **Example_LO_GroundAI.py**: Example script showing how to enter parameters and run the code for `ParamEstimator_LO_GroundAI.py`. An example of the output generated by this script (stored as a pickle object) can be found in **ExampleOutput/results_Example_LO_GroundAI**.
- **Example_results.ipynb**: Jupyter notebook showing how to load the the output generated from these examples and how to access the results.
