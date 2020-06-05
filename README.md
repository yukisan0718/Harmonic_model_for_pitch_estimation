Harmonic model for pitch estimation
====

## Overview
Implementation of a pitch (fundamental frequency) estimation algorithm using the harmonic summation model [1-2]. This parametric approach is known as a classical algorithm for pitch estimation.

The "Harmonic_model_for_pitch_estimation.py" can estimate pitch of audio segment by segment. The parameters such as "L_max" must be adjusted depending on the application.

## Requirement
soundfile 0.10.3

matplotlib 3.1.0

numpy 1.18.1

scipy 1.4.1


## Dataset preparation
You can apply this pitch estimation algorithm to any audio recording you want. An example of a running speech has been prepared for the demonstration.

## References
[1] M. Noll: 'Pitch Determination of Human Speech by the Harmonic Product Spectrum, the Harmonic Sum Spectrum, and a Maximum Likelihood Estimate', in Proceedings of the symposium on computer processing communications, pp.779–797, (1969)

[2] D. J. Hermes: 'Measurement of Pitch by Subharmonic Summation', The Journal of the Acoustical Society of America, Vol.83, No.1, pp.257–264, (1988)