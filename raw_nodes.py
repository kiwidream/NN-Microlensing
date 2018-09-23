# -*- coding: utf-8 -*-
"""
MASTER NODE FILE
"""

import numpy as np
import matplotlib.pyplot as plt
from lightcurve import *
from scipy import signal

# EXCURSION INPUT NODE BEGINS
"""
Notes on excursion:
        - Generally speaking, below 0.5 indicates a negative excursion while
        above 0.5 indicates a positive excursion.
        - A sinusoidal data set (e.g. Cepheid data) results in excursions averaging
        around 0.5, thus 0.5 is not a significant indicator of a confirmed event.
        - Only a portion of positve results (for example, excursions above a threshold
        such as 0.65) will be significant enough to confirm the excursion.

"""
def excursion(data):
    times = data[:,0]
    mags = data[:,1]
    normalised_mags = []
    for mag in mags:
        z = (mag - min(mags)) / (max(mags) - min(mags))
        normalised_mags.append(z)
    normalised_mean = np.mean(normalised_mags)
    above = []
    below = []
    for mag in normalised_mags:
        if mag >= normalised_mean:
            above.append(mag)
        else:
            below.append(mag)
    above_range = (max(above) - min(above))
    below_range = (max(below) - min(below))

#        """Uncomment to test"""
#        if above_range > below_range:
#                print('positive')
#        else:
#                print('negative / none')
#        print("Above: %s" % above_range)
#        print("Below: %s" % below_range)

    return above_range, below_range
# EXCURSION INPUT NODE ENDS

# POWER SPECTRUM NODE BEGINS
def pspec(event):
    data, times = event.curve[:, event.CURVE_Y], event.curve[:, event.CURVE_X]

    rate = 30
    t = np.arange(times[0], times[-1], 1/rate)
    data -= np.mean(data)

    ps = np.log10(np.abs(np.fft.rfft(data)))

    freqs = np.linspace(0, rate/2, len(ps))
    ps -= np.mean(ps)
    ps = np.maximum(ps, 0)
    return np.average(freqs, weights=(ps*(freqs ** 2)))
# POWER SPECTRUM NODE ENDS

if __name__ == "__main__":
    data = np.genfromtxt('OGLE-LMC-CEP-0001.txt', delimiter=' ', dtype=float)
    print(excursion(data))

