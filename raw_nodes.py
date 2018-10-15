# -*- coding: utf-8 -*-
"""
MASTER NODE FILE
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import sys

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

    mag_min = min(mags)
    mag_max = max(mags)
    normalised_mags = (mags - mag_min) / (mag_max - mag_min)
    normalised_mean = np.mean(normalised_mags)
    above = []
    below = []
    for mag in normalised_mags:
        if mag >= normalised_mean:
            above.append(mag)
        else:
            below.append(mag)

    percentile = 99
    above_range = np.percentile(above, percentile) - np.percentile(above, 100 - percentile)
    below_range = np.percentile(below, percentile) - np.percentile(below, 100 - percentile)

#        """Uncomment to test"""
#        if above_range > below_range:
#                print('positive')
#        else:
#                print('negative / none')
#        print("Above: %s" % above_range)
#        print("Below: %s" % below_range)

    return above_range - below_range, above_range, below_range
# EXCURSION INPUT NODE ENDS

# POWER SPECTRUM NODE BEGINS
def pspec(data):
    times = data[:,0]
    data = data[:,1]

    rate = 5
    num_out = 1000
    periods = np.linspace(0.1, 5, 500)
    ang_freqs = 2 * np.pi / periods
    data_shift = data - np.mean(data)
    ps = signal.lombscargle(times, data_shift, ang_freqs, normalize=True)

    return np.max(ps), np.average(ang_freqs[np.argmax(ps)])
# POWER SPECTRUM NODE ENDS

if __name__ == "__main__":
    data = np.genfromtxt('OGLE-LMC-CEP-0001.txt', delimiter=' ', dtype=float)
    print(excursion(data))

