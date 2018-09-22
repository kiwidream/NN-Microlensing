# -*- coding: utf-8 -*-
"""
MASTER NODE FILE
"""

import numpy as np
import matplotlib.pyplot as plt
import generate
from lightcurve import *

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

        return above_range - below_range
# EXCURSION INPUT NODE ENDS
        
# POWER SPECTRUM NODE BEGINS    
def pspec():
    event = Lightcurve()
    event.generate_curve()
    data, times = event.interpolate_smooth()
    d_list = []
    if len(times) % 2 == 0:
        np.delete(times, 0)
    for i in range(len(times)-1):
        d_list.append(times[i+1] - times[i])
    time_step = np.average(d_list)
    
    ps = np.abs(np.fft.fft(data))**2
    
    freqs = np.fft.fftfreq(data.size, time_step)
    
    ps = ps[0:int(len(ps)/2)]
    freqs = freqs[0:int(len(freqs)/2)]    
    
    idx = np.argsort(freqs)
    
    average_freq = np.average(freqs[idx], weights=ps[idx])

    return average_freq
# POWER SPECTRUM NODE ENDS

if __name__ == "__main__":
        data = np.genfromtxt('OGLE-LMC-CEP-0001.txt', delimiter=' ', dtype=float)
        print(excursion(data))

