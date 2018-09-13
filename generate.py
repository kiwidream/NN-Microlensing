import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import sys
import os
import numpy as np
import random
import time
from lightcurve import *
#from scipy import optimize

plot = None

def draw_plot(event):
    global plot, ax, eline
    event = Periodic()
    event.generate_curve()
    time_list, mag_list, sigma, no_noise_time, no_noise = np.split(event.curve, 5, 1)

    if plot is None:
        plot, = ax.plot(no_noise_time,no_noise, color=(0,0,0,0.3))
        eline = ax.errorbar(time_list,mag_list,sigma,color='r',fmt='.')
        plt.ion()
        plt.title(event.name)

        plt.xlabel("Time, t")
        plt.ylabel("Magnification, A")
    else:
        plot.set_data(no_noise_time, no_noise)
        eline[0].remove()
        for line in eline[1]:
            line.remove()
        for line in eline[2]:
            line.remove()
        eline = ax.errorbar(time_list,mag_list,sigma,color='r',fmt='.')
        ax.relim()
        ax.autoscale_view(True,True,True)
        plt.draw()

    return plot

def main():
    """ Executes the required calculations for an event, prints raw data and creates a graph.
    """
    global ax
    fig = plt.figure()
    ax = fig.add_subplot(111)
    draw_plot(None)
    b_ax = plt.axes([0.85, 0.05, 0.1, 0.075])
    bnext = Button(b_ax, 'Draw')
    fig.canvas.mpl_connect('button_press_event', draw_plot)

if __name__ == "__main__":
    main()
