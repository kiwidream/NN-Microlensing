import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import sys
import os
import numpy as np
import random
import time
#from scipy import optimize

plot = None 

def total_magnification(u):
    """ Calculates the total magnification 'a' for a given 'u' value. """
    return ((u**2) + 2) / (u * ((u**2) + 4)**(1/2))

def rel_lense_motion(uo, t, tE, to):
    """ Calculates the relative lens motion 'u' from the time and closest separation
    parameters. """
    return ( (uo**2) + (x_axis_value(t, to, tE)**2) )** (1/2)

def x_axis_value(t, to, tE):
    """Calculates an alternative x axis value."""
    return (t - to) / tE

def generate_values():
    """ Plots and displays a graph of magnitude and time. """
    time_list = []
    mag_list = []  

    plot = None
    #uo (the time of greatest magnification): values between 0 and 1.5
    #tE (the time to cross the Einstien Radius): values in increments of 5
    #t (the observation time) : values from 1 to 100
    #e_time (controls x axis values) : True or False
    uo = random.uniform(0, 1.5)
    tE = random.uniform(2, 30)
    to = 7668.97
    percent_noise = 0.04
    shift_range = 0.03 * to
    shift = int(random.uniform(-shift_range/2, shift_range/2))

    for t in range(int(to*0.99)*2-shift, int(to*1.01)*2-shift):
        t /= 2
        if random.randint(0, 5) == 0:
            continue
        u = rel_lense_motion(uo, t, tE, to)
        A = total_magnification(u)
        time_list.append(t)
        mag_list.append(A)
    
    max_mag = max(mag_list)
    for i in range(len(mag_list)):
        mag_list[i] += max_mag * random.uniform((1-percent_noise), (1+percent_noise))
        
    #Following code responsible for data point prints in shell
    print("time (t):", t)
    print("magnification:", A)
    print()
        
    return time_list, mag_list

def draw_plot(event):
    global plot, ax
    time_list, mag_list = generate_values()
    
    if plot is None:
        plot, = ax.plot(time_list, mag_list)
        plt.ion()
        plt.title("Paczynski Curve of a Gravitational Microlensing Event")
    
        plt.xlabel("Time, t")
        plt.ylabel("Magnification, A")      
    else:
        plot.set_data(time_list, mag_list)
        ax.relim()
        ax.autoscale_view(True,True,True)        
        plt.draw()
        
    return plot


def magnification(params, t):
    uo, tE, to = params
    return total_magnification(rel_lense_motion(uo, t, tE, to))
    
    
def chi2(params):
    global x, y, sigma
    i = magnification(params, x)
    return np.sum(((y-i)**2)/(sigma**2))

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
