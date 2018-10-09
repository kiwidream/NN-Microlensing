import matplotlib.pyplot as plt
import sys
import os
import numpy as np
import random
import time
from lightcurve import *
from network import Network, load
from raw_nodes import pspec, excursion
import datetime

plot = None

def pspec_test():
  lightcurves = []
  types = [MicroLensing, Periodic, NonEvent]
  avs = [[] for _ in range(6)]
  for i in range(250):
    for j in range(len(types)):
      event = types[j]()
      event.generate_curve()
      a, b = pspec(event.curve[:, :2])
      avs[j*2].append(a)
      avs[j*2+1].append(b)

  for j in range(len(types)):
    print(types[j])
    print(np.mean(avs[j*2]), np.mean(avs[j*2+1]))


def draw_plot(event):
  fig = plt.figure()
  ax = fig.add_subplot(111)

  plot, = ax.plot(event.curve[:, 0], event.curve[:, 1], color=(0,0,0,0.3))
  eline = ax.errorbar(event.curve[:, 0], event.curve[:, 1], event.curve[:, 2],color='r',fmt='.')
  plt.ion()
  plt.title("LightCurve")
  
  plt.xlabel("Time, t")
  plt.ylabel("Flux, f")
  
def main():
  args = sys.argv

  if len(args) <= 2 or not args[1].isdigit():
    print("Usage: python validate.py EXAMPLE_ID NETWORK_FILE")
    exit()

  ex_id = args[1]
  directory = 'validation/'+ex_id+'/'
  dates = np.loadtxt(directory+'dates')

  flux = np.load(directory+'flux.npy')
  flux_err = np.load(directory+'flux_err.npy')

  #event = LightCurve()
  #curve_of_interest = 4859
  #event.load_curve(dates, flux[curve_of_interest], flux_err[curve_of_interest])
  #return draw_plot(event)

  num_microlensing = 0
  ml_events = []
  network = load(args[2])

  for i in range(len(flux)):
    lightcurve = LightCurve()
    lightcurve.load_curve(dates, flux[i], flux_err[i])
    inputs = lightcurve.calculate_inputs()
    activations = network.activations(inputs)
    if np.argmax(activations[3]) == 0:
      num_microlensing += 1
      ml_events.append((i, activations[3][0][0]))
    percent = round(100 * num_microlensing / (i+1), 2)
    if i == len(flux) - 1:
      print("\r Validation complete, result: Error "+str(percent)+"% ("+str(num_microlensing)+" / "+str(i+1)+")")
      print("Index, Activation")
      for ind, act in sorted(ml_events, key=lambda x: x[1]):
        print(str(ind)+", "+str(round(act, 2)))
    else:
      print(" "+str(num_microlensing)+" / "+str(i+1)+" (Error "+str(percent)+"%)", end="\r")
    sys.stdout.flush()



if __name__ == "__main__":
  main()
