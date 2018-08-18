import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import sys
import os
import numpy as np
import random
import time
from lightcurve import *
from network import Network
import matplotlib.patheffects as PathEffects
#from scipy import optimize

plot = None

def draw_plot(event):
    global plot, ax1, eline
    time_list, mag_list, sigma, no_noise_time, no_noise = np.split(event.curve, 5, 1)

    if plot is None:
        plot, = ax1.plot(no_noise_time,no_noise, color=(0,0,0,0.3))
        eline = ax1.errorbar(time_list,mag_list,sigma,color='r',fmt='.')
        plt.ion()

        ax1.set_xlabel("Time, t")
        ax1.set_ylabel("Magnification, A")

    else:
        plot.set_data(no_noise_time, no_noise)
        eline[0].remove()
        for line in eline[1]:
            line.remove()
        for line in eline[2]:
            line.remove()
        eline = ax1.errorbar(time_list,mag_list,sigma,color='r',fmt='.')
        ax1.relim()
        ax1.autoscale_view(True,True,True)
        plt.draw()

    ax1.set_title(event.name)
    return event

def draw_neural_net(ax, left, right, bottom, top, nn, activations, layer_text=None):
    '''
    Draw a neural network cartoon using matplotilb.

    :usage:
        >>> fig = plt.figure(figsize=(12, 12))
        >>> draw_neural_net(fig.gca(), .1, .9, .1, .9, [4, 7, 2], ['x1', 'x2','x3','x4'])

    :parameters:
        - ax : matplotlib.axes.AxesSubplot
            The axes on which to plot the cartoon (get e.g. by plt.gca())
        - left : float
            The center of the leftmost node(s) will be placed here
        - right : float
            The center of the rightmost node(s) will be placed here
        - bottom : float
            The center of the bottommost node(s) will be placed here
        - top : float
            The center of the topmost node(s) will be placed here
        - layer_sizes : list of int
            List of layer sizes, including input and output dimensionality
        - layer_text : list of str
            List of node annotations in top-down left-right order
    '''
    text = layer_text[:]
    layer_sizes = nn.sizes
    weights = nn.weights

    n_layers = len(layer_sizes)
    v_spacing = (top - bottom)/float(max(layer_sizes))
    h_spacing = (right - left)/float(len(layer_sizes) - 1)
    ax.axis('off')
    act_flatten = []
    for n in range(len(layer_sizes)):
        for m in range(layer_sizes[n]):
            act_flatten.append(activations[n][m])
    max_act = max(act_flatten)

    # Nodes
    for n, layer_size in enumerate(layer_sizes):
        layer_top = v_spacing*(layer_size - 1)/2. + (top + bottom)/2.
        for m in range(layer_size):
            x = n*h_spacing + left
            y = layer_top - m*v_spacing
            col = "%x" % int((activations[n][m] / max_act) * 15)
            circle = plt.Circle((x,y), v_spacing/4.,
                                color='#'+col*6+'FF', ec='k', zorder=0)
            ax.add_artist(circle)

            # Node annotations
            if text:
                string = text.pop(0)
                txt = ax.text(x, y, string, ha='center', va='center')
                txt.set_path_effects([PathEffects.withStroke(linewidth=5, foreground='w')])

    green = "55BB55FF"
    red = "BB5555FF"

    # Edges
    for n, (layer_size_a, layer_size_b) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        layer_top_a = v_spacing*(layer_size_a - 1)/2. + (top + bottom)/2.
        layer_top_b = v_spacing*(layer_size_b - 1)/2. + (top + bottom)/2.
        for m in range(layer_size_a):
            for o in range(layer_size_b):
                weight = weights[n][o][m]
                col = green if weight > 0 else red
                line = plt.Line2D([n*h_spacing + left, (n + 1)*h_spacing + left],
                                  [layer_top_a - m*v_spacing, layer_top_b - o*v_spacing], c='#'+col, zorder=-1, linewidth=min(abs(weight), 5))
                ax.add_artist(line)

def main():
    """ Executes the required calculations for an event, prints raw data and creates a graph.
    """
    global ax1

    fig = plt.figure(figsize=(7, 7))
    fig2 = plt.figure(figsize=(7, 7))
    ax1 = fig.add_subplot(111)
    plt.ion()
    network_size = [LightCurve.INPUT_SIZE, 5, 5, LightCurve.OUTPUT_SIZE]
    nn = Network(network_size)
    recent_progress = []
    types = [MicroLensing, NonEvent]
    labels = ['x'+str(i) for i in range(LightCurve.INPUT_SIZE)]
    labels += ["" for i in range(sum(network_size[1:-1]))]
    labels += [ev().__class__.__name__ for ev in types]
    while True:
        training_data = []

        for i in range(150):
            event = types[random.randint(0, len(types)-1)]()
            event.generate_curve()
            x, y = event.calculate_inputs(), event.expected_outputs()
            training_data.append((x, y))

        draw_plot(event)

        nn.SGD(np.array(training_data),10,50,2,monitor_training_accuracy=True)
        sys.stdout.flush()

        ax2 = fig2.gca()

        activations = nn.activations(x)
        draw_neural_net(ax2, .05, .95, .05, .95, nn, activations, labels)

        plt.show()
        fig.canvas.draw()
        fig.canvas.flush_events()
        fig2.clf()



if __name__ == "__main__":
    main()
