#!/usr/bin/env python
"""
Plot click IPI and xchannel delays.

Usage: read 
    $ ./plot_utils.py --h
"""

import sys
import argparse
import numpy as np

import matplotlib.pyplot as plt


def plot_ipi_xchannel_delay(data, delay_file, time_offset):
    # data row: click_time, click_confidence, click_amplitude, x-channel delay, ipi, ipi salience

    f, axarr = plt.subplots(3, sharex=True)

    title = delay_file if time_offset == 0.0 else "{}\ntime offset: {}".format(delay_file, time_offset)

    axarr[0].set_title(title)
    t = np.asarray([c[0] for c in data]) + time_offset
    d = np.asarray([c[3] for c in data]) * 1000
    axarr[0].scatter(t, d, marker="x", c="b")
    ylim = np.max(np.abs(data[:,3])) * 1000
    ylim += ylim / 10
    axarr[0].set_ylim(-ylim, ylim)    
    axarr[0].grid()
    axarr[0].set_ylabel('x-channel delay (ms)')

    t = np.asarray([c[0] for c in data]) + time_offset
    ipi = np.asarray([c[4] for c in data]) * 1000
    axarr[1].scatter(t, ipi, marker="x", c="b")
    ylim = np.max(data[:,4]) * 1000
    ylim += ylim / 10
    axarr[1].set_ylim(1, ylim)
    axarr[1].grid()
    axarr[1].set_xlabel('Time (s)')
    axarr[1].set_ylabel('IPI (ms)')
    
    t = np.asarray([c[0] for c in data]) + time_offset
    ipi = np.asarray([c[2] for c in data])
    axarr[2].scatter(t, ipi, marker="x", c="b")
    ylim = np.max(data[:,2])
    ylim += ylim / 10
    axarr[2].set_ylim(0, ylim)
    axarr[2].grid()
    axarr[2].set_xlabel('Time (s)')
    axarr[2].set_ylabel('Click amplitude')

#    plt.tight_layout()

    plt.show()

    
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="""Plot click IPIs and xchannel delays.""")
    parser.add_argument("delay_file", help="Delay + IPI file.")
    parser.add_argument("--time_offset", type=float, default=0.0, help="Time offset.")
    parser.add_argument("--min_amp", type=float, default=0.0, help="Min click amplitude.")

    args = parser.parse_args()

    delay_file = args.delay_file
    time_offset = args.time_offset
    min_amp = args.min_amp

    # parse file
    data = np.loadtxt(delay_file, delimiter=',')
    if min_amp:
        data = data[data[:,2]>min_amp]

    if data.size == 0:
        sys.exit("No data")
    
    plot_ipi_xchannel_delay(data, delay_file, time_offset)
