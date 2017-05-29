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


def plot_ipi_xchannel_delay(data):
    # data row: click_time, click_strength, x-channel delay, ipi, ipi salience

    f, axarr = plt.subplots(2, sharex=True)

    axarr[0].set_title(delay_file)
    t = np.asarray([c[0] for c in data])
    d = np.asarray([c[2] for c in data]) * 1000
    axarr[0].scatter(t, d, marker="x", c="b")
    ylim = np.max(np.abs(data[:,2])) * 1000
    ylim += ylim / 10
    axarr[0].set_ylim(-ylim, ylim)    
    axarr[0].grid()
    axarr[0].set_ylabel('x-channel delay (ms)')

    t = np.asarray([c[0] for c in data])
    ipi = np.asarray([c[3] for c in data]) * 1000
    axarr[1].scatter(t, ipi, marker="x", c="b")
    ylim = np.max(data[:,3]) * 1000
    ylim += ylim / 10
    axarr[1].set_ylim(1, ylim)
    axarr[1].grid()
    axarr[1].set_xlabel('Time (s)')
    axarr[1].set_ylabel('IPI (ms)')

    plt.tight_layout()

    plt.show()

    
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="""Plot click IPIs and xchannel delays.""")
    parser.add_argument("delay_file", help="Delay + IPI file.")
    args = parser.parse_args()

    delay_file = args.delay_file

    # parse file
    data = np.loadtxt(delay_file, delimiter=',')

    if data.size == 0:
        sys.exit("No data")

    plot_ipi_xchannel_delay(data)
