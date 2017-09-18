#!/usr/bin/env python
"""
Plot click waveforms.

Usage: read 
    $ ./plot_clicks.py --h
"""

import os
import argparse
import textwrap
import numpy as np
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
import soundfile as sf

from click_analysis import CLICK_DURATION


MAX_SUBPLOTS = 30


def build_butter_highpass(cut, sr, order=4):
    nyq = 0.5 * sr
    cut = cut / nyq
    b, a = butter(order, cut, btype='highpass')
    return b, a


def plot_clicks(clicks, audio, sr, cutoff_freq, subplot_shape, filename):

    b, a = build_butter_highpass(cutoff_freq, sr)

    t = np.arange(CLICK_DURATION * sr * 3) / sr

    i = 0
    while i < clicks.size:
        
        fig, axarr = plt.subplots(*subplot_shape)
        fig.suptitle("{}\nClicks {}-{} / {}".format(
            filename, i+1, i+subplot_shape[0]*subplot_shape[1], clicks.size))

        for row_ind in range(subplot_shape[0]):
            for col_ind in range(subplot_shape[1]):

                c = clicks[i]

                start = int((c - CLICK_DURATION) * sr)
                end = start + t.size

                if cutoff_freq > 0:
                    chunk = filtfilt(b, a, audio[start:end])
                else:
                    chunk = audio[start:end]
                axarr[row_ind, col_ind].plot(t+c-CLICK_DURATION, chunk)
                axarr[row_ind, col_ind].grid()
                axarr[row_ind, col_ind].axvspan(c, c + CLICK_DURATION, alpha=0.5, color='g')

                i += 1

                if i == clicks.size:
                    plt.show()
                    raise SystemExit()

        plt.show()

    
if __name__ == "__main__":

    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                     description=textwrap.dedent("""
    Plot click waveforms. 
    Close a plot to display the following set of clicks."""))
    parser.add_argument("audio_file", help="Audio file.")
    parser.add_argument("click_file", help="Click file in csv format. Click times must be in col 0.")
    parser.add_argument("--delimiter", type=str, default=",", help="CSV field delimiter.")
    parser.add_argument("--channel", type=int, default=0, help="Audio channel.")
    parser.add_argument("--cutoff_freq", type=int, default="0", help="High-pass filter cutoff frequency.")
    parser.add_argument("--subplot_shape", type=int, nargs="+", default=[5,5],
                        help="Shape of the subplots (e.g. <5,5> will plot 5x5 clicks).")

    args = parser.parse_args()

    audio_file = args.audio_file
    click_file = args.click_file
    delimiter = args.delimiter
    channel = args.channel
    cutoff_freq = args.cutoff_freq
    subplot_shape = args.subplot_shape

    if len(subplot_shape) != 2 or subplot_shape[0] * subplot_shape[1] > MAX_SUBPLOTS:
        raise argparse.ArgumentTypeError("subplot_shape takes 2 values, and their product" +
                                         " must not exceed {}".format(MAX_SUBPLOTS))

    # parse file
    clicks = np.loadtxt(click_file, delimiter=delimiter)
    if clicks.size == 0:
        raise SystemExit("No data")
    clicks = clicks[:,0]

    audio, sr = sf.read(audio_file)
    audio = audio[:,channel]
    
    plot_clicks(clicks, audio, sr, cutoff_freq, subplot_shape, os.path.basename(audio_file))
