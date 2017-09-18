#!/usr/bin/env python
"""        
Inter Pulse Interval and x-channel delay measure.

Usage: read
    $ ./ipi_xchanneldelay_extractor.py --h
"""

import os
import sys
import logging
import argparse
from argparse import RawDescriptionHelpFormatter
import textwrap

import soundfile as sf
import numpy as np
from scipy.signal import butter, filtfilt
import git

import spectral_features


CLICK_DURATION = 0.002 # estimated click_duration
CLIPPING_THRESHOLD = 0.99


def build_butter_highpass(cut, sr, order=4):
    nyq = 0.5 * sr
    cut = cut / nyq
    b, a = butter(order, cut, btype='highpass')
    return b, a


def next_power_of_two(x):
    return 1<<(x-1).bit_length()


def get_ipi(click, ipi_min, ipi_max, sr, threshold):
    """Compute IPI from autocorrelation peak in the IPI range
    """

    # Compute autocorrelation of click
    acorr = np.abs(np.correlate(click, click, "full"))

    # Compute ratio between autocorrelation value at 0 delay and
    # highest value in the [ipi_min,ipi_max] autocorrelation range

    ac_d0 = np.argmax(acorr) # max at 0
    ac_v0 = acorr[ac_d0]

    ac_win_start = int(ac_d0 + ipi_min * sr)
    ac_win_end = int(ac_d0 + ipi_max * sr)
    peak_ind = ac_win_start + np.argmax(acorr[ac_win_start:ac_win_end])
    peak_value = acorr[peak_ind]

    # If the ratio is greater than a threshold, then we consider it an IPI
    salience = peak_value / ac_v0
    if  salience > threshold:
        return (peak_ind - ac_d0) / float(sr), salience
    else:
        return None, None


def get_tdoa(ch1, ch2, delay_max):

    xcorr = np.abs(np.correlate(ch1, ch2, "same"))
    delay0 = int(len(xcorr) / 2)
    delay_max_samples = int(delay_max * sr)
    delay = (delay_max_samples - np.argmax(xcorr[delay0-delay_max_samples:delay0+delay_max_samples])) / float(sr)

    return delay


if __name__ == "__main__":

    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                     description=textwrap.dedent('''
    Inter Pulse Interval detection and cross-channel delay measure.
    The algorithm is the following:
    For every click in the click file, in a window (default 0.01s) around the click: 
        - If check_clipping is set to True, check that the signal is not clipping
        - Filter both channels with highpass filter (default cutoff frequency = 10000 Hz).
        - In channel 1, check if the autocorrelation of the click has a peak in the [ipi_min, ipi_max]
          range (in second) with amplitude higher than a threshold
        - If yes compute the cross-correlation between channels 1 and 2. The position of the peak gives the delay between both channels.

    The format of each line in the output file is:
    
    <click time>, <click confidence>, <click amplitude>, <xcorr delay>, <acorr ipi delay>, <acorr value at 0 delay>, <acorr value at ipi delay>
    '''))
    parser.add_argument(
        '-v', "--verbose",
        help="Set verbose output", action="store_const", const=logging.DEBUG,
        dest="loglevel", default=logging.INFO)
    parser.add_argument("audio_file", help="Audio file.")
    parser.add_argument("click_file", help="Click file.")
    parser.add_argument("output_file", help="IPI and delay file.")
    parser.add_argument("--ipi_min", type=float, default=0.0015, help="Minimum IPI to be detected, in s.")
    parser.add_argument("--ipi_max", type=float, default=0.008, help="Maximum IPI to be detected, in s.")
    parser.add_argument("--cutoff_freq", type=int, default=100, help="""Cut-off frequency of the high-pass filter, in Hz.""")
    parser.add_argument("--delay_max", type=float, default=0.0015, help="Maximum cross-channel delay for a click, in s.")
    parser.add_argument("--min_salience", type=float, default=0.08, help="Min ratio between pulse autocorr value and max autocorr value.")
    parser.add_argument("--channels", type=int, nargs="+", default=[0, 1], help="""Respectively channels 1 and 2 in the algorithm.""")
    parser.add_argument("--check_clipping", type=int, default=1, help="Check if the signal is clipping.")
    args = parser.parse_args()

    logging.getLogger().setLevel(args.loglevel)

    audio_file = args.audio_file
    click_file = args.click_file
    output_file = args.output_file
    ipi_min = args.ipi_min
    ipi_max = args.ipi_max
    cutoff_freq = args.cutoff_freq
    delay_max = args.delay_max
    min_salience = args.min_salience
    channels = args.channels
    check_clipping = args.check_clipping

    #############################
    # open and parse click file #
    #############################

    clicks = np.loadtxt(click_file, delimiter=',')
    clicks = np.atleast_2d(clicks)

    ###################   
    # open audio file #
    ###################   

    audio, sr = sf.read(audio_file)

    if not audio.shape[1] > 1:
        raise Exception("File must have at least 2 channels")

    ###########
    # process #
    ###########

    with open(output_file, "w") as f:

        repo = git.Repo(search_parent_directories=True)
        sha = repo.head.object.hexsha
        f.write("#{}\n#Commit {}\n#Parameters: {}\n".format(__file__, sha, args))

        if clicks.size == 0:
            sys.exit()
    
        b, a = build_butter_highpass(cutoff_freq, sr)

        for t, v in clicks:
            
            # Get needed audio chunk around the click
            win_start_ind = int((t - delay_max) * sr)
            win_end_ind = int((t + delay_max + ipi_max + CLICK_DURATION)* sr)

            # If the chunk clips during more than CLICK_DURATION,
            # the click is removed.
            if (check_clipping and
                    (np.abs(audio[win_start_ind:win_end_ind, channels[0]]) > CLIPPING_THRESHOLD).sum() > CLICK_DURATION * sr):
                continue


            # Highpass filter
            ch1 = filtfilt(b, a, audio[win_start_ind:win_end_ind, channels[0]])
            ch2 = filtfilt(b, a, audio[win_start_ind:win_end_ind, channels[1]])

            # Get IPI on ch1
            ipi, salience = get_ipi(
                ch1[int(delay_max*sr):int((delay_max+ipi_max+CLICK_DURATION)*sr)],
                ipi_min,
                ipi_max,
                sr,
                min_salience)

            if ipi:

                # compute cross-channel delay from xcorrelation peak
                tdoa = get_tdoa(ch1, ch2, delay_max)

                # get max signal value
                max_ind = np.argmax(np.abs(ch1))
                max_value = min(1, np.abs(ch1[max_ind])) # filtered signal may have values > 1

                # get click chunk
                click = ch1[int(delay_max*sr):int((delay_max+CLICK_DURATION)*sr)]

                # compute spectral features
                spec = np.abs(np.fft.rfft(click, n=next_power_of_two(len(click))))
                freq_bin = sr / 2 / len(spec)
                spec_centroid = int(spectral_features.centroid(spec) * freq_bin)
                spec_max = int(np.argmax(spec) * freq_bin)

                f.write("{:.3f},{:.3f},{:.3f},{:.6f},{:.6f},{:.3f},{},{}\n".format(
                    t, v, max_value, tdoa, ipi, salience, spec_centroid, spec_max))
