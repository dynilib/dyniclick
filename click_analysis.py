#!/usr/bin/env python
"""        
Compute click features, including Inter Pulse Interval (IPI) and
Time Difference Of Arrival (TDOA).

Usage: read
    $ ./click_analysis.py --h
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


CLICK_DURATION = 0.002 # estimated click_duration
CLIPPING_THRESHOLD = 0.99


def build_butter_highpass(cut, sr, order=4):
    nyq = 0.5 * sr
    cut = cut / nyq
    b, a = butter(order, cut, btype='highpass')
    return b, a


def next_power_of_two(x):
        return 1<<(x-1).bit_length()


def get_ipi(click, chunk, ipi_min, ipi_max, sr, threshold):
    """Compute IPI from xcorrelation peak in the IPI range
    """

    # Compute xcorrelation of click and chunk
    xcorr = np.abs(np.correlate(chunk, click, "valid"))

    # Compute ratio between correlation value at 0 delay and
    # highest value in the [ipi_min,ipi_max] autocorrelation range
    
    start_ind = int(ipi_min * sr)
    end_ind = int(ipi_max * sr)

    peak_ind = start_ind + np.argmax(xcorr[start_ind:end_ind])
    peak_value = xcorr[peak_ind]

    # If the ratio is greater than a threshold, then we consider it an IPI
    salience = peak_value / xcorr[0]
    if  salience > threshold:
        return peak_ind / float(sr), salience
    else:
        return None, None


def get_tdoa(click, ch2, sr, delay_max):
    
    # compute cross-channel delay from xcorrelation peak
    xcorr = np.abs(np.correlate(ch2, click, "valid"))
    peak_ind = np.argmax(xcorr)
    delay = (peak_ind - delay_max * sr) / float(sr)

    return delay


if __name__ == "__main__":

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description=textwrap.dedent('''
    Inter Pulse Interval detection and cross-channel delay measure.
    The algorithm is the following:
    For every click in the click file, in a window (default 0.01s) around the click: 
        - If check_clipping is set to True, check that the signal is not clipping
        - Filter both channels with highpass filter (default cutoff frequency = 100 Hz).
        - In channel 1, check if the autocorrelation of the click has a peak in the [ipi_min, ipi_max]
          range (in second) with amplitude higher than a threshold (optional)
        - Compute the cross-correlation between channels 1 and 2. The position of the peak gives the delay between both channels (TDOA).
        - Compute 

    The format of each line in the output file is:
    
    <click time>, <click confidence>, <click amplitude>, <xcorr delay>, <acorr ipi delay>, <acorr value at 0 delay>, <acorr value at ipi delay>
    '''))
    parser.add_argument(
        '-v', "--verbose",
        help="Set verbose output", action="store_const", const=logging.DEBUG,
        dest="loglevel", default=logging.INFO)
    parser.add_argument("audio_file", help="Audio file.")
    parser.add_argument("click_file", help="Click file.")
    parser.add_argument("output_file", help="Output file.")
    parser.add_argument("--channels", type=int, nargs="+", default=[0, 1], help="""Respectively channels 1 and 2 in the algorithm.""")
    parser.add_argument("--cutoff_freq", type=int, default=100, help="""Cut-off frequency of the high-pass filter, in Hz.""")
    parser.add_argument("--check_clipping", type=int, default=1, help="Check if the signal is clipping.")
    parser.add_argument("--use_ipi", type=int, default=1, help="Detect pulse, compute IPI and only use clicks with IPI in subsequent steps.")
    parser.add_argument("--ipi_min", type=float, default=0.0015, help="Minimum IPI to be detected, in s.")
    parser.add_argument("--ipi_max", type=float, default=0.008, help="Maximum IPI to be detected, in s.")
    parser.add_argument("--min_pulse_salience", type=float, default=0.08, help="Min ratio between pulse autocorr value and max autocorr value.")
    parser.add_argument("--delay_max", type=float, default=0.0015, help="Maximum cross-channel delay for a click, in s.")
    args = parser.parse_args()

    logging.getLogger().setLevel(args.loglevel)

    audio_file = args.audio_file
    click_file = args.click_file
    output_file = args.output_file
    channels = args.channels
    cutoff_freq = args.cutoff_freq
    check_clipping = args.check_clipping
    use_ipi = args.use_ipi
    ipi_min = args.ipi_min
    ipi_max = args.ipi_max
    min_pulse_salience = args.min_pulse_salience
    delay_max = args.delay_max

    #############################
    # open and parse click file #
    #############################

    clicks = np.loadtxt(click_file, delimiter=',')
    clicks = np.atleast_2d(clicks)

    ###################   
    # open audio file #
    ###################   

    audio, sr = sf.read(audio_file)
    n_channels = len(channels)

    if n_channels > 1 and audio.shape[1] <= 1:
        raise Exception("File must have at least 2 channels to compute TDOA")

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

            # Get needed audio chunks around the click
            ch1 = audio[int(t*sr):int((t+ipi_max+CLICK_DURATION)*sr), channels[0]]
            ch2 = audio[int((t-delay_max)*sr):int((t+delay_max+CLICK_DURATION)*sr), channels[1]]
            
            # If the chunk clips during more than CLICK_DURATION,
            # the click is removed.
            if (check_clipping and
                    (np.abs(ch1) > CLIPPING_THRESHOLD).sum() > CLICK_DURATION * sr):
                continue

            # Highpass filter
            if cutoff_freq > 0:
                ch1 = filtfilt(b, a, ch1)
            if n_channels > 1 and cutoff_freq > 0:
                ch2 = filtfilt(b, a, ch2)

            # Get click chunk from channel 0
            click = ch1[:int(CLICK_DURATION*sr)]

            # Get IPI from ch1
            ipi, salience = get_ipi(
                click,
                ch1,
                ipi_min,
                ipi_max,
                sr,
                min_pulse_salience)

            if not use_ipi or ipi:
                
                # compute TDOA
                if n_channels > 1:
                    tdoa = get_tdoa(click, ch2, sr, delay_max)

                # get click amplitude, assuming it is the max of the clip chunk
                max_ind = np.argmax(np.abs(click))
                click_amp = min(1, np.abs(click[max_ind])) # filtered signal may have values > 1

                f.write("{:.3f},{:.3f},{:.3f},{:.6f},{:.6f},{:.3f}\n".format(
                    t, v, click_amp, tdoa, ipi, salience))
