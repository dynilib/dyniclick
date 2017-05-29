"""        
2-channels Inter Pulse Interval detection and delay measure.
"""

import os
import logging
import argparse
from argparse import RawDescriptionHelpFormatter
import textwrap

import soundfile as sf
import numpy as np
from scipy.signal import butter, lfilter, fftconvolve


def build_butter_highpass(cut, sr, order=5):
    nyq = 0.5 * sr
    cut = cut / nyq
    b, a = butter(order, cut, btype='highpass')
    return b, a

def butter_filter(data, filter):
    return lfilter(filter[0], filter[1], data)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=textwrap.dedent('''
    2-channels Inter Pulse Interval detection and delay measure.
    The algorithm is the following:
    For every click in the click file, in a 0.1s window around the click: 
        - Filter both channels with highpass filter with 8000 Hz cutoff frequency.
        - In channel 2, check if the autocorrelation of the click has a peak in the [ipi_min, ipi_max]
          range (in second) with amplitude higher than a threshold
        - If yes compute the cross-correlation between channels 1 and 2. The position of the peak gives the delay between both channels.
          Write output to click_root.

    The file written, if any, will be in click_root, have the same basename as the audio file and end with "delays.csv".
    The format of each line is:
    
    <date>, <time>, <click time>, <click confidence>, <xcorr delay>, <acorr ipi delay>, <acorr value at 0 delay>, <acorr value at ipi delay>
    '''), formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument(
        '-v', "--verbose",
        help="Set verbose output", action="store_const", const=logging.DEBUG,
        dest="loglevel", default=logging.INFO)
    parser.add_argument("input", help="Audio file.")
    parser.add_argument("click_root", help="Click file root directory.")
    parser.add_argument("--ipi_min", type=float, default=0.0015, help="Minimum IPI to be detected in s.")
    parser.add_argument("--ipi_max", type=float, default=0.008, help="Maximum IPI to be detected in s.")
    parser.add_argument("--filter_cutoff", type=int, default=8000, help="""Cut-off frequency of the high-pass filter in Hz.""")
    parser.add_argument("--swap_channels", type=int, default=0, help="""Swap channels 1 and 2 in algorithm.""")
    args = parser.parse_args()

    logging.getLogger().setLevel(args.loglevel)

    input = args.input
    click_root = args.click_root
    ipi_min = args.ipi_min
    ipi_max = args.ipi_max
    swap_channels = args.swap_channels
    filter_cutoff = args.filter_cutoff

    #############################
    # open and parse click file #
    #############################

    click_path = os.path.join(click_root, os.path.basename(input) + ".csv")
    
    with open(click_path, "r") as f:
        clicks = []
        for line in f.readlines():
            split_line = line.split(",")
            clicks.append((float(split_line[0]), float(split_line[1])))

    ###################   
    # open audio file #
    ###################   

    audio, sr = sf.read(input)

    if not len(audio.shape) == 2:
        raise Exception("File must have 2 channels")

    ###########
    # process #
    ###########

    delays = []

    for c, v in clicks:

        # Get bandpassed audio in 100ms around the detected clicks and compute
        # autocorrelation in channel 1

        filter = build_butter_highpass(filter_cutoff, sr)

        win_start_ind = int((c - 0.05) * sr)
        win_end_ind = int((c + 0.05) * sr)

        ch1 = butter_filter(audio[win_start_ind:win_end_ind, 1 if swap_channels else 0],
                filter)
        ch2 = butter_filter(audio[win_start_ind:win_end_ind, 0 if swap_channels else 1],
                filter)

        # Compute autocorrelation of ch2
        ch2_acorr = np.abs(np.correlate(ch2, ch2, "same"))

        # Compute ratio between highest value in the [ipi_min,ipi_max] autocorrelation
        # range and the autocorrelation value at 0 delay

        ac_d0 = np.argmax(ch2_acorr) # max at 0
        ac_v0 = ch2_acorr[ac_d0]

        ac_win_start = int(ac_d0 + ipi_min * sr)
        ac_win_end = int(ac_d0 + ipi_max * sr)
        peak_ind = ac_win_start + np.argmax(ch2_acorr[ac_win_start:ac_win_end])
        peak_value = ch2_acorr[peak_ind]

        ratio = ac_v0 / peak_value

        # If the ratio is smaller than a threshold, get intercorrelation peak
        threshold = 10 # arbitrary threshold to be investigated some day
        if ratio < threshold:
            xcorr = np.abs(np.correlate(ch1, ch2, "same"))
            xc_delay = (np.argmax(xcorr) - ac_d0) / float(sr)
            ipi = (peak_ind - ac_d0) / float(sr)
            delays.append((c, v, xc_delay, ipi, ac_v0, peak_value))

    if delays:

        # parse date / time from filename
        filename = os.path.basename(input)
        date = filename.split("_")[0]
        time = filename.split("_")[1]

        with open(click_path.replace("csv", "delays.csv"), "w") as f:
            for c, v, xc_delay, ipi, ac_v0, peak_value in delays:
                f.write("{},{},{},{},{},{},{:.5f},{:.5f}\n".format(
                    date,time, c, v, xc_delay, ipi, ac_v0, peak_value))
