#!/usr/bin/env python
"""        
Estimates cross-channel delay and Inter Pulse Interval.

Usage: read
    $ ./click_analysis.py --h
"""

import os
import sys
import logging
import argparse
from argparse import RawDescriptionHelpFormatter
import textwrap

from collections import defaultdict
import soundfile as sf
import numpy as np
from scipy.signal import butter, filtfilt
import pickle
import git

import spectral_features
from click_detector import CLICK_DURATION, CLIPPING_THRESHOLD


path = os.path.dirname(os.path.abspath(__file__))


# Threshold of pulse salience, measured as the ratio of
# click / pulse correlation to click autocorrelation,
# above which a pulse is detected.
MIN_PULSE_SALIENCE = 0.08


def build_butter_highpass(cut, sr, order=4):
    nyq = 0.5 * sr
    cut = cut / nyq
    b, a = butter(order, cut, btype='highpass')
    return b, a


def next_power_of_two(x):
    return 1<<(x-1).bit_length()


def get_ipi(click, chunk_ipi, ipi_min, sr, min_pulse_salience):
    """Compute IPI from click vs chunk_ipi cross-correlation.
    """

    # Compute auto-correlation of click at 0 delay
    acorr0 = np.abs(np.correlate(click, click, "valid"))[0]

    # Compute cross-correlation between click and chunk_ipi
    xcorr = np.abs(np.correlate(chunk_ipi, click, "valid"))
    
    # Get xcorr peak = ipi candidate
    ipi_candidate = np.argmax(xcorr)
    pulse_salience = xcorr[ipi_candidate] / acorr0

    # If pulse_salience is above threshold, ipi_candidate is a pulse
    if pulse_salience > min_pulse_salience:
        return (ipi_candidate / sr) + ipi_min, pulse_salience
    else:
        return None, None


def get_tdoa(click, chunk_tdoa, tdoa_max, sr):

    xcorr = np.abs(np.correlate(chunk_tdoa, click, "valid"))
    return np.argmax(xcorr) / sr - tdoa_max


def process(
        audio_file,
        click_file,
        output_file,
        highpass_freq,
        channels,
        compute_ipi,
        ipi_max,
        ipi_min,
        filter_by_ipi,
        tdoa_max
):

    #############################
    # open and parse click file #
    #############################

    data = pickle.load(open(click_file, 'rb'))
    clicks = np.atleast_2d(data['clicks'])

    ###################   
    # open audio file #
    ###################   

    audio, sr = sf.read(audio_file)

    if not audio.shape[1] > 1:
        raise Exception("File must have at least 2 channels")

    ###########
    # process #
    ###########

        d = defaultdict(list)

        # git info
        repo = git.Repo(path, search_parent_directories=True)
        d["commit"] = repo.head.object.hexsha
        d["file"] = __file__
        d["duration"] = data["duration"]
        d["config"] = {
            "audio_file": audio_file,
            "click_file": click_file,
            "output_file": output_file,
            "highpass_freq": highpass_freq,
            "channels": channels,
            "compute_ipi": compute_ipi,
            "ipi_max": ipi_max,
            "ipi_min": ipi_min,
            "filter_by_ipi": filter_by_ipi,
            "tdoa_max": tdoa_max
        }

        # params
        d["col_names"] = ["click_time", "click_value"]
        if compute_ipi:
            d["col_names"].append("ipi")
            d["col_names"].append("ipi_salience")
        if tdoa_max > 0:
            d["col_names"].append("tdoa")
        d["col_names"].append("spectrum_argmax")
        d["col_names"].append("spectral_centroid")

        if clicks.size == 0:
            sys.exit()

        # check channels
        n_channels = len(channels)
        if n_channels < 1 or n_channels > 2:
            raise ValueError("The number of channels must be 1 or 2")
        if n_channels > 1:
            ch1 = channels[0]
        if n_channels == 2:
            ch2 = channels[1]

        # check cutoff
        if highpass_freq > 0:
            b, a = build_butter_highpass(highpass_freq, sr)

        # check ipi related params
        if not compute_ipi:
            filter_by_ipi = 0

        for t, v in clicks:

            # params
            param_values = [t, v]
            
            # Get needed audio chunks around the click: 
            # - the click itself.
            click = audio[int(t*sr):int((t+CLICK_DURATION)*sr), ch1]
            
            if highpass_freq > 0:
                click = filtfilt(b, a, click)
            # - the chunk where the pulse is expected, on the same channel
            #   (used to compute the IPI)
            if compute_ipi:
                chunk_ipi =  audio[int((t+ipi_min)*sr):int((t+CLICK_DURATION+ipi_max)*sr), ch1]
                if highpass_freq > 0:
                    chunk_ipi = filtfilt(b, a, chunk_ipi)
            # - the chunk on the second channel to measure the TDOA
            if tdoa_max > 0 and n_channels == 2:
                chunk_tdoa = audio[int((t-tdoa_max)*sr):int((t+tdoa_max+CLICK_DURATION)*sr), ch2]
                is_ch2_clipping = np.any(np.abs(chunk_tdoa)>CLIPPING_THRESHOLD)
                if highpass_freq > 0 and not is_ch2_clipping:
                    chunk_tdoa = filtfilt(b, a, chunk_tdoa)
            
            # Estimate IPI
            if compute_ipi:
                ipi, ipi_salience = get_ipi(
                    click,
                    chunk_ipi,
                    ipi_min,
                    sr,
                    MIN_PULSE_SALIENCE)
                param_values += [ipi, ipi_salience]

            if not filter_by_ipi or ipi:

                # Estimate TDOA
                if tdoa_max > 0:
                    tdoa = np.NAN if is_ch2_clipping else get_tdoa(click, chunk_tdoa, tdoa_max, sr) 
                    param_values += [tdoa]

                # compute spectral features
                spec = np.abs(np.fft.rfft(click, n=next_power_of_two(len(click))))
                freq_bin = sr / 2 / len(spec)
                spec_argmax = int(np.argmax(spec) * freq_bin)
                spec_centroid = int(spectral_features.centroid(spec) * freq_bin)
                param_values += [spec_argmax, spec_centroid]

                d["features"].append(param_values)

        d["features"] = np.asarray(d["features"], dtype=np.float32)

        pickle.dump(d, open(output_file, 'wb'))


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
    parser.add_argument("--highpass_freq", type=int, default=1000, help="""Cut-off frequency of the high-pass filter, in Hz.""")
    parser.add_argument("--channels", type=int, nargs="+", default=[0, 1], help="""Respectively channels 1 and 2 in the algorithm.""")
    parser.add_argument("--compute_ipi", type=int, default=1, help="Compute Inter-Pulse Interval (IPI).")
    parser.add_argument("--ipi_min", type=float, default=0.0015, help="Minimum IPI to be detected, in s.")
    parser.add_argument("--ipi_max", type=float, default=0.008, help="Maximum IPI to be detected, in s.")
    parser.add_argument("--filter_by_ipi", type=int, default=1, help="Only keep clicks with pulse (i.e. with ipi != None.")
    parser.add_argument("--tdoa_max", type=float, default=0, help="Maximum cross-channel delay (or Time Delay Of Arrival) for a click, in s. TDOA is not computed if set to 0.") # 0.0005 s -> 0.75 m
    args = parser.parse_args()

    logging.getLogger().setLevel(args.loglevel)

    process(
        args.audio_file,
        args.click_file,
        args.output_file,
        args.highpass_freq,
        args.channels,
        args.compute_ipi,
        args.ipi_max,
        args.ipi_min,
        args.filter_by_ipi,
        args.tdoa_max
    )

