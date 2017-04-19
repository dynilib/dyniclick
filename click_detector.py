"""
Basic click detection. Ermites 2016.
"""

import logging
import argparse
import numpy as np
import copy
from collections import defaultdict, OrderedDict

import soundfile as sf
from scipy.signal import butter, lfilter, fftconvolve


def build_butter_bandpass(lowcut, highcut, sr, order=5):
    nyq = 0.5 * sr
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='bandpass')
    return b, a

def butter_filter(data, filter):
    return lfilter(filter[0], filter[1], data)

def get_envelope(data, sr, env_sr):

    # full-wave rectification
    data = np.abs(data)

    # half-hanning smoothing
    win_duration = 0.05 # in s
    win_size = win_duration * sr
    win = np.hanning(win_size)
    win_max_ind = np.argmax(win)
    win = win[win_max_ind:] / np.sum(win) * 2

    # downsampling
    downsampling_ratio = int(sr / env_sr)
    return np.log(fftconvolve(data, win, 'same'))[::downsampling_ratio]

def clean_detections(detection_dict, clean_size):
    "Remove detected values if a higher one is found within clean_size samples"
    prev_k = -1
    prev_v = -1
    to_remove = set()
    for k, v in detection_dict.items():
        if k - prev_k < clean_size:
            if v > prev_v:
                to_remove.add(prev_k)
                prev_k = k
                prev_v = v
            else:
                to_remove.add(k)
        else:
            prev_k = k
            prev_v = v
    
    for k in to_remove:
        try:
            del detection_dict[k]
        except:
            pass # only when k = -1


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description="""Click detector based on onset
        detection.\nA click is detected if an onset is detected in all the frequency
        bands.""")
    parser.add_argument(
        '-v', "--verbose",
        help="Set verbose output", action="store_const", const=logging.DEBUG,
        dest="loglevel", default=logging.INFO)
    parser.add_argument("input", help="Audio file.")
    parser.add_argument("output", help="Output csv file with detections.")
    parser.add_argument("--show", type=int, default=0, help="""Plot audio, clicks
            and some more stuff.""")
    parser.add_argument("--antares", type=int, default=0, help="""Reject clicks with
        onsets in antares bands (check acutofffreqs in the code).""")
    parser.add_argument("--channel", type=int, default=0, help="Audio channel to process")
    args = parser.parse_args()

    logging.getLogger().setLevel(args.loglevel)

    input = args.input
    output = args.output
    show = args.show
    antares = args.antares
    channel = args.channel

    # open audio file
    audio, sr = sf.read(input)

    if len(audio.shape) > 1:
        audio = audio[:, channel]

    # compute downsampling ratio
    env_sr = 1000
    downsampling_ratio = int(sr / env_sr)

    # build N bandpass filters for whale detection
    wcutofffreqs = [(11000, 12000), (9000, 10000), (8000, 9000), (7000, 8000),
            (6000, 7000)] # keep bands in decreasing frequency
    whale_filters = []
    for f in wcutofffreqs:
        whale_filters.append(build_butter_bandpass(f[0], f[1], sr))

    if antares:
        # build M bandpass filters to detect antares clicks
        acutofffreqs = [(60000, 80000)] # keep bands in decreasing frequency
        rejection_filters = []
        for f in acutofffreqs:
            rejection_filters.append(build_butter_bandpass(f[0], f[1], sr))

    # filter signal
    whale_bands = []
    for f in whale_filters:
        whale_bands.append(butter_filter(audio, f))
    if antares:
        rejection_bands = []
        for f in rejection_filters:
            rejection_bands.append(butter_filter(audio, f))

    # get log envelopes
    whale_envelopes = []
    for s in whale_bands:
        whale_envelopes.append(get_envelope(s, sr, env_sr))
    if antares:
        rejection_envelopes = []
        for s in rejection_bands:
            rejection_envelopes.append(get_envelope(s, sr, env_sr))

    # get derivative
    whale_derivatives = []
    for s in whale_envelopes:
        whale_derivatives.append(np.diff(s))
    if antares:
        rejection_derivatives = []
        for s in rejection_envelopes:
            rejection_derivatives.append(np.diff(s))
    
    # get values above a threshold
    threshold = 0.2
    whale_detections = []
    for s in whale_derivatives:
        d = defaultdict(int)
        for i in np.where(s>threshold)[0]:
            d[i] += s[i]
        whale_detections.append(OrderedDict(sorted(d.items())))
    if antares:
        rejection_detections = []
        for s in rejection_derivatives:
            d = defaultdict(int)
            for i in np.where(s>threshold)[0]:
                d[i] += s[i]
            rejection_detections.append(OrderedDict(sorted(d.items())))

    clicks = copy.deepcopy(whale_detections[0])

    # keep the highest values in a 100ms window (only in highest band = the
    # reference band)
    clean_size = int(0.05 * env_sr)
    clean_detections(clicks, clean_size)

    # frequency integration: keep only clicks detected in all whale bands
    # get clicks in the highest band and check if some click exists in other
    # bands between -50ms and +50ms
    to_remove = set()
    for k0 in clicks.keys():
        for d in whale_detections[1:]:
            found = False
            for k in d.keys():
                if np.abs(k0 - k) < clean_size:
                    found = True
                    break
            if not found:
                to_remove.add(k0)
                break

    for k in to_remove:
        del clicks[k]

    if antares:
        # now if a click is found in rejection bands, remove it
        to_remove = set()
        for k0 in clicks.keys():
            for d in rejection_detections:
                found = False
                for k in d.keys():
                    if np.abs(k0 - k) < clean_size:
                        found = True
                        break
                if found:
                    to_remove.add(k0)
                    break

        for k in to_remove:
            del clicks[k]

    with open(output, "w") as f:
        for k, v in clicks.items():
            f.write("{:.3f},{:.3f}\n".format(k / float(env_sr), v))

    if show:
        # plot stuff
        x = np.arange(len(audio)) / sr

        
        fig = plt.figure()
        
        ax0 = fig.add_subplot(411)
        ax0.plot(x, audio, 'b')
        ax0.set_title("Full band: Audio signal + final clicks")
        ax1 = ax0.twinx()
        t = np.asarray(list(clicks.keys())) / env_sr
        c = np.asarray(list(clicks.values()))
        ax1.scatter(t, c, marker="x", c="r")
        ax1.set_xlim(left=0)

        ax2 = fig.add_subplot(412)
        ax2.plot(x, whale_bands[0], 'g')
        ax2.set_title("Detection band 1: Audio signal + derivatives + detections)")
        ax3 = ax2.twinx()
        ax3.plot(x[::downsampling_ratio][:-1], whale_derivatives[0], 'r')
        t = np.asarray(list(whale_detections[0].keys())) / env_sr
        c = np.asarray(list(whale_detections[0].values()))
        ax3.scatter(t, c, marker="x")
        ax3.set_xlim(left=0)
        
        ax4 = fig.add_subplot(413)
        ax4.plot(x, whale_bands[1], 'g')
        ax4.set_title("Detection band 2: Audio signal + derivatives + detections)")
        ax5 = ax4.twinx()
        ax5.plot(x[::downsampling_ratio][:-1], whale_derivatives[1], 'r')
        t = np.asarray(list(whale_detections[1].keys())) / env_sr
        c = np.asarray(list(whale_detections[1].values()))
        ax5.scatter(t, c, marker="x")
        ax5.set_xlim(left=0)
        
        if antares:
            ax6 = fig.add_subplot(414)
            ax6.plot(x, rejection_bands[0], 'g')
            ax6.set_title("Retection band 1: Audio signal + derivatives + detections)")
            ax7 = ax6.twinx()
            ax7.plot(x[::downsampling_ratio][:-1], rejection_derivatives[0], 'r')
            t = np.asarray(list(rejection_detections[0].keys())) / env_sr
            c = np.asarray(list(rejection_detections[0].values()))
            ax7.scatter(t, c, marker="x")
            ax7.set_xlim(left=0)

        plt.show()
