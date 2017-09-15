#!/usr/bin/env python
"""
Multi-band, envelope derivative-based click detector.

Usage: read 
    $ ./click_detector.py --h
"""


import logging
import argparse
import numpy as np
import copy
from collections import defaultdict, OrderedDict

from scipy.signal import butter, filtfilt, fftconvolve
import matplotlib.pyplot as plt
import soundfile as sf
import git


logger = logging.getLogger(__name__)


ENV_SR = 1000 # envelope sample rate, in Hz
HALF_HANN_DURATION = 0.05 # in s
DEFAULT_THRESHOLD = 0.2 # detection threshold on the log-envelope derivative
DEFAULT_MIN_TIME_BETWEEN_CLICKS = 0.01 # in s


def build_half_hann(sr, half_hann_duration=0.05):
    
    half_hann_size = int(half_hann_duration * sr)
    half_hann = np.hanning(half_hann_size * 2 + 1)[half_hann_size:]
    return half_hann / np.sum(half_hann)


def get_envelope(x, sr, win, env_sr=1000):

    # full-wave rectification
    x = np.abs(x)

    # filter signal and decimate
    # use fftconvolve, faster than lfilter for large number of coef
    decimation_ratio = int(sr / env_sr)
    return fftconvolve(x, win, 'same')[::decimation_ratio]


def time_clean(clicks, min_size_between_clicks):
    "Remove detected values if a higher one is found within min_size_between_clicks"
    prev_i = 0
    prev_t = clicks[0][0]
    prev_v = clicks[0][1]
    to_remove = set()
    for i, (t, v) in enumerate(clicks[1:]):
        i += 1
        if t - prev_t < min_size_between_clicks:
            if v > prev_v:
                to_remove.add(prev_i)
                prev_i = i
                prev_t = t
                prev_v = v
            else:
                to_remove.add(i)
        else:
            prev_i = i
            prev_t = t
            prev_v = v

    return [clicks[i] for i in range(len(clicks)) if i not in to_remove]


def frequency_clean(clicks, detections, min_size_between_clicks):

    to_remove = set()
    for i, (t0, _) in enumerate(clicks):
        for d in detections[1:]:
            found = False
            for t, _ in d:
                if np.abs(t0 - t) < min_size_between_clicks:
                    found = True
                    break
            if not found:
                to_remove.add(i)
                break

    return [clicks[i] for i in range(len(clicks)) if i not in to_remove]


def detect_clicks(
        audio,
        sr,
        cutoff_freqs,
        threshold,
        keep_data=False):

    # arrange cutoff freqs
    if len(cutoff_freqs) % 2 != 0:
        raise Exception("The number of cutoff frequencies must be even.")
    cutoff_freqs.sort()

    nyq = sr / 2.0

    # process
    bands = []
    envs = []
    ders = []
    detections = []

    for i, freqs in enumerate(list(zip(cutoff_freqs[::2], cutoff_freqs[1::2]))[::-1]):

        # build bandpass filter
        b, a = butter(3, [freqs[0] / nyq, freqs[1] / nyq], btype='bandpass')

        # filter signal
        # use filtfilt for 0 phase delay
        # then order is doubled
        if keep_data:
            bands.append(filtfilt(b, a, audio))
        else:
            band = filtfilt(b, a, audio)

        # compute log envelope
        half_hann = build_half_hann(sr, half_hann_duration=HALF_HANN_DURATION)
        delay = int(((len(half_hann) - 1) / 2) / sr * ENV_SR)
        if keep_data:
            envs.append(np.log(get_envelope(bands[-1], sr, half_hann, env_sr=ENV_SR).clip(min=np.finfo(bands[-1].dtype).eps)))
        else:
            env = np.log(get_envelope(band, sr, half_hann, env_sr=ENV_SR).clip(min=np.finfo(band.dtype).eps))

        # compute derivative
        if keep_data:
            ders.append(np.diff(envs[-1]))
        else:
            der = np.diff(env)

        # get peak values above the threshold
        d = []
        if keep_data:
            for i in np.where(ders[-1][1:-1]>threshold)[0]:
                i += 1 # we start at index 1 in the where condition
                if ders[-1][i] > ders[-1][i-1] and ders[-1][i] > ders[-1][i+1]:
                    d.append((i + delay, ders[-1][i]))
        else:
            for i in np.where(der[1:-1]>threshold)[0]:
                i += 1 # we start at index 1 in the where condition
                if der[i] > der[i-1] and der[i] > der[i+1]:
                    d.append((i + delay, der[i]))
        detections.append(d)


    # take the highest frequency band as the reference
    clicks = copy.deepcopy(detections[0])

    if clicks:
        # keep the highest values in a DEFAULT_MIN_TIME_BETWEEN_CLICKS window
        clicks = time_clean(clicks, DEFAULT_MIN_TIME_BETWEEN_CLICKS * ENV_SR)

        # frequency integration: keep only clicks detected in all bands,
        # between -DEFAULT_MIN_TIME_BETWEEN_CLICKS/2 and +DEFAULT_MIN_TIME_BETWEEN_CLICKS/2
        # of the reference click
        clicks = frequency_clean(clicks, detections, DEFAULT_MIN_TIME_BETWEEN_CLICKS * ENV_SR / 2)

    return clicks, bands, envs, ders, detections, delay


def plot(audio,
         sr,
         env_sr,
         clicks,
         bands,
         envs,
         ders,
         detections,
         cutoff_freqs,
         delay,
         offset):
    
    cutoff_freqs.sort()

    # plot stuff
    
    fig = plt.figure()
    x = np.arange(len(audio)) / sr + offset
    ax0 = fig.add_subplot(len(bands) + 1, 1, 1)
    ax0.plot(x, audio, 'b')
    ax0.set_title("Full band: Audio signal + final clicks")
    ax1 = ax0.twinx()
    t = np.asarray([t for t, _ in clicks]) / ENV_SR + offset
    c = np.asarray([v for _, v in clicks])
    ax1.scatter(t, c, marker="x", c="r")
    ax1.set_xlim(left=0)
    ax1.xaxis.grid(True, which='both')

    x_dec = (np.arange(len(ders[0])) + delay) / ENV_SR + offset

    for i in range(len(bands)):

        ax0 = fig.add_subplot(len(bands)+1, 1, i+2)
        ax0.plot(x, bands[i], 'g')
        ax0.set_title("Band {}-{} Hz: Audio + derivatives + detections".format(cutoff_freqs[-i*2-2], cutoff_freqs[-i*2-1]))
        ax1 = ax0.twinx()
        ax1.plot(x_dec, ders[i], 'r')
        t = np.asarray([t for t, _ in detections[i]]) / ENV_SR + offset
        c = np.asarray([v for _, v in detections[i]])
        ax1.scatter(t, c, marker="x")
        ax1.set_xlim(left=0)

    plt.grid()

    plt.tight_layout()

    plt.show()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description="""Click detector based on onset
        detection.\nA click is detected if an onset is detected in all the frequency
        bands.""")
    parser.add_argument(
        '-v', "--verbose",
        help="Set verbose output", action="store_const", const=logging.DEBUG,
        dest="loglevel", default=logging.INFO)
    parser.add_argument("input", help="Audio file.")
    parser.add_argument("output", help="Output csv file with detections.")
    parser.add_argument('--cutoff_freqs', type=int, nargs='+', default=[10000, 15000, 15000, 20000], help='Cutoff frequencies of the bandpass filters.')
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help="Detection threshold")
    parser.add_argument("--channel", type=int, default=0, help="Audio channel to process")
    parser.add_argument("--time_range", type=int, nargs="+", default=[], help="Audio channel to process")
    parser.add_argument("--show", type=int, default=0, help="""Plot audio, clicks
            and some more stuff.""")
    args = parser.parse_args()

    logging.getLogger().setLevel(args.loglevel)

    input = args.input
    output = args.output
    cutoff_freqs = args.cutoff_freqs
    threshold = args.threshold
    channel = args.channel
    time_range = args.time_range
    show = args.show

    # open audio file
    audio, sr = sf.read(input, dtype="float32")

    if len(audio.shape) > 1:
        audio = audio[:, channel]

    offset = 0
    if time_range:
        offset = time_range[0]
        audio = audio[int(offset * sr):int(time_range[1] * sr)]

    # detect clicks
    clicks, bands, envs, ders, detections, delay = detect_clicks(
        audio,
        sr,
        cutoff_freqs,
        threshold,
        keep_data=show)

    # write file
    with open(output, "w") as f:
        repo = git.Repo(search_parent_directories=True)
        sha = repo.head.object.hexsha
        f.write("#{}\n#Commit {}\n#Parameters: {}\n".format(__file__, sha, args))
        for k, v in clicks:
            f.write("{:.3f},{:.3f}\n".format(offset + (k / float(ENV_SR)), v))
    
    if clicks:

        if show:
            plot(audio,
                 sr,
                 ENV_SR,
                 clicks,
                 bands,
                 envs,
                 ders,
                 detections,
                 cutoff_freqs,
                 delay,
                 offset)

    else:
        logger.info("No click found.")
