#!/usr/bin/env python
"""
Multi-band, envelope derivative-based click detector.

Usage: read 
    $ ./click_detector.py --h
"""

import os
import logging
import argparse
import textwrap
import numpy as np
import copy
from collections import defaultdict, OrderedDict
import pickle

from scipy.signal import butter, filtfilt, fftconvolve
import matplotlib.pyplot as plt
import soundfile as sf
import git


logger = logging.getLogger(__name__)
path = os.path.dirname(os.path.abspath(__file__))


# default parameters
CLICK_DURATION = 0.002 # estimated click_duration
HALF_HANN_DURATION = 0.01 # duration of the half-hann window used to smooth the signal, in s
ENV_SR = 1000 # envelope sample rate, in Hz
THRESHOLD = 0.2 # detection threshold on the log-envelope derivative
MIN_TIME_BETWEEN_CLICKS = 0.01 # in s. The value is supposed to be longer than largest possible IPI.
CLIPPING_THRESHOLD = 0.999
CLIPPING_WINDOW = 0.05 # duration of the window in which clipping is checked around a click, in s


def build_half_hann(sr, half_hann_duration=0.05):
    half_hann_size = int(half_hann_duration * sr)
    half_hann = np.hanning(half_hann_size * 2 + 1)[half_hann_size:]
    return half_hann / np.sum(half_hann)


def get_envelope(x, sr, win, env_sr=1000, log=False):

    # Full-wave rectification
    x = np.abs(x)

    # Filter signal and decimate.
    # Use fftconvolve, faster than lfilter for large number of coef.
    decimation_ratio = int(sr / env_sr)
    env = fftconvolve(x, win, 'same')[::decimation_ratio]

    if log:
        env = np.log(env.clip(min=np.finfo(env.dtype).eps))

    return env


def get_peaks(data, threshold=0):
    """Returns peak indices and values above threshold."""
    return [(i, data[i]) for i in np.where(data[1:-1]>threshold)[0]+1 if data[i] > data[i-1] and data[i] > data[i+1]]


def detection2maxamp(audio, start, end, sr):
    """Returns argmax max around detection."""
    start_ind = int(max(0, start * sr))
    end_ind = int(min(len(audio), end * sr))
    ind = np.argmax(np.abs(audio[start_ind:end_ind])) + start_ind
    return ind / sr, np.abs(audio[ind])


def time_integration(clicks, min_time_between_clicks):    
    """Removes detected values if a higher one is found within min_time_between_clicks."""
    prev_i = 0
    prev_t = clicks[0][0]
    prev_v = clicks[0][1]
    to_remove = set()
    for i, (t, v) in enumerate(clicks[1:]):
        i += 1
        if t - prev_t < min_time_between_clicks:
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

    return np.delete(clicks, list(to_remove), axis=0)


def frequency_integration(clicks, detections, min_time_between_clicks):
    """Returns only clicks found in all detection bands.
    Set click value to sum of detection value."""
    to_remove = set()
    for i, (t0, _) in enumerate(clicks):
        sum_v = 0
        for d in detections[1:]:
            found = np.where(np.abs(t0 - [t for t, _ in d]) < min_time_between_clicks)[0]
            if found.size:
                best_match = np.argmin(np.abs(t0 - [d[j][0] for j in found]))
                sum_v += d[found[best_match]][1]
            else:
                sum_v = 0
                to_remove.add(i)
                break
        if sum_v:
            clicks[i][1] = sum_v

    return np.delete(clicks, list(to_remove), axis=0)


def detect_clicks(
        audio,
        sr,
        bandpass_freqs,
        threshold,
        keep_data=False):

    # Sort cutoff freqs
    if len(bandpass_freqs) % 2 != 0:
        raise Exception("The number of cutoff frequencies must be even.")
    bandpass_freqs.sort()

    nyq = sr / 2.0

    # Process
    bands = []
    envs = []
    ders = []
    detections = []

    for i, freqs in enumerate(list(zip(bandpass_freqs[::2], bandpass_freqs[1::2]))[::-1]):

        # Build bandpass filter
        b, a = butter(3, [freqs[0] / nyq, freqs[1] / nyq], btype='bandpass')

        # Filter signal
        # Use filtfilt for 0 phase delay
        if keep_data:
            bands.append(filtfilt(b, a, audio))
        else:
            band = filtfilt(b, a, audio)

        # Compute log envelope
        half_hann = build_half_hann(sr, half_hann_duration=HALF_HANN_DURATION)
        delay = (len(half_hann) - 1) / 2 / sr # filter delay
        if keep_data:
            envs.append(get_envelope(bands[-1], sr, half_hann, env_sr=ENV_SR, log=True))
        else:
            env = get_envelope(band, sr, half_hann, env_sr=ENV_SR, log=True)

        # Compute derivative (detection function)
        if keep_data:
            ders.append(np.diff(envs[-1]))
        else:
            der = np.diff(env)

        # Get detections above the threshold
        if keep_data:
            d = get_peaks(ders[-1], threshold=threshold)
        else:
            d = get_peaks(der, threshold=threshold)

        # Convert detection indices to times and compensate for delay
        # introduced by half-hanning filter
        d = [(ind / ENV_SR + delay, v) for ind, v in d]

        # Refine click time by picking the max amplitude around the detection
        # and keep this amplitude.
        # Chunk (a bit arbitrarily) taken between (detection - click duration/2) and
        # (detection + click duration/2)
        d = [detection2maxamp(
            bands[-1] if keep_data else band,
            t - CLICK_DURATION/2,
            t + CLICK_DURATION/2,
            sr) for t, _ in d]
        
        detections.append(d)

    # take the highest frequency band as the reference
    clicks = copy.deepcopy(detections[0])

    if clicks:

        # keep the highest values in a MIN_TIME_BETWEEN_CLICKS window
        clicks = time_integration(clicks, MIN_TIME_BETWEEN_CLICKS)

        # frequency integration: keep only clicks detected in all bands,
        # between -CLICK_DURATION and +CLICK_DURATION
        # of the reference click
        clicks = frequency_integration(clicks, detections, CLICK_DURATION)

        # now set click time as the argmax of the raw audio
        # in [t-CLICK_DURATION/2:t+CLICK_DURATION/2] minus CLICK_DURATION * 0.1,
        # and click value as the max
        for i in range(len(clicks)):
            t = clicks[i][0]
            new_t, v = detection2maxamp(
                audio,
                t - CLICK_DURATION/2,
                t + CLICK_DURATION/2,
                sr)
            clicks[i] = (max(0, new_t - CLICK_DURATION*0.1), min(1, v)) # min(1, v) because filtered signal may have values > 1

    return clicks, bands, envs, ders, detections, delay


def plot(audio,
         sr,
         env_sr,
         clicks,
         bands,
         envs,
         ders,
         detections,
         bandpass_freqs,
         delay,
         offset):
    
    bandpass_freqs.sort()

    # plot stuff
    fig, axarr = plt.subplots((len(bands)+1), sharex=True)
    x = np.arange(len(audio)) / sr + offset
    #axarr[0] = fig.add_subplot(len(bands) + 1, 1, 1)

    x_dec = np.arange(len(ders[0])) / ENV_SR + delay + offset

    for i in range(len(bands)):

        axarr[i].plot(x, bands[i], 'k')
#        axarr[i].set_title("Band {}-{} Hz: Audio + derivatives + detections".format(bandpass_freqs[-i*2-2], bandpass_freqs[-i*2-1]))
        axarr[i].xaxis.grid()
        axarr_ = axarr[i].twinx()
        axarr_.plot(x_dec, ders[i], 'g')
        axarr_.yaxis.grid()
        t = np.asarray([t for t, _ in detections[i]]) + offset
        c = np.asarray([v for _, v in detections[i]])
        axarr[i].scatter(t, c, marker="x")
        axarr[i].set_xlim(left=offset)
    
    i += 1
    axarr[i].plot(x, audio, 'k')
#    axarr[i].set_title("Full band: Audio signal + final clicks")
    axarr[i].xaxis.grid()
    axarr_ = axarr[i].twinx()
    t = np.asarray([t for t, _ in clicks]) + offset
    c = np.asarray([v for _, v in clicks])
    axarr_.scatter(t, c, marker="x", c="r")
    axarr_.vlines(t, 0, c, colors="r")
    axarr_.set_xlim(left=offset)
    axarr_.yaxis.grid()

    axarr[-1].set_xlabel("Time (s)")

    return fig

    
def process(filename_in,
            filename_out,
            bandpass_freqs=[10000, 15000, 15000, 20000],
            highpass_freq=1000,
            threshold=THRESHOLD,
            channel=0,
            time_range=[],
            show=False):

    # open audio file
    audio, sr = sf.read(filename_in, dtype="float32")
    duration = len(audio) / sr

    if len(audio.shape) > 1:
        audio = audio[:, channel]
    
    # compute clipping mask
    clipping_mask = np.abs(audio) > CLIPPING_THRESHOLD

    # highpass filtering
    nyq = sr / 2.0
    b, a = butter(4, highpass_freq / nyq, btype='highpass')
    audio = filtfilt(b, a, audio)

    # use specified time range
    offset = 0
    if time_range:
        offset = time_range[0]
        audio = audio[int(offset * sr):int(time_range[1] * sr)]

    # detect clicks
    clicks, bands, envs, ders, detections, delay = detect_clicks(
        audio,
        sr,
        bandpass_freqs,
        threshold,
        keep_data=show)
    
    # remove clicks when the signal has one clipping value
    # in a CLIPPING_WINDOW long window centered on the click time
    num_clicks = len(clicks)
    clicks = [c for c in clicks if not np.any(
        clipping_mask[
            max(0, int((c[0] - CLIPPING_WINDOW/2) * sr)):
            int((c[0] + CLIPPING_WINDOW/2) * sr)
        ]
    )]
    num_clipped_clicks_removed = num_clicks - len(clicks)

    # store in dict and save as pickle file
    d = dict()
    d['clicks'] = np.asarray(clicks, dtype=np.float32)
    d['config'] = {
        "filename_in": filename_in,
        "filename_out": filename_out,
        "bandpass_freqs": bandpass_freqs,
        "highpass_freq": highpass_freq,
        "threshold": threshold,
        "channel": channel,
        "time_range": time_range
    }
    d['file'] = __file__
    repo = git.Repo(path, search_parent_directories=True)
    d['commit'] = repo.head.object.hexsha
    d['duration'] = duration
    d['num_clipped_clicks_removed'] = num_clipped_clicks_removed
    pickle.dump(d, open(filename_out, 'wb'))
    
    if len(clicks) > 0:

        if show:
            fig = plot(audio,
                 sr,
                 ENV_SR,
                 clicks,
                 bands,
                 envs,
                 ders,
                 detections,
                 bandpass_freqs,
                 delay,
                 offset)
            return fig

    else:
        logger.info("No click found.")

    return clicks, num_clipped_clicks_removed, duration


if __name__ == "__main__":

    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                     description=textwrap.dedent("""
        Click detector based on onset detection.
        A click is detected if an onset is detected in all the frequency bands."""))
    parser.add_argument(
        '-v', "--verbose",
        help="Set verbose output", action="store_const", const=logging.DEBUG,
        dest="loglevel", default=logging.INFO)
    parser.add_argument("filename_in", help="Audio file.")
    parser.add_argument("filename_out", help="Output pickle file with detections.")
    parser.add_argument('--bandpass_freqs', type=int, nargs='+', default=[10000, 15000, 15000, 20000], help='Cutoff frequencies of the bandpass filters.')
    parser.add_argument('--highpass_freq', type=int, default=1000, help='Cutoff frequency of the high pass filter.')
    parser.add_argument("--threshold", type=float, default=THRESHOLD, help="Detection threshold")
    parser.add_argument("--channel", type=int, default=0, help="Audio channel to process")
    parser.add_argument("--time_range", type=float, nargs="+", default=[], help="Time range to process")
    parser.add_argument("--show", type=int, default=0, help="""Plot audio, clicks
            and some more stuff.""")
    args = parser.parse_args()
    
    logging.getLogger().setLevel(args.loglevel)

    process(args.filename_in, args.filename_out,
            bandpass_freqs=args.bandpass_freqs,
            highpass_freq=args.highpass_freq,
            threshold=args.threshold,
            channel=args.channel,
            time_range=args.time_range,
            show=args.show)




