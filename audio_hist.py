#!/usr/bin/env python
"""
Generate histogram of abs(audio) values.

Usage: read 
    $ ./audio_hist.py --h
"""


import argparse
import numpy as np

import soundfile as sf


if __name__ == "__main__":

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description="""Computes histogram of abs(audio sample values). Returns one histogram per channel.""")
    parser.add_argument("audio_file", help="Audio file.")
    parser.add_argument('--bins', type=float, nargs='+', default=[0, .25, .50, .75, .95, 1], help='Boundaries of the histogram bins.')
    parser.add_argument('--norm', default=1, help='Normalize sum to 1.')

    args = parser.parse_args()

    audio_file = args.audio_file
    bins = args.bins
    bins = args.bins
    norm = args.norm

    audio, sr = sf.read(audio_file, dtype="float32")
    for c in range(audio.shape[1]):
        hist = np.histogram(np.abs(audio[:,c]), bins)[0]
        if norm:
            hist = hist / np.sum(hist)
        print(",".join(format(x, "1.3f") for x in hist))



