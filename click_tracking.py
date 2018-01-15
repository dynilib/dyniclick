#!/usr/bin/env python
"""        
Click tracking from click tdoas.

Usage: read
    $ ./click_tracking.py --h
"""

import os
import sys
import logging
import argparse
from argparse import RawDescriptionHelpFormatter
import textwrap

import numpy as np
import git


path = os.path.dirname(os.path.abspath(__file__))


def track_clicks(clicks, click_interval_max, diff_max, polynomial_expectation=False):

    tracks = []

    for i, c in enumerate(clicks):

        if i == 0:
            tracks.append([i])
            continue

        diff_min = np.finfo(np.float32).max
        track_ind = None

        for j, track in enumerate(tracks):

            if c[0] - clicks[track[-1]][0] < click_interval_max:

                if polynomial_expectation:
                    # fit 2nd order polynomial on last 3 points
                    deg = min(len(track)-1, 2) # max degree is 2
                    last = track[-3:]
                    x = [clicks[k][0] for k in last]
                    y = [clicks[k][1] for k in last]
                    p = np.polyfit(x, y, deg)
                    # compute polynomial value at current click time
                    expected_tdoa = np.polyval(p, c[0])

                else:
                    expected_tdoa = clicks[track[-1]][1]

                diff  = np.abs(expected_tdoa - c[1])
                if diff < diff_min and diff < diff_max:
                    diff_min = diff
                    track_ind = j

        if track_ind:
            tracks[track_ind].append(i)
        else:
            tracks.append([i])

    return tracks


if __name__ == "__main__":

    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                     description=textwrap.dedent('''
    Click tracking.
    '''))
    parser.add_argument(
        '-v', "--verbose",
        help="Set verbose output", action="store_const", const=logging.DEBUG,
        dest="loglevel", default=logging.INFO)
    parser.add_argument("click_file", help="Click file, with tdoa.")
    parser.add_argument("output_file", help="Same as input, with an additional column specifying a track id, if a track is found.")
    parser.add_argument("--tdoa_col", type=int, help="Index of the tdoa column.")
    parser.add_argument("--click_interval_max", type=float, default=0.1, help="Maximum interval between clicks before ending a track.")
    parser.add_argument("--diff_max", type=float, default=2e-5, help="Maximum difference between expection and actual value to assign a click to a track .")
    args = parser.parse_args()

    logging.getLogger().setLevel(args.loglevel)

    click_file = args.click_file
    output_file = args.output_file
    tdoa_col = args.tdoa_col
    click_interval_max = args.click_interval_max
    diff_max = args.diff_max

    clicks = np.loadtxt(click_file, delimiter=",", usecols=[0,tdoa_col])

    tracks = track_clicks(clicks, click_interval_max, diff_max)

    # get track ind per click
    track_ind = []
    for i in range(clicks.shape[0]):
        track_ind.append(next(j for j, sublist in enumerate(tracks) if i in sublist))

    with open(output_file, "w") as f:

        # git info
        repo = git.Repo(path, search_parent_directories=True)
        sha = repo.head.object.hexsha
        f.write("#{}\n#Commit {}\n#Parameters: {}\n".format(__file__, sha, args))

        for i in track_ind:
            f.write("{}\n".format(i))
