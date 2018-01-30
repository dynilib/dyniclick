#!/usr/bin/env python
"""
Plot click features.

Usage: read 
    $ ./plot_utils.py --h
"""

import sys
import argparse
import numpy as np

import matplotlib.pyplot as plt


def plot_data(click_file, data, feat_names, feat_thres, time_offset, track_file):
    
    f, axarr = plt.subplots(len(feat_names), sharex=True)

    title = click_file if time_offset == 0.0 else "{}\ntime offset: {}".format(click_file, time_offset)
    axarr[0].set_title(title)


    # map track_ind to color if tracks are set
    colorlist = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    colors = [colorlist[int(d%(len(colorlist)-1))] for d in data[:,-1]] if track_file else 'b'
    
    t = np.asarray([d[0] for d in data])

    for i in range(len(feat_names)):

        d = np.asarray([d[i+1] for d in data])
        axarr[i].scatter(t, d, marker="x", c=colors)
        ymax = np.max(data[:,i+1])
        ymax = ymax + np.abs(ymax) / 10
        ymin = np.min(data[:,i+1])
        ymin = ymin - np.abs(ymax) / 10 # using ymax because ymin is often close to 0
        axarr[i].set_ylim(ymin, ymax)    
        axarr[i].grid()
        axarr[i].set_ylabel(feat_names[i])
    
    axarr[-1].set_xlabel('Time (s)')

#    plt.tight_layout()

    plt.show()

    
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="""Plot click features""")
    parser.add_argument("click_file", help="Click file with features. The click times must be in col 0.")
    parser.add_argument("--feat_names", type=str, nargs="+",
                        required=True, help="Feature names.")
    parser.add_argument("--feat_cols", type=int, nargs="+",
                        required=True, help="Indices of feature columns.")
    parser.add_argument("--feat_thres", type=float, nargs="+", default=[],
                        help="Feature threshold (not used if 0, otherwise clicks" +
                        " with feature value smaller than the threshold are not displayed.")
    parser.add_argument("--feat_scale", type=float, nargs="+", default=[],
                        help="Feature scaling factor.")
    parser.add_argument("--time_offset", type=float, default=0.0, help="Time offset.")
    parser.add_argument("--track_file", type=str, default='', help="Track file.")

    args = parser.parse_args()

    click_file = args.click_file
    feat_names = args.feat_names
    feat_cols = args.feat_cols
    feat_thres = args.feat_thres
    feat_scale = args.feat_scale
    time_offset = args.time_offset
    track_file = args.track_file


    if (len(feat_names) != len(feat_cols) or
            feat_thres and len(feat_names) != len(feat_thres) or
            feat_scale and len(feat_names) != len(feat_scale)):
        raise Exception("feat_names, feat_cols, feat_thres and feat_scale" +
                        " must have same length")

    # parse file
    data = np.loadtxt(click_file, delimiter=',')
    feat_cols = [0] + feat_cols # col 0 is click time
    data = data[:,feat_cols]

    # if track_file is set, add track ind to data
    if track_file:
        track_ind = np.loadtxt(track_file, ndmin=2)
        data = np.hstack((data, track_ind))

    for i, t in enumerate(feat_thres):
        if t != 0:
            data = data[data[:,i+1]>t]
    for i, s in enumerate(feat_scale):
        data[:, i+1] *= s
    
    if data.size == 0:
        raise Exception("No data")
    
    plot_data(click_file, data, feat_names, feat_thres, time_offset, track_file)
