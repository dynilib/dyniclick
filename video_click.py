""" Add click visualization on the video """

import os
import argparse
import subprocess, shutil
from collections import deque

import numpy as np
import imageio
import skimage.draw


TMP_FILE = "deleteme.mp4"
CENTER_LINE_WIDTH = 20
CLICK_REC_WIDTH = 50


def draw_rectangle(r0, c0, width, height):
    rr, cc = [r0, r0, r0 + height, r0 + height], [c0, c0 + width, c0 + width, c0]
    return skimage.draw.polygon(rr, cc)

def get_indices(times, start, end):
    return np.where(np.logical_and(times>=start, times<=end))[0]

def fit_exp(p0, p1):
    # fit exp such as y = np.exp(x-a) + b
    a = - np.log((p1[1] - p0[1]) / (np.exp(p1[0]) - np.exp(p0[0])))
    b = p0[1] - np.exp(p0[0] - a)
    return a, b

def draw_click(x, y, click_w, click_h, im_w):

    r0 = y
    c0 = min(max(0, x - click_w/2), im_w - click_w)
    return draw_rectangle(r0, c0, click_w, click_h)

def process(video_file, click_file, output_file, max_tdoa, cols, offset, min_amp, max_amp, decay):

    # open video file
    reader = imageio.get_reader(video_file)
    fps = reader.get_meta_data()["fps"] # frame per second
    im_size = reader.get_meta_data()["size"]
    im_w = im_size[0]
    im_h = im_size[1]
    writer = imageio.get_writer(TMP_FILE, fps=fps)

    # open and parse click_file
    clicks = np.loadtxt(click_file, delimiter=',', usecols=cols)
    clicks = np.atleast_2d(clicks)

    # only keep rows with amp > min_amp
    clicks = clicks[clicks[:,1]>=min_amp]

    # apply click offset and delete negative times
    clicks[:,0] += offset
    clicks = clicks[clicks[:,0]>0]

    # 0-delay line
    ref_line = draw_rectangle(0, (im_w-CENTER_LINE_WIDTH)/2, CENTER_LINE_WIDTH, im_h)

    buf_size = int(decay * fps)
    click_buf = deque([np.array([]) for _ in range(buf_size)])

    for i, im in enumerate(reader):

        # draw 0-delay line
        im[ref_line] = (0,0,0)

        # get indices of all clicks in frame
        click_buf.popleft()
        click_buf.append(get_indices(clicks[:, 0], i/fps, (i+1)/fps))

        if np.any([a.size for a in click_buf]):

            # draw all clicks
            for j in range(buf_size):
                for k in click_buf[j]:

                    amp = clicks[k,1]
                    tdoa = clicks[k,2]

                    # compute click position and height
                    x = im_w / 2 * (1 + (tdoa / max_tdoa))
                    h = im_h * amp / max_amp
                    y = (im_h - h) / 2

                    # draw click, with fade out of duration <decay> s
                    rec = draw_click(x, y, CLICK_REC_WIDTH, h, im_w)
                    skimage.draw.set_color(im, rec, (255,255,255), alpha=(j+1)/buf_size)

        writer.append_data(im)

    writer.close()

    # merge audio and new video streams to new file
    subprocess.run(
        "ffmpeg -i {} -i {} -map 0:a -map 1:v -c copy {}".format(video_file, TMP_FILE, output_file),
        shell=True,
        check=True)

    # delete tmp file
    os.remove(TMP_FILE)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description="""TODO""")
    parser.add_argument("video_file", help="Video file.")
    parser.add_argument("click_file", help="click file.")
    parser.add_argument("output_file", help="Output video file.")
    parser.add_argument("max_tdoa", type=float, help="Max Time Difference Of Arrival.")
    parser.add_argument("--cols", type=int, nargs="+", help="Indices of columns <time>, <amplitude>, <tdoa>")
    parser.add_argument("--offset", type=float, default=0, help="Click offset.")
    parser.add_argument("--min_amp", type=float, default=0.5, help="Min click amplitude.")
    parser.add_argument("--max_amp", type=float, default=1, help="Max click amplitude.")
    parser.add_argument("--decay", type=float, default=0.25, help="Min click amplitude.")

    args = parser.parse_args()

    video_file = args.video_file
    click_file = args.click_file
    output_file = args.output_file
    max_tdoa = args.max_tdoa
    cols = args.cols
    offset = args.offset
    min_amp = args.min_amp
    max_amp = args.max_amp
    decay = args.decay

    process(video_file, click_file, output_file, max_tdoa, cols, offset, min_amp, max_amp, decay)
