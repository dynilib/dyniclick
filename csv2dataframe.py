"""
Utility module to concatenate click feature files
into a pandas' DataFrame.
"""

import sys
import os
import traceback
import re
import argparse
import logging
import datetime
import pandas as pd
import numpy as np

import git


logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

path = os.path.dirname(os.path.abspath(__file__))


DEFAULT_FEAT_FILE_EXT = "feat"
DEFAULT_TRACK_FILE_EXT = "tracks"

DATE_REGEX = [
    r'(\d{4})-(\d{2})-(\d{2})_(\d{2})(\d{2})(\d{2})UTC',
    r'.*(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})UTC',
    r'.*(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})_UTC'
]


def parse_date(basename):
    
    for regex in DATE_REGEX:
        match = re.match(regex, basename)
        if match:
            break

    if not match:
        return None

    # get offset, in milliseconds
    offset = 0
    if basename.split(".")[0].endswith("_1"):
        offset = 150
    elif basename.split(".")[0].endswith("_2"):
        offset = 300

    year = int(match.group(1))
    month = int(match.group(2))
    day = int(match.group(3))
    hour = int(match.group(4))
    minute = int(match.group(5))
    second = int(match.group(6))

    return datetime.datetime(year, month, day,
                             hour, minute, second) + datetime.timedelta(seconds=offset)



def add_to_date(date, toadd):
    return d + datetime.timedelta(seconds=t)


def process(feat_root, output, feat_names,
            feat_file_ext=DEFAULT_FEAT_FILE_EXT,
            track_root='', track_file_ext=DEFAULT_TRACK_FILE_EXT):

    df_feat = pd.DataFrame()
    df_file = pd.DataFrame()

    last_track_id = 0

    for root, _, filenames in os.walk(feat_root):
        for filename in filenames:
            if filename.endswith(feat_file_ext):
                try:
                    date = parse_date_sarano(filename)

                    if not date:
                        logging.warning("Wrong date format: {}".format(filename))
                        continue

                    df_i = pd.read_csv(os.path.join(root, filename), names=feat_names, comment="#", dtype=np.float32)
                    if df_i.empty:
                        logging.debug("{} empty".format(filename))
                        continue

                    # Add click time to date.
                    # Click time must be in second and 
                    # in first column.
                    df_i[df_i.columns[0]] = df_i[df_i.columns[0]].apply(
                        lambda t: date + datetime.timedelta(seconds=float(t))
                    )

                    # Add track
                    if track_root:
                        track_ids = np.loadtxt(os.path.join(root, filename.replace(feat_file_ext, track_file_ext)), ndmin=1, dtype=np.int32)
                        max_track_id = max(track_ids)
                        track_ids[track_ids>-1] += last_track_id
                        df_i['track_id'] = track_ids
                        last_track_id += max_track_id + 1 if max_track_id != -1 else 0 # track_id starts from 0 in every file, so we must increment

                    df_feat = df_feat.append(df_i)

                except Exception as e:
                    exc_type, exc_value, exc_traceback = sys.exc_info()
                    traceback.print_exception(exc_type, exc_value, exc_traceback,
                                              limit=2, file=sys.stdout)

    if not df_feat.empty:
        df_feat = df_feat.set_index(df_feat.columns[0]).sort_index()

    return df_feat


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Concatenate click feature files into a compressed Pandas\' DataFrame..')
    parser.add_argument("feat_root", help="Feature files root.")
    parser.add_argument("output", help="Output file.")
    parser.add_argument('--feat_names', type=str, nargs='+', help='Feature names.')
    parser.add_argument("--feat_file_ext", type=str, default=DEFAULT_FEAT_FILE_EXT, help="Feature file extension.")
    parser.add_argument("--track_root", type=str, default='', help=" Track files root.")
    parser.add_argument("--track_file_ext", type=str, default=DEFAULT_TRACK_FILE_EXT, help="Track file extension.")
    args = parser.parse_args()

    feat_root = args.feat_root
    output = args.output
    feat_names = args.feat_names
    feat_file_ext = args.feat_file_ext
    track_root = args.track_root
    track_file_ext = args.track_file_ext

    df = process(feat_root, output, feat_names, feat_file_ext, track_root, track_file_ext)

    store = pd.HDFStore(output, complib='zlib', complevel=5)
    store['clicks'] = df
    store.close()
