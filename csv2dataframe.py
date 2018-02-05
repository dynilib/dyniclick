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

import git


logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

path = os.path.dirname(os.path.abspath(__file__))


DEFAULT_FEAT_FILE_EXT = "feat"

DATE_REGEX = [
    r'(\d{4})-(\d{2})-(\d{2})_(\d{2})(\d{2})(\d{2})UTC',
    r'.*(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})UTC'
]


def parse_date_sarano(basename):
    
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


def process(root_dir, output, feat_names, feat_file_ext=DEFAULT_FEAT_FILE_EXT):

    df = pd.DataFrame()

    for root, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(feat_file_ext):
                try:
                    date = parse_date_sarano(filename)

                    if not date:
                        logging.warning("Wrong date format: {}".format(filename))
                        continue

                    df_i = pd.read_csv(os.path.join(root, filename), names=feat_names, comment="#")

                    # Add click time to date.
                    # Click time must be in second and 
                    # in first column.
                    df_i[df_i.columns[0]] = df_i[df_i.columns[0]].apply(
                        lambda t: date + datetime.timedelta(seconds=float(t))
                    )

                    df = df.append(df_i)

                except Exception as e:
                    exc_type, exc_value, exc_traceback = sys.exc_info()
                    traceback.print_exception(exc_type, exc_value, exc_traceback,
                                              limit=2, file=sys.stdout)

    if not df.empty:
        df = df.set_index(df.columns[0]).sort_index()

    return df


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Concatenate click feature files into a compressed Pandas\' DataFrame..')
    parser.add_argument("root_dir", help="Feature file root.")
    parser.add_argument("output", help="Output file.")
    parser.add_argument('--feat_names', type=str, nargs='+', help='Feature names.')
    parser.add_argument("--feat_file_ext", type=str, default=DEFAULT_FEAT_FILE_EXT, help="Feature file extension.")
    args = parser.parse_args()

    root_dir = args.root_dir
    output = args.output
    feat_names = args.feat_names
    feat_file_ext = args.feat_file_ext

    df = process(root_dir, output, feat_names, feat_file_ext)

    store = pd.HDFStore(output, complib='zlib', complevel=5)
    store['data'] = df
    store.close()
