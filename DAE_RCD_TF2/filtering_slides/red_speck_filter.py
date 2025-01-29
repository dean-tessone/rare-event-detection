# Copyright (c) 2024 University of Southern California
# Licensed under the MIT License (see LICENSE file for details)

import os
import numpy as np
import pandas as pd
from filtering_slides.utils import utils
import argparse
import subprocess


def is_edge(frame_id):
    if frame_id <= 48 or frame_id > 2256 or frame_id % 24 in {0, 1, 2, 23}:
        return True
    else:
        return False


list_frames = []

for i in range(2300):
    if not is_edge(i):
        list_frames.append(i)


def frameid_and_coords_from_tile_idx(tile_idx, list_frames):
    frame_id = list_frames[tile_idx // 1376]
    remainder = tile_idx % 1376
    x_coord = (remainder // 32) * 32
    y_coord = (remainder % 32) * 32
    return frame_id, x_coord, y_coord


def run_detection_threshold(slideID, slides_dir, savedir):

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

    command = [
        "python3",
        os.path.join(base_dir, "filtering_slides/count_red_spots.py"),
        f"--slideID={slideID}",
        f"--input={slides_dir}",
        f"--output={savedir}",
    ]

    subprocess.run(command)


def speck_filter(slideID, savedir, slides_dir, ntiles, threshold=500):

    csv_dir = savedir + "/speck_filter/csv_files"
    indices_dir = savedir + "/speck_filter/indices"
    if not os.path.exists(csv_dir):
        os.makedirs(csv_dir)
    if not os.path.exists(indices_dir):
        os.makedirs(indices_dir)

    run_detection_threshold(slideID, slides_dir, csv_dir)

    speck_df = pd.read_csv(csv_dir + f"/{slideID}.csv", sep="\t")

    speck_dict = {}
    for i in range(len(speck_df)):
        speck_dict[speck_df["frame_id"][i]] = speck_df["spot_count"][i]

    ans = np.zeros(ntiles)

    for i in range(ntiles):
        frameid, x, y = frameid_and_coords_from_tile_idx(i, list_frames)
        if speck_dict[frameid] > threshold:
            ans[i] = 1

    np.save(indices_dir + f"/{slideID}.npy", ans.astype(bool).squeeze())

    return ans.astype(bool).squeeze()
