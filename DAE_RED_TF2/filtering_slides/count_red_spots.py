# Copyright (c) 2024 University of Southern California
# Licensed under the MIT License (see LICENSE file for details)

from skimage import measure, color
import pandas as pd
import numpy as np
import os
import sys
import cv2
import tarfile
import shutil
import argparse
from utils import utils
from utils.frame import Frame


def get_noise_qc(frames, threshold):
    "count number of bright spots"
    frame_ids = []
    spot_count = []
    spot_area_mean = []
    spot_area_sdev = []

    for frame in frames:
        frame.readImage()
        if frame.image.ndim == 3:
            frame.image = frame.image[:, :, 0]
        mask = (frame.image > threshold).astype("uint8")
        num, _, stats, _ = cv2.connectedComponentsWithStats(mask, 4)
        area_mean = np.mean(stats[1:, cv2.CC_STAT_AREA]) if num > 1 else 0
        area_sdev = np.std(stats[1:, cv2.CC_STAT_AREA]) if num > 1 else 0
        frame_ids.append(frame.frame_id)
        spot_count.append(num - 1)
        spot_area_mean.append(area_mean)
        spot_area_sdev.append(area_sdev)

    df = pd.DataFrame(
        {
            "frame_id": frame_ids,
            "spot_count": spot_count,
            "area_mean": spot_area_mean,
            "area_dev": spot_area_sdev,
        }
    )

    return df


def is_edge(frame_id):
    if frame_id <= 48 or frame_id > 2256 or frame_id % 24 in {0, 1, 2, 23}:
        return True
    else:
        return False


def count_spots(nframes, offset, input, start, format, channel, threshold, output):

    frames = []
    for i in range(nframes):
        frame_id = i + offset + 1
        if not is_edge(frame_id):
            paths = utils.generate_tile_paths(
                path=input,
                frame_id=frame_id,
                starts=[start],
                name_format=format,
            )

            frame = Frame(frame_id=frame_id, channels=[channel], paths=paths)
            frames.append(frame)

    df = get_noise_qc(frames, threshold)
    df.to_csv(output, index=False, sep="\t")

    print("Noise QC completed successfully!")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Process slide images to detect LEVs with a single channel",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "-s", "--slideID", type=str, required=True, help="Slide ID to process"
    )

    parser.add_argument(
        "-i", "--input", type=str, required=True, help="path to slide images directory"
    )

    parser.add_argument(
        "-o", "--output", type=str, required=True, help="output file path"
    )

    args = parser.parse_args()

    input_path = args.input
    output_path = args.output
    slide_ID = args.slideID

    if os.path.exists(output_path) is False:
        os.makedirs(output_path)

    channel = "TRITC"
    nframes = 2304
    threshold = 50000
    offset = 0
    start = 2305
    format = "Tile%06d.jpg"

    input_path_cur = input_path + f"/{slide_ID}"
    output_path_cur = output_path + f"/{slide_ID}.csv"

    count_spots(
        nframes,
        offset,
        input_path_cur,
        start,
        format,
        channel,
        threshold,
        output_path_cur,
    )
