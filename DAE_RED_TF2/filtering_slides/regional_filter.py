# Copyright (c) 2024 University of Southern California
# Licensed under the MIT License (see LICENSE file for details)

import os
import numpy as np


def filter_regionally(
    ordered_idxs,
    savedir,
    slideID,
    filter_threshold=10,
):
    # Filtering regionally

    indices_dir = savedir + "/regional_filter/indices"
    if not os.path.exists(indices_dir):
        os.makedirs(indices_dir)

    occurrences_dict_5k = dict([])
    occurrences_dict_10k = dict([])

    for i in range(5000):
        tile_idx = int(ordered_idxs[i])
        frame_id = tile_idx // 1376
        if frame_id not in occurrences_dict_5k:
            occurrences_dict_5k[frame_id] = 1
        else:
            occurrences_dict_5k[frame_id] += 1

    for i in range(10000):
        tile_idx = int(ordered_idxs[i])
        frame_id = tile_idx // 1376
        if frame_id not in occurrences_dict_10k:
            occurrences_dict_10k[frame_id] = 1
        else:
            occurrences_dict_10k[frame_id] += 1

    region_excl_list = []
    region_excl_array = np.zeros(2531840)

    for i in range(5000):
        tile_idx = int(ordered_idxs[i])
        frame_id = tile_idx // 1376
        if occurrences_dict_5k[frame_id] >= filter_threshold:
            region_excl_list.append(i)
            region_excl_array[tile_idx] = 1

    for i in range(10000):
        tile_idx = int(ordered_idxs[i])
        frame_id = tile_idx // 1376
        if occurrences_dict_10k[frame_id] >= filter_threshold * 1.7:
            if i not in region_excl_list:
                region_excl_list.append(i)
                region_excl_array[tile_idx] = 1

    np.save(indices_dir + f"/{slideID}.npy", region_excl_array.astype(bool))

    return region_excl_array.astype(bool).squeeze()
