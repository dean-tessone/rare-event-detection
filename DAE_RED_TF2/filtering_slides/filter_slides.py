# Copyright (c) 2024 University of Southern California
# Licensed under the MIT License (see LICENSE file for details)

import os
import numpy as np

from filtering_slides.regional_filter import filter_regionally
from filtering_slides.red_speck_filter import speck_filter


def filter_slides(
    ordered_idxs,
    slideID,
    slides_dir,
    savedir,
    filter_region=True,
    filter_red_speck=True,
    region_thres=20,
    speck_thres=500,
):
    # Filtering slides

    if filter_region:
        region_excl_idxs = filter_regionally(
            ordered_idxs, savedir, slideID, filter_threshold=region_thres
        )
    else:
        region_excl_idxs = np.zeros(ordered_idxs.shape[0]).astype(bool)

    if filter_red_speck:
        speck_idxs = speck_filter(
            slideID, savedir, slides_dir, ordered_idxs.shape[0], threshold=speck_thres
        )
    else:
        speck_idxs = np.zeros(ordered_idxs.shape[0]).astype(bool)

    combined_idxs = region_excl_idxs + speck_idxs
    combined_idxs = combined_idxs > 0

    return combined_idxs
