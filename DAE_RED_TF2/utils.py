# Copyright (c) 2024 University of Southern California
# Licensed under the MIT License (see LICENSE file for details)

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
import tarfile
import os
import shutil


def save_loss(loss, loss_name, savedir, n_epoch, plotting=False):

    np.savetxt(f"{savedir}/{loss_name}.txt", loss)
    if plotting:
        fig, ax1 = plt.subplots()
        ax1.plot(loss)
        ax1.set_xlabel("Steps")
        ax1.set_ylabel(loss_name)
        ax1.set_xlim([1, len(loss)])

        ax2 = ax1.twiny()
        ax2.set_xlim([0, n_epoch])
        ax2.set_xlabel("Epochs")
        plt.tight_layout()
        plt.savefig(f"{savedir}/{loss_name}.png", dpi=200)
        plt.close()


def filter_regionally(results_table_np):
    # Filtering regionally

    occurrences_dict_5k = dict([])
    occurrences_dict_10k = dict([])
    n_samples = results_table_np.shape[0]

    for i in range(5000):
        tile_idx = int(results_table_np[i, 0])
        frame_id = tile_idx // 1376
        if frame_id not in occurrences_dict_5k:
            occurrences_dict_5k[frame_id] = 1
        else:
            occurrences_dict_5k[frame_id] += 1

    for i in range(10000):
        tile_idx = int(results_table_np[i, 0])
        frame_id = tile_idx // 1376
        if frame_id not in occurrences_dict_10k:
            occurrences_dict_10k[frame_id] = 1
        else:
            occurrences_dict_10k[frame_id] += 1

    idxs_list = []
    region_excl_list = []

    for i in range(5000):
        tile_idx = int(results_table_np[i, 0])
        frame_id = tile_idx // 1376
        if occurrences_dict_5k[frame_id] >= 10:
            region_excl_list.append(i)

    for i in range(10000):
        tile_idx = int(results_table_np[i, 0])
        frame_id = tile_idx // 1376
        if occurrences_dict_10k[frame_id] >= 17:
            if i not in region_excl_list:
                region_excl_list.append(i)

    for i in range(11000):
        if i not in region_excl_list:
            idxs_list.append(i)

    return idxs_list, region_excl_list


def make_plots_training(synthetic_images, n_rows, savedir, i, type_im="synth"):

    fig1, axs1 = plt.subplots(n_rows, 5, dpi=200, figsize=(20, int(4.4 * n_rows)))
    max_value = np.amax(synthetic_images)
    for idx in range(n_rows):
        combined_channels = [
            synthetic_images[idx, :, :, 1:2] + synthetic_images[idx, :, :, 3:4],
            synthetic_images[idx, :, :, 2:3] + synthetic_images[idx, :, :, 3:4],
            synthetic_images[idx, :, :, 0:1] + synthetic_images[idx, :, :, 3:4],
        ]
        synthetic_RGB_image = np.concatenate(combined_channels, axis=2)
        axs1[idx, 0].imshow(synthetic_RGB_image / 2, vmin=0, vmax=1)
        axs1[idx, 1].imshow(synthetic_images[idx, :, :, 0], cmap="gray", vmin=0, vmax=1)
        axs1[idx, 2].imshow(synthetic_images[idx, :, :, 1], cmap="gray", vmin=0, vmax=1)
        axs1[idx, 3].imshow(synthetic_images[idx, :, :, 2], cmap="gray", vmin=0, vmax=1)
        axs1[idx, 4].imshow(synthetic_images[idx, :, :, 3], cmap="gray", vmin=0, vmax=1)
        axs1[idx, 0].set_xticks([])
        axs1[idx, 0].set_yticks([])
        axs1[idx, 1].set_xticks([])
        axs1[idx, 1].set_yticks([])
        axs1[idx, 2].set_xticks([])
        axs1[idx, 2].set_yticks([])
        axs1[idx, 3].set_xticks([])
        axs1[idx, 3].set_yticks([])
        axs1[idx, 4].set_xticks([])
        axs1[idx, 4].set_yticks([])
        if idx == 0:
            axs1[idx, 1].set_title("DAPI", fontsize=24)
            axs1[idx, 2].set_title("TRITC", fontsize=24)
            axs1[idx, 3].set_title("CY5", fontsize=24)
            axs1[idx, 4].set_title("FITC", fontsize=24)

    fig1.tight_layout()
    fig1.subplots_adjust(wspace=0.0)
    fig1.savefig(f"{savedir}/{type_im}_sample_stats_{i+1}.png")
    plt.close("all")


def plot_grid(tiles, savedir, label, ntiles=500):

    rows = ntiles // 10
    fig, ax = plt.subplots(
        rows, 10, figsize=(60, 320)
    )  # 50 rows x 10 columns = 500 images
    for i in range(rows * 10):
        if i < len(tiles):
            curr_sample = tiles[i : i + 1]
            combined_channels = [
                curr_sample[0, :, :, 1:2] + curr_sample[0, :, :, 3:4],
                curr_sample[0, :, :, 2:3] + curr_sample[0, :, :, 3:4],
                curr_sample[0, :, :, 0:1] + curr_sample[0, :, :, 3:4],
            ]
            RGB_image = np.concatenate(combined_channels, axis=2)
            ax[i // 10, i % 10].imshow(RGB_image / 2, vmin=0, vmax=1)
            ax[i // 10, i % 10].set_xticks([])
            ax[i // 10, i % 10].set_yticks([])
            ax[i // 10, i % 10].set_title(
                f"{i}", fontsize=20
            )  # Change fontsize to a larger value
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.02, hspace=0.05)
    fig.savefig(savedir + "/" + f"rarest_tiles_{label}_500.png", dpi=50)
    plt.close("all")


def apply_top_hat(image, kernel_size):
    """Apply a top-hat morphological transformation."""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    tophat_img = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)
    return tophat_img


def decompress_directory(tar_gz_path, dest_dir):
    """
    Decompress a .tar.gz file to the specified directory.

    Parameters:
    tar_gz_path (str): The path to the .tar.gz file.
    dest_dir (str): The directory where the contents should be extracted.
    """
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    with tarfile.open(tar_gz_path, "r:gz") as tar:
        tar.extractall(path=dest_dir)


def remove_directory(dir_path):
    """
    Remove the original directory.

    Parameters:
    dir_path (str): The path to the directory to be remove.
    """

    shutil.rmtree(dir_path)
