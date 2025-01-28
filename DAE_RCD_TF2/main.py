import os
import numpy as np
import tensorflow as tf
from models import DAE
from utils import plot_grid
from config import cla
from DAE_trainer import train_DAE
from readprocess_tiles import create_tile_dataset
from filtering_slides.filter_slides import filter_slides
from utils import decompress_directory, remove_directory, l2distance

PARAMS = cla()

do_train = PARAMS.do_train
do_rank = PARAMS.do_rank
slideID = PARAMS.slideID
sigma = PARAMS.sigma
noise_type = PARAMS.noise_type
arch_type = PARAMS.architec
dapi_coef = PARAMS.dapi_coef
tritc_coef = PARAMS.tritc_coef
cd45_coef = PARAMS.cd45_coef
fitc_coef = PARAMS.fitc_coef
z_dim = PARAMS.zdim

print("Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")))

slide_directory = ""  # TO BE FILLED
savedir = ""  # TO BE FILLED
decompress_directory(slide_directory + f"/{slideID}.tar.gz", slide_directory)
samples = create_tile_dataset(slideID, slide_directory)
n_samples = len(samples)
batch_size = 500
target_slide = f"{slideID}"

print(f"\n --- Running Slide: {target_slide} \n")

print("\n --- Creating network folder \n")
experiment_name_model = (
    f"models_zdim_{z_dim}_{arch_type}_{noise_type}_sigma_{int(100*sigma)}"
)
experiment_name_rank = (
    f"rankings_zdim_{z_dim}_{arch_type}_{noise_type}_"
    + f"sigma_{int(100*sigma)}_dapi_{int(100*dapi_coef)}_"
    + f"tritc_{int(100*tritc_coef)}_cd45_{int(100*cd45_coef)}_"
    + f"fitc_{int(100*fitc_coef)}_v2_{PARAMS.epoch}"
)
savedir_model = savedir + experiment_name_model + f"/{target_slide}"
savedir_rank = savedir + experiment_name_rank + f"/{target_slide}"

if not os.path.exists(savedir_model + f"/model_{PARAMS.epoch}"):
    do_train = True
else:
    do_train = False

if do_train:

    if not os.path.exists(savedir_model):
        os.makedirs(savedir_model)
    else:
        print("\n     *** Folder already exists!\n")

    print("\n ============== LAUNCHING TRAINING SCRIPT =================\n")

    train_DAE(samples, savedir_model)


if do_rank:

    if not os.path.exists(savedir_rank):
        os.makedirs(savedir_rank)
    else:
        print("\n     *** Folder already exists!\n")

    print("\n ============== RANKING TILES =================\n")

    # ============== Parameters ============= cond =========

    dae_model = DAE(arch_type, latent_dim=z_dim)
    dae_model.load(savedir_model, epoch=PARAMS.epoch)

    total_l2_distances = []
    total_l2_distances_per_channel = []

    batch_size = 5000

    ### CHECK WHETHER NUMPY ORDERS ACORDING TO ONE COLUMN

    for i in range(n_samples // batch_size + 1):  # n_samples):
        batch_curr = (
            samples[batch_size * i : (i + 1) * batch_size].astype(np.float32) / 255.0
        )
        reconstruction = dae_model.reconstruct(batch_curr)
        l2_distances1 = l2distance(
            batch_curr[:, :, :, 0:1], reconstruction[:, :, :, 0:1]
        )
        l2_distances2 = l2distance(
            batch_curr[:, :, :, 1:2], reconstruction[:, :, :, 1:2]
        )
        l2_distances3 = l2distance(
            batch_curr[:, :, :, 2:3], reconstruction[:, :, :, 2:3]
        )
        l2_distances4 = l2distance(
            batch_curr[:, :, :, 3:4], reconstruction[:, :, :, 3:4]
        )

        l2_distances = np.reshape(
            l2_distances1 * dapi_coef
            + l2_distances2 * tritc_coef
            + l2_distances3 * cd45_coef
            + l2_distances4 * fitc_coef,
            newshape=(len(l2_distances1), 1),
        )

        total_l2_distances.append(l2_distances)
        total_l2_distances_per_channel.append(
            np.concatenate(
                [
                    l2_distances1.reshape(-1, 1),
                    l2_distances2.reshape(-1, 1),
                    l2_distances3.reshape(-1, 1),
                    l2_distances4.reshape(-1, 1),
                ],
                axis=1,
            )
        )

    i_range = np.reshape(np.arange(0, n_samples), newshape=(n_samples, 1))
    total_l2_distances = np.concatenate(total_l2_distances, axis=0)
    l2_distances = np.concatenate(total_l2_distances_per_channel, axis=0)

    results_table_np = np.concatenate([i_range, total_l2_distances], axis=1)

    sorted_idxs_unfiltered = np.flip(results_table_np[:, 1].argsort())
    results_table_np_unfiltered = results_table_np[sorted_idxs_unfiltered]
    indices_sorted_unfiltered = results_table_np_unfiltered[
        :, 0
    ]  # Extracting just the indices from results_table_np

    idxs_to_filter = filter_slides(
        indices_sorted_unfiltered,
        slideID,
        slide_directory,
        savedir_rank,
        filter_region=True,
        filter_red_speck=True,
    )

    results_table_np_filtered = results_table_np[~idxs_to_filter]
    sorted_idxs_filtered = np.flip(results_table_np_filtered[:, 1].argsort())
    results_table_np_filtered = results_table_np_filtered[sorted_idxs_filtered]
    indices_sorted_filtered = results_table_np_filtered[:, 0]

    rarest_tiles_filtered = (
        samples[indices_sorted_filtered[:2500].astype(int)].astype(np.float32) / 255.0
    )
    rarest_tiles_unfiltered = (
        samples[indices_sorted_unfiltered[:2500].astype(int)].astype(np.float32) / 255.0
    )

    ##############################

    np.save(savedir_rank + "/" "rarest_tiles_filtered", rarest_tiles_filtered)
    np.save(savedir_rank + "/" "rarest_tiles_unfiltered", rarest_tiles_unfiltered)
    np.save(savedir_rank + "/" + "ranked_indices_unfiltered", indices_sorted_unfiltered)
    np.save(savedir_rank + "/" + "ranked_indices_filtered", indices_sorted_filtered)
    np.save(savedir_rank + "/" + "l2_distances", l2_distances)

    ##############################

    plot_grid(rarest_tiles_unfiltered, savedir_rank, label="unfiltered", ntiles=500)
    plot_grid(rarest_tiles_filtered, savedir_rank, label="filtered", ntiles=500)

remove_directory(slide_directory + f"/{slideID}")
