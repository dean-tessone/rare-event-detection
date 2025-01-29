import numpy as np


def l2distance(x, y):
    return np.squeeze(
        np.sqrt(
            np.sum(
                np.power(x - y, 2),
                axis=(1, 2, 3),
            )
        )
    )


def reconstruction_error_calc(
    dae_model,
    samples,
    n_samples,
    batch_size,
    dapi_coef=1,
    tritc_coef=1,
    cd45_coef=1,
    fitc_coef=1,
):
    total_l2_distances = []
    total_l2_distances_per_channel = []

    batch_size = 5000

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

        return total_l2_distances, total_l2_distances_per_channel
