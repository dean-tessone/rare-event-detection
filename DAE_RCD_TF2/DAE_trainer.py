import os
import numpy as np
import tensorflow as tf
from config import cla
from models import DAE
from utils import save_loss, make_plots_GAN_training, GaussianCorrNoise

# === Setting up training step with tf.function ===


@tf.function
def train_step(model, x, xnoisy, optimizer):
    """Executes one training step and returns the loss.

    This function computes the loss and gradients, and uses the latter to
    update the model's parameters.
    """
    with tf.GradientTape() as tape:
        tape.watch(xnoisy)
        x_recons = model.reconstruct(xnoisy)
        loss = tf.reduce_mean(
            tf.math.reduce_sum(tf.math.pow(x - x_recons, 2), axis=[1, 2, 3])
        )

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return loss


def cast_and_normalize(image):
    image = tf.cast(image, tf.float32)
    image = image / 255.0
    return image


def train_DAE(train_data, savedir):

    PARAMS = cla()

    # ============== Parameters ======================
    n_epoch = PARAMS.epoch
    sigma = PARAMS.sigma
    arch_type = PARAMS.architec
    noise_type = PARAMS.noise_type
    z_dim = PARAMS.zdim
    batch_size = 1000
    n_out = 5
    learning_rate = 1e-5
    lr_sched = True

    if not os.path.exists(savedir):
        os.makedirs(savedir)
    else:
        print("\n     *** Folder already exists!\n")

    print("\n --- Saving parameters to file \n")
    param_file = savedir + "/parameters.txt"
    with open(param_file, "w") as fid:
        for pname in vars(PARAMS):
            fid.write(f"{pname} = {vars(PARAMS)[pname]}\n")

    print("\n --- Creating GAN models\n")

    dae_model = DAE(arch_type, latent_dim=z_dim)

    dae_optim = tf.keras.optimizers.Adam(learning_rate, beta_1=0.5, beta_2=0.9)

    if noise_type == "normal":
        noise_gen = tf.random.normal
    elif noise_type == "gaussian_cor":
        noise_gen = GaussianCorrNoise()

    # ============ Training ==================
    print("\n --- Starting training \n")
    n_iters = 1
    loss_log = []

    train_dataloader = (
        tf.data.Dataset.from_tensor_slices(train_data)
        .shuffle(train_data.shape[0])
        .map(cast_and_normalize)
        .batch(batch_size, drop_remainder=True)
        .prefetch(10)
    )

    for i in range(n_epoch):

        for x in train_dataloader:

            xnoisy = x + sigma * noise_gen(shape=x.shape)

            loss = train_step(dae_model, x, xnoisy, dae_optim)

            loss_log.append(loss.numpy())

            if n_iters % 100 == 0:
                print(
                    f"     *** iter:{n_iters} ---> reconstruction_loss:{loss.numpy():.4e}"
                )
            n_iters += 1

        if lr_sched:
            if n_epoch > 50:
                dae_optim.lr.assign(learning_rate * 0.97 ** (i + 1))
            else:
                dae_optim.lr.assign(learning_rate * 0.95 ** (i + 1))

        if (i == 0) or ((i + 1) % 25 == 0) or (i == n_epoch - 1):
            print("     *** Saving plots and network checkpoint")

            x_reconstructed = dae_model.reconstruct(xnoisy[0:5, :, :, :])

            make_plots_GAN_training(
                x_reconstructed, n_out, savedir, i, type_im="synth_reconstruction"
            )
            make_plots_GAN_training(
                xnoisy[0:5, :, :, :], n_out, savedir, i, type_im="true"
            )

            dae_model.save(savedir, epoch=i + 1)

            save_loss(loss_log, "loss", savedir, n_epoch)

    print("\n ============== DONE =================\n")
