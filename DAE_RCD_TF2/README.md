# Rare Cell Detection Pipeline

This repository provides a pipeline for detecting rare cells in microscopy slides using a denoising autoencoder (DAE) model. The pipeline includes training the model, ranking tiles based on reconstruction errors, and filtering slides to identify rare tiles.

## Requirements

- Python 3.8+
- TensorFlow 2.x
- NumPy
- Matplotlib
- Other dependencies specified in `requirements.txt`

## Features

1. **Training a DAE Model**: 
   - Train a denoising autoencoder on microscopy slide tiles.
2. **Ranking Tiles**:
   - Rank tiles based on reconstruction errors to identify rare events.
3. **Filtering Slides**:
   - Apply additional filtering based on region and red specks to refine the results.
4. **Visualization**:
   - Generate grid plots of the rarest tiles for filtered and unfiltered datasets.

## Usage

### 1. Set Parameters
Configure the parameters in the `config.py` file or use command-line arguments. Key parameters include:
- `do_train`: Whether to train the DAE model (`True` or `False`).
- `do_rank`: Whether to rank tiles based on reconstruction errors.
- `slideID`: Identifier for the slide being processed.
- `sigma`: Standard deviation for adding noise during training.
- `noise_type`: Type of noise to use.
- `arch_type`: Architecture type of the DAE.
- `z_dim`: Latent dimension of the DAE.

### 2. Prepare Slide Data
- Place the compressed slide files (`.tar.gz`) in the directory specified by `slide_directory`.
- Ensure slide IDs match the naming convention.

### 3. Train the DAE
Run the script with `do_train=True` to train the DAE model. Trained models will be saved in a directory named based on the configuration.


### 4. Rank and Filter Tiles
Run the script with `do_rank=True` to rank tiles based on reconstruction errors. This step identifies rare tiles in the dataset based on their reconstruction error values. Results, including ranked tiles and filtered tiles, are saved in the output directory.


