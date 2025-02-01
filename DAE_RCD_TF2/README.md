# Rare Cell Detection Pipeline

This repository provides a pipeline for detecting rare cells in microscopy slides using a denoising autoencoder (DAE) model. The pipeline includes training the model, ranking tiles based on reconstruction errors, and filtering slides to identify rare tiles.

## Requirements

- Python 3.9.16
- TensorFlow 2.4.1
- The conda environment used is shown in: `environment.yml`

## Features

1. **Splitting large images into tiles**:
   - For the application of this method we split large immunofluorescent microscopy images into approximately 2,5 million tiles of shape (32,32,4). Tile size was chosen to be around 3 times larger than the average cell size. This should vary depending on the dataset.
2. **Training a DAE Model**: 
   - Train a denoising autoencoder on microscopy slide tiles.
3. **Ranking Tiles**:
   - Rank tiles based on reconstruction errors to identify rare events.
4. **Filtering Slides**:
   - Apply additional filtering based on region and red specks to refine the results. Note that the filters included in this repository are developed for the type of immunofluorescent microscopy images that we deal with in this study. If this method was to be applied to other datasets, it is likely that separate filters would need to be developed.

## Usage

### Note

- This respository is set up to allow replicating the results published in the manuscript. Changes are likely required to apply it to other datesets.

### 1. Prepare Slide Data
- Place the compressed slide files (`.tar.gz`) in the directory specified by `slide_directory`.
- Images within the `.tar.gz` file are saved as `.jpg` files. 
- The code is set up to read the images that we handle in our dataset. This code would need to be changed for other datasets that are saved differently.

### 2. Set Parameters
Configure the parameters in the `config.py` file or use command-line arguments. Key parameters include:
- `do_train`: Whether to train the DAE model (`True` or `False`).
- `do_rank`: Whether to rank tiles based on reconstruction errors.
- `slideID`: Identifier for the slide being processed.

### 3. Train the DAE
Run the script with `do_train=True` to train the DAE model. Trained models will be saved in a directory named based on the configuration.


### 4. Rank and Filter Tiles
Run the script with `do_rank=True` to rank tiles based on reconstruction errors. This step identifies rare tiles in the dataset based on their reconstruction error values. Results, including ranked tiles and filtered tiles, are saved in the output directory.

## Acknowledgements

I would like to acknowledg Amin Naghdloo for the methods he implemented [slide-image-utils](https://github.com/aminnaghdloo/slide-image-utils) repository, some of which are included here.

## License

This project is licensed under the MIT License - see the [LICENSE](../LICENSE) file for details.


