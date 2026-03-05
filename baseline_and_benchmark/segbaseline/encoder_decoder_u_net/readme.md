This code is designed for cloud segmentation using a U-Net model. It includes scripts for preparing data, training the model, and evaluating its performance.

## Files Overview


### 1. `segmentation_config.json`
This file contains configuration settings for training the model. It specifies paths to the data, where to save the model checkpoints, and training settings like the number of epochs.

### 2. `bootstrap_segmentation.py`
This script helps automate the training process with multiple random seeds. It creates different configurations for each seed, trains the model, and logs the results.

### 3. `train_segmentation.py`
This is the main training script. It handles loading the data, training the Encoder-decoder model, and evaluating it on the test set. It uses the configuration file to guide the process.

### 4. `train_segmentation_Unet.py`
Similar to `train_segmentation.py`, but specifically for training a U-Net model with cloud segmentation data.
