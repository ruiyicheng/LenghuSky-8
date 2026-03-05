SegMAN System Overview

This code uses the SegMAN architecture for semantic segmentation tasks. The setup includes scripts for dataset splitting, model training, evaluation, and testing. Below is an overview of the files and their functionalities:

File List and Description

run_all_scripts.py This script acts as a controller to run multiple training sessions with different configuration files. It runs the bootstrap_segman.py script with various configuration files (bootstrap_config_tiny.json, bootstrap_config_small.json, etc.). Each configuration corresponds to a specific variant of the SegMAN architecture (e.g., "tiny", "small", "base", "large").


bootstrap_segman.py This file manages the bootstrapping process for training and evaluation across multiple data splits. It splits the dataset into training, validation, and test sets using a custom splitter. Configures training settings for each seed and trains the model using the specified configuration. Collects the results for each seed and saves them into a JSON file.


train_segman.py This script performs the actual training of the SegMAN model. It Loads the model configuration. Handles the training loop for a given dataset, including data loading, loss computation, and backpropagation. Supports early stopping based on validation performance.


train_test_split.py This script splits the dataset into training, validation, and test sets. 

train_base_config_{}.json This JSON file contains the configuration parameters for the "tiny" variant of the SegMAN model. This configuration is used by bootstrap_segman.py to define the model training process for the "tiny" version.

bootstrap_config_{}.json This configuration file is used by the bootstrap_segman.py script to control the overall process. I

segman_encoder.py Contains the core encoder structure of the SegMAN model. It implements a custom neural network architecture using residual blocks and layer normalization. This model is designed to handle semantic segmentation tasks efficiently.

model_segman.py Defines the complete SegMAN model, combining the encoder defined in segman_encoder.py with a decoder to perform semantic segmentation. 

csm_triton.py Implements custom Triton kernel functions (cross_scan_kernel and cross_merge_kernel) for optimized pixel scanning and merging. These kernels help in speeding up the computation of pixel-wise operations in the model.

train_test_split.py This file handles the splitting of JSON files (which contain annotations in LabelMe format) into training, validation, and test sets. It ensures the datasets are organized and ready for training.

How to Run the System

run_all_test.sh