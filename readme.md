# LenghuSky-8

This repository contains the code to reproduce the main results from the LenghuSky-8 paper. It provides a complete pipeline for cloud image analysis including preprocessing, segmentation, astrometric calibration, and benchmarking.

## Overview

LenghuSky-8 is a comprehensive framework for analyzing all-sky cloud camera images, including:
- Image preprocessing and enhancement
- Cloud segmentation using DINOv3
- Background classification
- Astrometric calibration
- Baseline models and benchmarking

## Repository Structure

### 1. `preprocess/`
Preprocessing scripts to convert raw cloud camera images into formats suitable for analysis.

- **`preprocess.py`** - Converts raw images for segmentation (S3 enhancement, 512x512) or calibration (no enhancement, 4096x4096)

### 2. `inference_segmentation_dinov3/`
Cloud segmentation inference using DINOv3 model.

- **`inference.py`** - Main inference script for cloud segmentation
- **`infer_config.json`** - Configuration file for inference parameters

### 3. `background_classify/`
Binary classification for determining background position (upper or lower) in cloud camera images.

- **`binary_2023_two_class_tokenization.py`** - Extracts CLS tokens from training and test images
- **`binary_2023_two_class_linear.py`** - Trains and tests a linear classifier on CLS tokens
- **`binary_2023_two_class_inference.py`** - Runs inference on images after 2023-09-27, outputs CSV with time, class, and probability

### 4. `calibration/`
Astrometric calibration tools for mapping between pixel and sky coordinates.

- **`Jia25_ensemble.py`** - Performs astrometric calibration to resolve WCS parameters for images
- **`calibrate_and_save.py`** - Aggregates calibration results for global fitting over time durations
- **`allsky_mapper.py`** - Maps pixel coordinates to sky coordinates and vice versa using calibration results

### 5. `baseline_and_benchmark/`
Baseline models and benchmarking code that produce the results shown in the paper.

- **`segbaseline/`** - Baseline implementations for cloud segmentation task
- **`nowcast_baseline/`** - Baseline implementations for nowcasting task
  - ConvLSTM baseline
  - Optical flow baseline
  - Trivial baseline

## Usage

Each script contains detailed documentation at the beginning explaining its function and usage. Additional readme files are provided in subfolders for more details.

### Quick Start

1. **Preprocess images**: Use `preprocess/preprocess.py` to prepare your raw images
2. **Run segmentation**: Use `inference_segmentation_dinov3/inference.py` for cloud segmentation
3. **Calibrate**: Use scripts in `calibration/` for astrometric calibration
4. **Benchmark**: Use scripts in `baseline_and_benchmark/` to reproduce paper results

## Data

The dataset is available on Hugging Face: [https://huggingface.co/datasets/ruiyicheng/LenghuSky-8](https://huggingface.co/datasets/ruiyicheng/LenghuSky-8)

The `baseline_and_benchmark/data/` directory contains manually labeled 252 images used for benchmarking cloud segmentation performance.

## Requirements

Please refer to individual script headers for specific dependencies and requirements.

## Citation

If you use this code in your research, please cite the LenghuSky-8 paper.

## License

See LICENSE file for details.
