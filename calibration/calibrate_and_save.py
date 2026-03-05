#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
calibrate_and_save.py

Runs the joint alt/az model fit for all tested image sets (including those
previously commented in your script), aligns each result to your setup history
time slots, and records the fitted model for reload.

Outputs (under TEMP_DIR / 'calibrations'):
  - <start_file>_calibration.json         (for need_new_astrometry=1 slots)
  - <start_file>_azfit.png
  - <start_file>_rfit.png
  - <start_file>_grid_preview.png         (optional, joint visualization)
  - calibration_index.json                (index of usable calibrations)
"""

import os
import json
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass

import numpy as np
import cv2
import healpy as hp

import astropy
import astropy.wcs
from astropy.io import fits
from astropy.time import Time
from astropy.coordinates import EarthLocation, AltAz
import astropy.units as units
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

# =========================
# Configuration
# =========================
UTC_PLUS_8 = timezone(timedelta(hours=8))

# IMPORTANT: base image size used in all fits
IMG_SIZE = 4096
BOX = 64
X_START = 32
Y_START = 32

# Model settings
MAX_ODD_ORDER = 9
NEWTON_MAX_ITERS = 30
NEWTON_TOL = 1e-9

# Aggregation for fitting: "all" or "median"
AGG_MODE = "all"

# Site
SITE_LAT = 38.9586
SITE_LON = 93.2681
SITE_ALT = 4200

# IO
TEMP_DIR = "/mnt/hgfs/cloud/data/calibration/out"
os.makedirs(TEMP_DIR, exist_ok=True)
CAL_DIR = os.path.join(TEMP_DIR, "calibrations")
os.makedirs(CAL_DIR, exist_ok=True)

# ---------------------------------------------------------------------
# 1) Hard-coded tested image sets (from your commented + active INPUT_IMG_LISTs)
# ---------------------------------------------------------------------
ALL_IMAGE_SETS: Dict[str, List[str]] = {
    # ---- 2018-06-10 ----
    "2018-06-10": [
        "/mnt/hgfs/cloud/data/calibration/all_input/2018-06-10-00-01-07.png",
        "/mnt/hgfs/cloud/data/calibration/all_input/2018-06-10-00-06-43.png",
        "/mnt/hgfs/cloud/data/calibration/all_input/2018-06-10-00-12-19.png",
        "/mnt/hgfs/cloud/data/calibration/all_input/2018-06-10-00-17-56.png",
        "/mnt/hgfs/cloud/data/calibration/all_input/2018-06-10-00-23-32.png",
        "/mnt/hgfs/cloud/data/calibration/all_input/2018-06-10-00-29-09.png",
        "/mnt/hgfs/cloud/data/calibration/all_input/2018-06-10-00-34-45.png",
        "/mnt/hgfs/cloud/data/calibration/all_input/2018-06-10-00-40-22.png",
        "/mnt/hgfs/cloud/data/calibration/all_input/2018-06-10-00-45-58.png",
        "/mnt/hgfs/cloud/data/calibration/all_input/2018-06-10-00-51-35.png",
    ],

    # ---- 2018-11-13 ----
    "2018-11-13": [
        "/mnt/hgfs/cloud/data/calibration/all_input/2018-11-13-00-02-21.png",
        "/mnt/hgfs/cloud/data/calibration/all_input/2018-11-13-00-07-54.png",
        "/mnt/hgfs/cloud/data/calibration/all_input/2018-11-13-00-13-28.png",
        "/mnt/hgfs/cloud/data/calibration/all_input/2018-11-13-00-24-35.png",
        "/mnt/hgfs/cloud/data/calibration/all_input/2018-11-13-00-30-09.png",
        "/mnt/hgfs/cloud/data/calibration/all_input/2018-11-13-00-35-42.png",
        "/mnt/hgfs/cloud/data/calibration/all_input/2018-11-13-00-41-16.png",
        "/mnt/hgfs/cloud/data/calibration/all_input/2018-11-13-00-46-50.png",
        "/mnt/hgfs/cloud/data/calibration/all_input/2018-11-13-00-52-23.png",
        "/mnt/hgfs/cloud/data/calibration/all_input/2018-11-13-00-57-57.png",
        "/mnt/hgfs/cloud/data/calibration/all_input/2018-11-13-01-03-31.png",
        "/mnt/hgfs/cloud/data/calibration/all_input/2018-11-13-01-09-04.png",
        "/mnt/hgfs/cloud/data/calibration/all_input/2018-11-13-01-14-38.png",
        "/mnt/hgfs/cloud/data/calibration/all_input/2018-11-13-01-20-11.png",
        "/mnt/hgfs/cloud/data/calibration/all_input/2018-11-13-01-31-19.png",
        "/mnt/hgfs/cloud/data/calibration/all_input/2018-11-13-01-36-52.png",
    ],

    # ---- 2019-05-30 ----
    "2019-05-30": [
        "/mnt/hgfs/cloud/data/calibration/all_input/2019-05-30-00-00-29.png",
        "/mnt/hgfs/cloud/data/calibration/all_input/2019-05-30-03-28-07.png",
        "/mnt/hgfs/cloud/data/calibration/all_input/2019-05-30-00-06-06.png",
        "/mnt/hgfs/cloud/data/calibration/all_input/2019-05-30-00-11-43.png",
        "/mnt/hgfs/cloud/data/calibration/all_input/2019-05-30-00-17-19.png",
        "/mnt/hgfs/cloud/data/calibration/all_input/2019-05-30-00-22-56.png",
        "/mnt/hgfs/cloud/data/calibration/all_input/2019-05-30-00-28-33.png",
        "/mnt/hgfs/cloud/data/calibration/all_input/2019-05-30-00-34-09.png",
        "/mnt/hgfs/cloud/data/calibration/all_input/2019-05-30-00-39-46.png",
        "/mnt/hgfs/cloud/data/calibration/all_input/2019-05-30-00-45-23.png",
        "/mnt/hgfs/cloud/data/calibration/all_input/2019-05-30-00-51-00.png",
        "/mnt/hgfs/cloud/data/calibration/all_input/2019-05-30-00-56-36.png",
        "/mnt/hgfs/cloud/data/calibration/all_input/2019-05-30-01-02-13.png",
        "/mnt/hgfs/cloud/data/calibration/all_input/2019-05-30-01-07-50.png",
        "/mnt/hgfs/cloud/data/calibration/all_input/2019-05-30-01-13-26.png",
        "/mnt/hgfs/cloud/data/calibration/all_input/2019-05-30-01-19-03.png",
        "/mnt/hgfs/cloud/data/calibration/all_input/2019-05-30-01-24-40.png",
        "/mnt/hgfs/cloud/data/calibration/all_input/2019-05-30-01-30-17.png",
        "/mnt/hgfs/cloud/data/calibration/all_input/2019-05-30-01-35-53.png",
        "/mnt/hgfs/cloud/data/calibration/all_input/2019-05-30-01-41-30.png",
        "/mnt/hgfs/cloud/data/calibration/all_input/2019-05-30-01-47-07.png",
        "/mnt/hgfs/cloud/data/calibration/all_input/2019-05-30-01-52-43.png",
        "/mnt/hgfs/cloud/data/calibration/all_input/2019-05-30-01-58-20.png",
        "/mnt/hgfs/cloud/data/calibration/all_input/2019-05-30-02-03-57.png",
        "/mnt/hgfs/cloud/data/calibration/all_input/2019-05-30-02-09-34.png",
        "/mnt/hgfs/cloud/data/calibration/all_input/2019-05-30-02-15-10.png",
        "/mnt/hgfs/cloud/data/calibration/all_input/2019-05-30-02-20-47.png",
        "/mnt/hgfs/cloud/data/calibration/all_input/2019-05-30-02-26-24.png",
        "/mnt/hgfs/cloud/data/calibration/all_input/2019-05-30-02-32-00.png",
        "/mnt/hgfs/cloud/data/calibration/all_input/2019-05-30-02-37-37.png",
        "/mnt/hgfs/cloud/data/calibration/all_input/2019-05-30-02-43-14.png",
        "/mnt/hgfs/cloud/data/calibration/all_input/2019-05-30-02-48-50.png",
        "/mnt/hgfs/cloud/data/calibration/all_input/2019-05-30-02-54-27.png",
        "/mnt/hgfs/cloud/data/calibration/all_input/2019-05-30-03-00-04.png",
        "/mnt/hgfs/cloud/data/calibration/all_input/2019-05-30-03-05-41.png",
        "/mnt/hgfs/cloud/data/calibration/all_input/2019-05-30-03-11-17.png",
        "/mnt/hgfs/cloud/data/calibration/all_input/2019-05-30-03-16-54.png",
        "/mnt/hgfs/cloud/data/calibration/all_input/2019-05-30-03-22-31.png",
    ],

    # ---- 2019-07-01 ----
    "2019-07-01": [
        "/mnt/hgfs/cloud/data/calibration/all_input/2019-07-01-03-17-41.png",
        "/mnt/hgfs/cloud/data/calibration/all_input/2019-07-01-00-06-25.png",
        "/mnt/hgfs/cloud/data/calibration/all_input/2019-07-01-00-12-19.png",
        "/mnt/hgfs/cloud/data/calibration/all_input/2019-07-01-00-18-13.png",
        "/mnt/hgfs/cloud/data/calibration/all_input/2019-07-01-00-24-08.png",
        "/mnt/hgfs/cloud/data/calibration/all_input/2019-07-01-00-30-02.png",
        "/mnt/hgfs/cloud/data/calibration/all_input/2019-07-01-00-35-56.png",
        "/mnt/hgfs/cloud/data/calibration/all_input/2019-07-01-00-41-50.png",
        "/mnt/hgfs/cloud/data/calibration/all_input/2019-07-01-00-47-44.png",
        "/mnt/hgfs/cloud/data/calibration/all_input/2019-07-01-00-53-38.png",
        "/mnt/hgfs/cloud/data/calibration/all_input/2019-07-01-00-59-48.png",
        "/mnt/hgfs/cloud/data/calibration/all_input/2019-07-01-01-05-42.png",
        "/mnt/hgfs/cloud/data/calibration/all_input/2019-07-01-01-11-36.png",
        "/mnt/hgfs/cloud/data/calibration/all_input/2019-07-01-01-17-47.png",
        "/mnt/hgfs/cloud/data/calibration/all_input/2019-07-01-01-23-41.png",
        "/mnt/hgfs/cloud/data/calibration/all_input/2019-07-01-01-29-35.png",
        "/mnt/hgfs/cloud/data/calibration/all_input/2019-07-01-01-35-45.png",
        "/mnt/hgfs/cloud/data/calibration/all_input/2019-07-01-01-41-39.png",
        "/mnt/hgfs/cloud/data/calibration/all_input/2019-07-01-01-47-49.png",
        "/mnt/hgfs/cloud/data/calibration/all_input/2019-07-01-01-59-38.png",
        "/mnt/hgfs/cloud/data/calibration/all_input/2019-07-01-02-05-48.png",
        "/mnt/hgfs/cloud/data/calibration/all_input/2019-07-01-02-11-42.png",
        "/mnt/hgfs/cloud/data/calibration/all_input/2019-07-01-02-17-36.png",
        "/mnt/hgfs/cloud/data/calibration/all_input/2019-07-01-02-23-46.png",
        "/mnt/hgfs/cloud/data/calibration/all_input/2019-07-01-02-29-40.png",
        "/mnt/hgfs/cloud/data/calibration/all_input/2019-07-01-02-35-50.png",
        "/mnt/hgfs/cloud/data/calibration/all_input/2019-07-01-02-41-44.png",
        "/mnt/hgfs/cloud/data/calibration/all_input/2019-07-01-02-47-39.png",
        "/mnt/hgfs/cloud/data/calibration/all_input/2019-07-01-02-53-49.png",
        "/mnt/hgfs/cloud/data/calibration/all_input/2019-07-01-02-59-43.png",
        "/mnt/hgfs/cloud/data/calibration/all_input/2019-07-01-03-05-53.png",
        "/mnt/hgfs/cloud/data/calibration/all_input/2019-07-01-03-11-47.png",
    ],

    # ---- 2020-01-01 (merged two blocks from your comments) ----
    "2020-01-01": [
        "/mnt/hgfs/cloud/data/calibration/all_input/2020-01-01-05-16-57.png",
        "/mnt/hgfs/cloud/data/calibration/all_input/2020-01-01-05-28-11.png",
        "/mnt/hgfs/cloud/data/calibration/all_input/2020-01-01-05-33-47.png",
        "/mnt/hgfs/cloud/data/calibration/all_input/2020-01-01-05-39-24.png",
        "/mnt/hgfs/cloud/data/calibration/all_input/2020-01-01-05-45-01.png",
        "/mnt/hgfs/cloud/data/calibration/all_input/2020-01-01-05-50-38.png",
        "/mnt/hgfs/cloud/data/calibration/all_input/2020-01-01-05-56-14.png",
        "/mnt/hgfs/cloud/data/calibration/all_input/2020-01-01-06-01-51.png",
        "/mnt/hgfs/cloud/data/calibration/all_input/2020-01-01-06-07-28.png",

        "/mnt/hgfs/cloud/data/calibration/all_input/2020-01-01-00-02-40.png",
        "/mnt/hgfs/cloud/data/calibration/all_input/2020-01-01-00-08-17.png",
        "/mnt/hgfs/cloud/data/calibration/all_input/2020-01-01-00-13-54.png",
        "/mnt/hgfs/cloud/data/calibration/all_input/2020-01-01-00-19-31.png",
        "/mnt/hgfs/cloud/data/calibration/all_input/2020-01-01-00-25-07.png",
        "/mnt/hgfs/cloud/data/calibration/all_input/2020-01-01-00-30-44.png",
        "/mnt/hgfs/cloud/data/calibration/all_input/2020-01-01-00-36-21.png",
        "/mnt/hgfs/cloud/data/calibration/all_input/2020-01-01-00-41-57.png",
        "/mnt/hgfs/cloud/data/calibration/all_input/2020-01-01-00-53-12.png",
        "/mnt/hgfs/cloud/data/calibration/all_input/2020-01-01-00-58-48.png",
        "/mnt/hgfs/cloud/data/calibration/all_input/2020-01-01-01-04-25.png",
        "/mnt/hgfs/cloud/data/calibration/all_input/2020-01-01-01-10-02.png",
        "/mnt/hgfs/cloud/data/calibration/all_input/2020-01-01-01-15-38.png",
        "/mnt/hgfs/cloud/data/calibration/all_input/2020-01-01-01-21-15.png",
        "/mnt/hgfs/cloud/data/calibration/all_input/2020-01-01-01-26-52.png",
        "/mnt/hgfs/cloud/data/calibration/all_input/2020-01-01-01-32-28.png",
        "/mnt/hgfs/cloud/data/calibration/all_input/2020-01-01-01-38-05.png",
        "/mnt/hgfs/cloud/data/calibration/all_input/2020-01-01-01-43-42.png",
        "/mnt/hgfs/cloud/data/calibration/all_input/2020-01-01-01-49-19.png",
        "/mnt/hgfs/cloud/data/calibration/all_input/2020-01-01-01-54-55.png",
        "/mnt/hgfs/cloud/data/calibration/all_input/2020-01-01-02-00-32.png",
        "/mnt/hgfs/cloud/data/calibration/all_input/2020-01-01-02-06-09.png",
        "/mnt/hgfs/cloud/data/calibration/all_input/2020-01-01-02-11-46.png",
        "/mnt/hgfs/cloud/data/calibration/all_input/2020-01-01-02-17-22.png",
        "/mnt/hgfs/cloud/data/calibration/all_input/2020-01-01-02-28-36.png",
        "/mnt/hgfs/cloud/data/calibration/all_input/2020-01-01-02-34-12.png",
        "/mnt/hgfs/cloud/data/calibration/all_input/2020-01-01-02-39-49.png",
        "/mnt/hgfs/cloud/data/calibration/all_input/2020-01-01-02-45-26.png",
        "/mnt/hgfs/cloud/data/calibration/all_input/2020-01-01-02-51-03.png",
        "/mnt/hgfs/cloud/data/calibration/all_input/2020-01-01-02-56-39.png",
        "/mnt/hgfs/cloud/data/calibration/all_input/2020-01-01-03-02-16.png",
        "/mnt/hgfs/cloud/data/calibration/all_input/2020-01-01-03-07-53.png",
        "/mnt/hgfs/cloud/data/calibration/all_input/2020-01-01-03-13-29.png",
        "/mnt/hgfs/cloud/data/calibration/all_input/2020-01-01-03-19-06.png",
        "/mnt/hgfs/cloud/data/calibration/all_input/2020-01-01-03-24-43.png",
        "/mnt/hgfs/cloud/data/calibration/all_input/2020-01-01-03-30-20.png",
        "/mnt/hgfs/cloud/data/calibration/all_input/2020-01-01-03-35-56.png",
        "/mnt/hgfs/cloud/data/calibration/all_input/2020-01-01-03-41-33.png",
        "/mnt/hgfs/cloud/data/calibration/all_input/2020-01-01-03-47-10.png",
        "/mnt/hgfs/cloud/data/calibration/all_input/2020-01-01-03-52-47.png",
        "/mnt/hgfs/cloud/data/calibration/all_input/2020-01-01-03-58-23.png",
        "/mnt/hgfs/cloud/data/calibration/all_input/2020-01-01-04-04-00.png",
    ],

    # ---- 2023-11-14 ----
    "2023-11-14": [
        "/mnt/hgfs/cloud/data/calibration/all_input/2023-11-14-00-07-45.png",
        "/mnt/hgfs/cloud/data/calibration/all_input/2023-11-14-00-13-19.png",
        "/mnt/hgfs/cloud/data/calibration/all_input/2023-11-14-00-18-54.png",
        "/mnt/hgfs/cloud/data/calibration/all_input/2023-11-14-00-24-28.png",
        "/mnt/hgfs/cloud/data/calibration/all_input/2023-11-14-00-30-10.png",
        "/mnt/hgfs/cloud/data/calibration/all_input/2023-11-14-00-35-44.png",
        "/mnt/hgfs/cloud/data/calibration/all_input/2023-11-14-00-41-19.png",
        "/mnt/hgfs/cloud/data/calibration/all_input/2023-11-14-00-46-54.png",
        "/mnt/hgfs/cloud/data/calibration/all_input/2023-11-14-00-52-28.png",
        "/mnt/hgfs/cloud/data/calibration/all_input/2023-11-14-00-58-03.png",
        "/mnt/hgfs/cloud/data/calibration/all_input/2023-11-14-01-03-37.png",
        "/mnt/hgfs/cloud/data/calibration/all_input/2023-11-14-01-09-12.png",
        "/mnt/hgfs/cloud/data/calibration/all_input/2023-11-14-01-14-46.png",
        "/mnt/hgfs/cloud/data/calibration/all_input/2023-11-14-01-20-21.png",
        "/mnt/hgfs/cloud/data/calibration/all_input/2023-11-14-01-25-55.png",
        "/mnt/hgfs/cloud/data/calibration/all_input/2023-11-14-01-31-30.png",
        "/mnt/hgfs/cloud/data/calibration/all_input/2023-11-14-01-37-04.png",
        "/mnt/hgfs/cloud/data/calibration/all_input/2023-11-14-01-42-39.png",
        "/mnt/hgfs/cloud/data/calibration/all_input/2023-11-14-01-48-13.png",
        "/mnt/hgfs/cloud/data/calibration/all_input/2023-11-14-01-53-48.png",
        "/mnt/hgfs/cloud/data/calibration/all_input/2023-11-14-01-59-23.png",
    # ],

    # # ---- 2024-04-08 (merged both blocks) ----
    # "2024-04-08": [
        "/mnt/hgfs/cloud/data/calibration/all_input/2024-04-08-04-41-34.png",
        "/mnt/hgfs/cloud/data/calibration/all_input/2024-04-08-04-47-09.png",
        "/mnt/hgfs/cloud/data/calibration/all_input/2024-04-08-04-52-43.png",
        "/mnt/hgfs/cloud/data/calibration/all_input/2024-04-08-04-58-18.png",
        "/mnt/hgfs/cloud/data/calibration/all_input/2024-04-08-05-03-52.png",
        "/mnt/hgfs/cloud/data/calibration/all_input/2024-04-08-05-09-27.png",
        "/mnt/hgfs/cloud/data/calibration/all_input/2024-04-08-05-20-36.png",
        "/mnt/hgfs/cloud/data/calibration/all_input/2024-04-08-05-26-11.png",
        "/mnt/hgfs/cloud/data/calibration/all_input/2024-04-08-05-31-45.png",
        "/mnt/hgfs/cloud/data/calibration/all_input/2024-04-08-05-37-20.png",
        "/mnt/hgfs/cloud/data/calibration/all_input/2024-04-08-05-42-54.png",
        "/mnt/hgfs/cloud/data/calibration/all_input/2024-04-08-05-48-29.png",
        "/mnt/hgfs/cloud/data/calibration/all_input/2024-04-08-05-54-10.png",

        "/mnt/hgfs/cloud/data/calibration/all_input/2024-04-08-00-03-48.png",
        "/mnt/hgfs/cloud/data/calibration/all_input/2024-04-08-00-09-29.png",
        "/mnt/hgfs/cloud/data/calibration/all_input/2024-04-08-00-15-10.png",
        "/mnt/hgfs/cloud/data/calibration/all_input/2024-04-08-00-20-52.png",
        "/mnt/hgfs/cloud/data/calibration/all_input/2024-04-08-00-26-33.png",
        "/mnt/hgfs/cloud/data/calibration/all_input/2024-04-08-00-32-14.png",
        "/mnt/hgfs/cloud/data/calibration/all_input/2024-04-08-00-37-55.png",
        "/mnt/hgfs/cloud/data/calibration/all_input/2024-04-08-00-43-37.png",
        "/mnt/hgfs/cloud/data/calibration/all_input/2024-04-08-00-49-19.png",
        "/mnt/hgfs/cloud/data/calibration/all_input/2024-04-08-01-06-23.png",
        "/mnt/hgfs/cloud/data/calibration/all_input/2024-04-08-01-12-04.png",
        "/mnt/hgfs/cloud/data/calibration/all_input/2024-04-08-01-17-46.png",
    ],
}

# ---------------------------------------------------------------------
# 2) Setup history (from your table)
# ---------------------------------------------------------------------
@dataclass(frozen=True)
class TimeSlot:
    start_file: str               # "YYYY-MM-DD-HH-MM-SS"
    need_new_astrometry: int      # 0/1
    special_class: Optional[str]  # e.g., "l" or "u" per your last two rows

HISTORY: List[TimeSlot] = [
    TimeSlot("2018-05-01-00-02-44", 1, None),
    TimeSlot("2018-09-27-19-19-49", 1, None),
    TimeSlot("2019-04-24-15-39-36", 1, None),
    TimeSlot("2019-06-26-18-23-18", 1, None),
    TimeSlot("2019-07-05-11-59-14", 1, None),
    TimeSlot("2020-08-26-17-24-23", 0, None),
    TimeSlot("2020-09-16-15-30-23", 0, None),
    TimeSlot("2020-09-23-12-18-53", 0, None),
    TimeSlot("2020-10-18-14-31-56", 0, None),
    TimeSlot("2021-05-10-14-39-39", 0, None),
    TimeSlot("2022-06-01-19-41-30", 0, None),
    TimeSlot("2023-03-27-11-15-06", 0, None),
    TimeSlot("2023-09-27-18-09-48", 1, "l"),
    # TimeSlot("2023-09-27-20-47-06", 0, "u"),
]

# =========================
# Utilities (mostly your originals, lightly adapted)
# =========================
def load_image(image_path: str) -> np.ndarray:
    img_color = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img_color is None:
        raise FileNotFoundError(f"Cannot read {image_path}")
    gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray = cv2.GaussianBlur(gray, ksize=(15, 15), sigmaX=0)
    return gray

def to_astropy_time_utc8(ts: str, *, scale: str = "utc") -> Time:
    dt_local = datetime.strptime(ts, "%Y-%m-%d-%H-%M-%S").replace(tzinfo=UTC_PLUS_8)
    return Time(dt_local.astimezone(timezone.utc), scale=scale)

def observation_conditions(time_str: str, site_lat: float, site_lon: float, site_alt: float):
    obstime = to_astropy_time_utc8(time_str)
    location = EarthLocation(lat=site_lat*units.deg, lon=site_lon*units.deg, height=site_alt*units.m)
    altaz_frame = AltAz(obstime=obstime, location=location)
    return obstime, location, altaz_frame

def uv2rphi(u: np.ndarray, v: np.ndarray, u0: float, v0: float) -> Tuple[np.ndarray, np.ndarray]:
    du = u - u0
    dv = v - v0
    r = np.sqrt(du**2 + dv**2)
    phi = np.arctan2(dv, du) % (2*np.pi)
    return r, phi

def rphi2uv(r: np.ndarray, phi: np.ndarray, u0: float, v0: float) -> Tuple[np.ndarray, np.ndarray]:
    u = r*np.cos(phi) + u0
    v = r*np.sin(phi) + v0
    return u, v

def header_dict_to_wcs(hdr_dict: Dict[str, Any]) -> astropy.wcs.WCS:
    hdr = fits.Header()
    for k, v in hdr_dict.items():
        if isinstance(v, (np.floating, np.integer)):
            v = v.item()
        hdr[k] = v
    return astropy.wcs.WCS(hdr)

def make_ra_dec_query_from_json(json_path: str):
    with open(json_path, "r", encoding="utf-8") as f:
        blob = json.load(f)

    meta = blob.get("meta", {})
    results = blob.get("result", {})

    CCAM = meta.get("CENTER", [np.nan, np.nan])
    R = meta.get("R", np.nan)
    nside = meta.get("NSIDE", 4)
    proj_types_present: List[str] = meta.get(
        "PROJECTION_TYPES",
        list(results[next(iter(results))].keys()) if results else ['SIN']
    )
    CCAM_u0, CCAM_v0 = float(CCAM[0]), float(CCAM[1])
    f0 = float(R)

    wcs_cache: Dict[Tuple[int, str], astropy.wcs.WCS] = {}
    center_cache: Dict[Tuple[int, str], Tuple[float, float]] = {}
    for k_str, per_proj in results.items():
        hpix = int(k_str)
        for proj_type, payload in per_proj.items():
            theta_c = payload.get("theta_center", None)
            phi_c = payload.get("phi_center", None)
            if theta_c is not None and phi_c is not None:
                center_cache[(hpix, proj_type)] = (float(theta_c), float(phi_c))
            if "WCS" in payload:
                try:
                    w = header_dict_to_wcs(payload["WCS"])
                    wcs_cache[(hpix, proj_type)] = w
                except Exception:
                    pass

    def r2theta(r):
        return np.arcsin(r / f0)

    def theta2r(theta):
        return f0 * np.sin(theta)

    def query(x_vec: np.ndarray, y_vec: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        x_vec = np.asarray(x_vec, dtype=float)
        y_vec = np.asarray(y_vec, dtype=float)
        n = x_vec.size
        ra_out = np.full(n, np.nan, dtype=float)
        dec_out = np.full(n, np.nan, dtype=float)

        r, phi = uv2rphi(x_vec, y_vec, CCAM_u0, CCAM_v0)
        theta = r2theta(r)

        for i in range(n):
            th = theta[i]
            ph = phi[i]
            if not (np.isfinite(th) and 0 <= th <= np.pi and np.isfinite(ph)):
                continue
            try:
                hpix = int(hp.ang2pix(nside, th, ph))
            except Exception:
                continue

            for proj_type in proj_types_present:
                key = (hpix, proj_type)
                if key not in wcs_cache or key not in center_cache:
                    continue
                theta_c, phi_c = center_cache[key]
                xyz = hp.ang2vec(th, ph)[None, :]
                Rz = np.array([[np.cos(-phi_c), -np.sin(-phi_c), 0],
                               [np.sin(-phi_c),  np.cos(-phi_c), 0],
                               [0,                0,              1]])
                Ry = np.array([[np.cos(-theta_c), 0, np.sin(-theta_c)],
                               [0,                 1, 0],
                               [-np.sin(-theta_c), 0, np.cos(-theta_c)]])
                Rmat = Ry @ Rz
                xyz_rot = xyz @ Rmat.T
                thp, php = hp.vec2ang(xyz_rot)
                rp = theta2r(thp)
                up, vp = rphi2uv(np.array([rp]), np.array([php]), CCAM_u0, CCAM_v0)
                try:
                    ra, dec = wcs_cache[key].wcs_pix2world([[float(up[0]), float(vp[0])]], 0)[0]
                    ra_out[i] = float(ra)
                    dec_out[i] = float(dec)
                    break
                except Exception:
                    continue
        return ra_out, dec_out

    return query

# ----------------- Circular stats -----------------
def circ_mean_rad(vals: np.ndarray) -> float:
    vals = np.asarray(vals, dtype=float)
    C = np.nanmean(np.cos(vals))
    S = np.nanmean(np.sin(vals))
    return float(np.arctan2(S, C) % (2*np.pi))

def circ_median_rad(vals: np.ndarray) -> float:
    vals = np.asarray(vals, dtype=float)
    vals = vals[~np.isnan(vals)]
    if vals.size == 0:
        return np.nan
    cand = np.linspace(0, 2*np.pi, 361, endpoint=True)
    def circ_dist(a, b):
        d = np.abs((a-b+np.pi)%(2*np.pi)-np.pi)
        return d
    costs = [np.sum(circ_dist(vals, c)) for c in cand]
    return float(cand[int(np.argmin(costs))])

# ----------------- Model fitting -----------------
def build_design_z(z: np.ndarray, max_odd_order: int) -> np.ndarray:
    assert max_odd_order % 2 == 1 and max_odd_order >= 1
    cols = [(z**k) for k in range(1, max_odd_order+1, 2)]
    return np.vstack(cols).T

def fit_r_of_z(r: np.ndarray, z: np.ndarray, max_odd_order: int) -> np.ndarray:
    A = build_design_z(z, max_odd_order)
    a, *_ = np.linalg.lstsq(A, r, rcond=None)
    return a

def eval_r_of_z(z: np.ndarray, a: np.ndarray) -> np.ndarray:
    r = np.zeros_like(z)
    k = 1
    for ai in a:
        r += ai * (z**k)
        k += 2
    return r

def invert_r_to_z(r: np.ndarray, a: np.ndarray) -> np.ndarray:
    r = np.asarray(r, dtype=float)
    a1 = a[0] if a.size >= 1 else 1.0
    z = np.clip(r / max(a1, 1e-9), 0.0, np.pi/2)
    for _ in range(NEWTON_MAX_ITERS):
        rz = eval_r_of_z(z, a)
        f = rz - r
        k = 1
        drdz = np.zeros_like(z)
        for ai in a:
            drdz += ai * k * (z**(k-1))
            k += 2
        drdz = np.where(np.abs(drdz) < 1e-12, 1e-12, drdz)
        z_new = np.clip(z - f/drdz, 0.0, np.pi/2)
        if np.nanmax(np.abs(z_new - z)) < NEWTON_TOL:
            z = z_new
            break
        z = z_new
    return z

def fit_az_offset(az_rad: np.ndarray, img_angle_from_y_rad: np.ndarray) -> float:
    diff = (az_rad - img_angle_from_y_rad + np.pi) % (2*np.pi) - np.pi
    C = np.nanmean(np.cos(diff))
    S = np.nanmean(np.sin(diff))
    return float(np.arctan2(S, C))

# ----------------- Joint data collection -----------------
def grid_points(img_size: int, box: int, x_start: int, y_start: int):
    x = x_start + np.arange(0, img_size, box)
    y = y_start + np.arange(0, img_size, box)
    X, Y = np.meshgrid(x, y)
    return X.ravel(), Y.ravel()

def collect_joint_samples(input_img_list: List[str]) -> Dict[Tuple[int,int], Dict[str, List[float]]]:
    xq, yq = grid_points(IMG_SIZE, BOX, X_START, Y_START)
    nx = len(np.unique(X_START + np.arange(0, IMG_SIZE, BOX)))
    ny = len(np.unique(Y_START + np.arange(0, IMG_SIZE, BOX)))

    def idx_from_uv(u, v):
        ix = int(round((u - X_START)/BOX))
        iy = int(round((v - Y_START)/BOX))
        return ix, iy

    store: Dict[Tuple[int,int], Dict[str, List[float]]] = {}
    for path in input_img_list:
        img_name = os.path.basename(path)
        stamp = os.path.splitext(img_name)[0]
        json_path = os.path.join(TEMP_DIR, f"{stamp}_astrometry.json")
        query = make_ra_dec_query_from_json(json_path)
        ra, dec = query(xq, yq)
        _, _, altaz_frame = observation_conditions(stamp, SITE_LAT, SITE_LON, SITE_ALT)
        sky = astropy.coordinates.SkyCoord(ra=ra*units.deg, dec=dec*units.deg, frame='icrs')
        altaz = sky.transform_to(altaz_frame)
        az = altaz.az.rad
        alt = altaz.alt.rad

        valid = (~np.isnan(az)) & (~np.isnan(alt))
        u_valid = xq[valid]; v_valid = yq[valid]
        az_valid = az[valid]; alt_valid = alt[valid]

        for u, v, aaz, aalt in zip(u_valid, v_valid, az_valid, alt_valid):
            ix, iy = idx_from_uv(u, v)
            key = (ix, iy)
            if key not in store:
                store[key] = {"az_list": [], "alt_list": []}
            store[key]["az_list"].append(float(aaz))
            store[key]["alt_list"].append(float(aalt))

    store["_shape"] = {"nx": nx, "ny": ny, "xq": xq.tolist(), "yq": yq.tolist()}
    return store

# ----------------- Zenith (center) from joint data -----------------
def get_center_from_joint(store: Dict[Tuple[int,int], Dict[str, List[float]]],
                          input_img_list: List[str],
                          top_n: int = 1) -> Tuple[float, float]:
    xq = np.array(store["_shape"]["xq"])
    yq = np.array(store["_shape"]["yq"])

    block_max_alt = np.full(xq.shape, -np.inf)
    for i in range(xq.size):
        ix = int(round((xq[i] - X_START)/BOX))
        iy = int(round((yq[i] - Y_START)/BOX))
        key = (ix, iy)
        if key in store and store[key]["alt_list"]:
            block_max_alt[i] = np.max(store[key]["alt_list"])
    seed_idx = np.nanargmax(block_max_alt)
    seed_u, seed_v = xq[seed_idx], yq[seed_idx]

    xg = seed_u + np.arange(-32, 33, 1)
    yg = seed_v + np.arange(-32, 33, 1)
    Xg, Yg = np.meshgrid(xg, yg)
    xg_flat = Xg.ravel(); yg_flat = Yg.ravel()

    alts_all = []
    for path in input_img_list:
        stamp = os.path.splitext(os.path.basename(path))[0]
        query = make_ra_dec_query_from_json(os.path.join(TEMP_DIR, f"{stamp}_astrometry.json"))
        ra, dec = query(xg_flat, yg_flat)
        _, _, altaz_frame = observation_conditions(stamp, SITE_LAT, SITE_LON, SITE_ALT)
        sky = astropy.coordinates.SkyCoord(ra=ra*units.deg, dec=dec*units.deg, frame='icrs')
        altaz = sky.transform_to(altaz_frame)
        alts_all.append(altaz.alt.deg)

    A = np.array(alts_all)
    with np.errstate(invalid='ignore'):
        alt_med = np.nanmedian(A, axis=0)

    idx_top = np.argsort(alt_med)[-top_n:]
    u0 = float(np.mean(xg_flat[idx_top]))
    v0 = float(np.mean(yg_flat[idx_top]))
    return u0, v0

# ----------------- Build training arrays -----------------
def build_training_from_store(store: Dict[Tuple[int,int], Dict[str, List[float]]],
                              u0: float, v0: float, mode: str):
    xq = np.array(store["_shape"]["xq"])
    yq = np.array(store["_shape"]["yq"])

    r_list, z_list, imgang_list, az_list = [], [], [], []

    for i in range(xq.size):
        u = xq[i]; v = yq[i]
        ix = int(round((u - X_START)/BOX))
        iy = int(round((v - Y_START)/BOX))
        key = (ix, iy)
        if key not in store:
            continue
        alts = np.array(store[key]["alt_list"])
        azs  = np.array(store[key]["az_list"])
        if alts.size == 0:
            continue

        r, phi = uv2rphi(np.array([u]), np.array([v]), u0, v0)
        r = r[0]
        img_angle_from_y = (np.pi/2 - phi[0]) % (2*np.pi)

        if mode == "all":
            z = (np.pi/2) - alts
            r_list.append(np.full_like(z, r))
            imgang_list.append(np.full_like(z, img_angle_from_y))
            z_list.append(z)
            az_list.append(azs)
        else:
            alt_med = np.nanmedian(alts)
            az_med = circ_median_rad(azs)
            z = (np.pi/2) - alt_med
            r_list.append(np.array([r]))
            imgang_list.append(np.array([img_angle_from_y]))
            z_list.append(np.array([z]))
            az_list.append(np.array([az_med]))

    if not r_list:
        raise RuntimeError("No training samples found (after aggregation).")

    r = np.concatenate(r_list)
    z = np.concatenate(z_list)
    imgang = np.concatenate(imgang_list)
    az = np.concatenate(az_list)
    return r, z, imgang, az

# ----------------- Diagnostics and visualization -----------------
def _wrap_deg180(x_deg: np.ndarray) -> np.ndarray:
    return ((x_deg + 180.0) % 360.0) - 180.0

def plot_az_fit(az_rad: np.ndarray, imgang_from_y_rad: np.ndarray, az_offset: float,
                save_path: Optional[str] = None) -> float:
    az_meas_deg = np.degrees(az_rad % (2*np.pi))
    az_pred_deg = np.degrees((imgang_from_y_rad + az_offset) % (2*np.pi))
    resid_deg = _wrap_deg180(az_pred_deg - az_meas_deg)
    circ_rmse_deg = float(np.degrees(np.sqrt(np.mean(np.radians(resid_deg) ** 2))))

    fig, axes = plt.subplots(1, 1, figsize=(6, 5), constrained_layout=True)
    # axes[0].scatter(az_meas_deg, az_pred_deg, s=8, alpha=0.35)
    # axes[0].plot([0, 360], [0, 360], 'r--', lw=1)
    # axes[0].set_xlim(0, 360); axes[0].set_ylim(0, 360)
    # axes[0].set_xlabel("Measured az (deg)",fontsize = 15); axes[0].set_ylabel("Predicted az (deg)",fontsize = 15)
    # axes[0].set_title(f"Azimuth fit (Δ = {np.degrees(az_offset):.3f}°)",fontsize = 15)

    # axes[0].scatter(az_meas_deg, resid_deg, s=8, alpha=0.35)
    # axes[0].axhline(0, color='r', linestyle='--', lw=1)
    # axes[0].set_xlim(0, 360); axes[0].set_ylim(-180, 180)
    # axes[0].set_xlabel("Measured az (deg)",fontsize = 15); axes[0].set_ylabel("Residual (pred - meas) [deg]",fontsize = 15)
    # axes[0].set_title(f"Azimuth residuals (circular RMSE ≈ {circ_rmse_deg:.3f}°)",fontsize = 15)
    axes.scatter(az_meas_deg, resid_deg, s=8, alpha=0.35)
    axes.axhline(0, color='r', linestyle='--', lw=1)
    axes.set_xlim(0, 360); axes.set_ylim(-180, 180)
    axes.set_xlabel("Measured az (deg)",fontsize = 15); axes.set_ylabel("Residual (pred - meas) [deg]",fontsize = 15)
        # ylim set to pm 5sigma
    axes.set_ylim(-5*resid_deg.std(), + 5*resid_deg.std())
    axes.set_title(f"Azimuth residuals (circular RMSE ≈ {circ_rmse_deg:.3f}°)",fontsize = 15)
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
    else:
        plt.show()
    return circ_rmse_deg

def plot_r_fit(r_meas: np.ndarray, z_rad: np.ndarray, a_coeffs: np.ndarray,
               save_path: Optional[str] = None) -> float:
    r_pred = eval_r_of_z(z_rad, a_coeffs)
    resid = r_meas - r_pred
    rmse_px = float(np.sqrt(np.nanmean(resid**2)))

    z_deg = np.degrees(z_rad)

    fig, axes = plt.subplots(1, 1, figsize=(6, 5), constrained_layout=True)
            # axes[0].scatter(r_meas, r_pred, s=8, alpha=0.35)
            # mn = float(np.nanmin([np.nanmin(r_meas), np.nanmin(r_pred)]))
            # mx = float(np.nanmax([np.nanmax(r_meas), np.nanmax(r_pred)]))
            # axes[0].plot([mn, mx], [mn, mx], 'r--', lw=1)
            # axes[0].set_xlabel("Measured r (pixels)",fontsize = 15); axes[0].set_ylabel("Predicted r (pixels)",fontsize = 15)
            # axes[0].set_title("r(z) fit: measured vs predicted",fontsize = 15)

    # axes[0].scatter(z_deg, resid, s=8, alpha=0.35)
    # axes[0].axhline(0, color='r', linestyle='--', lw=1)
    # axes[0].set_xlabel("z (deg)",fontsize = 15); axes[0].set_ylabel("Residual (meas - pred) [pixels]",fontsize = 15)
    # axes[0].set_title(f"r(z) residuals (RMSE ≈ {rmse_px:.3f} px)",fontsize = 15)
    axes.scatter(z_deg, resid, s=8, alpha=0.35)
    axes.axhline(0, color='r', linestyle='--', lw=1)
    axes.set_xlabel("z (deg)",fontsize = 15); axes.set_ylabel("Residual (meas - pred) [pixels]",fontsize = 15)
    # ylim set to pm 5sigma
    axes.set_ylim(-5*resid.std(), + 5*resid.std())
    axes.set_title(f"r(z) residuals (RMSE ≈ {rmse_px:.3f} px)",fontsize = 15)
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
    else:
        plt.show()
    return rmse_px

def infer_joint_and_visualize(store, u0, v0, a_coeffs, az_offset,
                              background_image_path, mode: str,
                              save_path: Optional[str] = None):
    xq = np.array(store["_shape"]["xq"])
    yq = np.array(store["_shape"]["yq"])

    alt_disp = np.full(xq.shape, np.nan)
    az_disp  = np.full(xq.shape, np.nan)
    inferred = np.zeros(xq.shape, dtype=bool)

    for i in range(xq.size):
        u = xq[i]; v = yq[i]
        ix = int(round((u - X_START)/BOX))
        iy = int(round((v - Y_START)/BOX))
        key = (ix, iy)
        if key in store and len(store[key]["alt_list"]) > 0:
            alts = np.array(store[key]["alt_list"])
            azs  = np.array(store[key]["az_list"])
            if mode == "all":
                alt_disp[i] = np.degrees(np.nanmean(alts))
                az_disp[i]  = np.degrees(circ_mean_rad(azs))
            else:
                alt_disp[i] = np.degrees(np.nanmedian(alts))
                az_disp[i]  = np.degrees(circ_median_rad(azs))

    miss = np.isnan(alt_disp)
    if np.any(miss):
        um = xq[miss]; vm = yq[miss]
        r_m, phi_m = uv2rphi(um, vm, u0, v0)
        z_m = invert_r_to_z(r_m, a_coeffs)
        alt_m = (np.pi/2) - z_m
        img_angle_from_y_m = (np.pi/2 - phi_m) % (2*np.pi)
        az_m = (img_angle_from_y_m + az_offset) % (2*np.pi)
        alt_disp[miss] = np.degrees(alt_m)
        az_disp[miss]  = np.degrees(az_m)
        inferred[miss] = True

    img = load_image(background_image_path)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(img, cmap='gray', vmin=0, vmax=50)
    norm = Normalize(vmin=0, vmax=90)
    cmap = plt.cm.viridis
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
    fig.colorbar(sm, ax=ax, label='alt (degrees)')

    #ax.set_title(f"Joint alt/az grid ({mode}) – inferred squares outlined red",fontsize = 15)
    ax.set_xlabel("X (pixels)",fontsize = 15)
    ax.set_ylabel("Y (pixels)",fontsize = 15)
    ax.set_xlim(0, IMG_SIZE)
    ax.set_ylim(0, IMG_SIZE)
    ax.grid(color='white', linestyle='--', alpha=0.4)

    half = BOX/2
    for i in range(xq.size):
        if not np.isfinite(alt_disp[i]):
            continue
        rect = plt.Rectangle(
            (xq[i]-half, yq[i]-half),
            BOX, BOX,
            linewidth=1.0 if inferred[i] else 0.0,
            edgecolor='red' if inferred[i] else 'none',
            facecolor=cmap(norm(alt_disp[i])),
            fill=True
        )
        ax.add_patch(rect)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path+"_alt.pdf")
        plt.close(fig)
    else:
        plt.show()
    #-----------------------------------------------------------------------
    img = load_image(background_image_path)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(img, cmap='gray', vmin=0, vmax=50)
    norm = Normalize(vmin=0, vmax=360)
    cmap = plt.cm.viridis
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
    fig.colorbar(sm, ax=ax, label='az (degrees)')

    #ax.set_title(f"Joint alt/az grid ({mode}) – inferred squares outlined red",fontsize = 15)
    ax.set_xlabel("X (pixels)",fontsize = 15)
    ax.set_ylabel("Y (pixels)",fontsize = 15)
    ax.set_xlim(0, IMG_SIZE)
    ax.set_ylim(0, IMG_SIZE)
    ax.grid(color='white', linestyle='--', alpha=0.4)

    half = BOX/2
    for i in range(xq.size):
        if not np.isfinite(az_disp[i]):
            continue
        rect = plt.Rectangle(
            (xq[i]-half, yq[i]-half),
            BOX, BOX,
            linewidth=1.0 if inferred[i] else 0.0,
            edgecolor='red' if inferred[i] else 'none',
            facecolor=cmap(norm(az_disp[i])),
            fill=True
        )
        ax.add_patch(rect)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path+"_az.pdf")#, dpi=150)
        plt.close(fig)
    else:
        plt.show()
# =========================
# Run manager / persistence
# =========================
def parse_stamp_local(ts: str) -> datetime:
    # "YYYY-MM-DD-HH-MM-SS" in UTC+8 local time
    return datetime.strptime(ts, "%Y-%m-%d-%H-%M-%S").replace(tzinfo=UTC_PLUS_8)

def image_stamp(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]

def align_to_timeslot(stamp_str: str, history: List[TimeSlot]) -> TimeSlot:
    """Pick the most recent slot with start <= image timestamp."""
    t = parse_stamp_local(stamp_str)
    slots = sorted(history, key=lambda s: parse_stamp_local(s.start_file))
    chosen = None
    for s in slots:
        if parse_stamp_local(s.start_file) <= t:
            chosen = s
        else:
            break
    if chosen is None:
        # earlier than first slot -> use first
        chosen = slots[0]
    return chosen

def save_calibration(slot: TimeSlot,
                     u0: float, v0: float,
                     a_coeffs: np.ndarray, az_offset: float,
                     training_meta: Dict[str, Any],
                     az_plot_path: str, r_plot_path: str, grid_preview_path: Optional[str]) -> str:
    out = {
        "time_slot": {
            "start_file": slot.start_file,
            "need_new_astrometry": slot.need_new_astrometry,
            "special_class": slot.special_class
        },
        "site": {"lat_deg": SITE_LAT, "lon_deg": SITE_LON, "alt_m": SITE_ALT},
        "fit": {
            "img_size": IMG_SIZE,
            "box": BOX,
            "x_start": X_START,
            "y_start": Y_START,
            "agg_mode": AGG_MODE,
            "max_odd_order": MAX_ODD_ORDER,
            "newton_max_iters": NEWTON_MAX_ITERS,
            "newton_tol": NEWTON_TOL,
            "u0": float(u0),
            "v0": float(v0),
            "a_coeffs": [float(x) for x in a_coeffs.tolist()],
            "az_offset_rad": float(az_offset),
        },
        "training_meta": training_meta,
        "diagnostics": {
            "az_fit_png": os.path.basename(az_plot_path),
            "r_fit_png": os.path.basename(r_plot_path),
            "grid_preview_png": os.path.basename(grid_preview_path) if grid_preview_path else None
        },
        "created_utc": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    }
    out_path = os.path.join(CAL_DIR, f"{slot.start_file}_calibration.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    return out_path

def build_index(calibration_files: List[str], index_path: str) -> None:
    """Builds an index sorted by slot start time with need_new_astrometry = 1."""
    entries = []
    for p in calibration_files:
        try:
            with open(p, "r", encoding="utf-8") as f:
                blob = json.load(f)
            slot = blob["time_slot"]
            if int(slot["need_new_astrometry"]) != 1:
                continue
            entries.append({
                "start_file": slot["start_file"],
                "special_class": slot.get("special_class"),
                "file": os.path.basename(p),
            })
        except Exception:
            continue
    entries.sort(key=lambda e: parse_stamp_local(e["start_file"]))
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump({"calibrations": entries}, f, indent=2)

def run_one_set(tag: str, input_img_list: List[str],
                force_visual_preview: bool = False,
                overwrite_existing: bool = False) -> Optional[str]:
    """
    Runs the full pipeline for one image set. Saves calibration only
    if the aligned slot has need_new_astrometry=1. Returns path to
    saved calibration JSON (or None if not saved).
    """
    # choose the alignment slot based on the earliest image in the set
    stamps = [image_stamp(p) for p in input_img_list]
    stamps.sort()
    slot = align_to_timeslot(stamps[0], HISTORY)
    slot_json_path = os.path.join(CAL_DIR, f"{slot.start_file}_calibration.json")

    if os.path.exists(slot_json_path) and not overwrite_existing:
        print(f"[{tag}] Slot {slot.start_file} already has a calibration: {slot_json_path} (skipping save).")

    # --- run the fit
    store = collect_joint_samples(input_img_list)
    u0, v0 = get_center_from_joint(store, input_img_list)
    print(f"[{tag}] [Center] u0={u0:.2f}, v0={v0:.2f}")

    r, z, imgang, az = build_training_from_store(store, u0, v0, AGG_MODE)
    a_coeffs = fit_r_of_z(r, z, MAX_ODD_ORDER)
    az_offset = fit_az_offset(az, imgang)

    terms = [f"a{k}={a:.6e}" for k, a in zip(range(1, MAX_ODD_ORDER+1, 2), a_coeffs)]
    print(f"[{tag}] [r(z) fit] " + ", ".join(terms))
    print(f"[{tag}] [az offset] Δ = {np.degrees(az_offset):.3f} deg")

    # diagnostics
    az_fit_png = os.path.join(CAL_DIR, f"{slot.start_file}_azfit.pdf")
    r_fit_png  = os.path.join(CAL_DIR, f"{slot.start_file}_rfit.pdf")
    az_rmse = plot_az_fit(az, imgang, az_offset, save_path=az_fit_png)
    r_rmse  = plot_r_fit(r, z, a_coeffs, save_path=r_fit_png)
    print(f"[{tag}] [az plot] saved to {az_fit_png} (circ RMSE ≈ {az_rmse:.3f}°)")
    print(f"[{tag}] [r(z) plot] saved to {r_fit_png} (RMSE ≈ {r_rmse:.3f} px)")

    grid_preview_png = None
    if force_visual_preview:
        grid_preview_png = os.path.join(CAL_DIR, f"{slot.start_file}_grid_preview")
        infer_joint_and_visualize(store, u0, v0, a_coeffs, az_offset,
                                  background_image_path=input_img_list[0],
                                  mode=AGG_MODE, save_path=grid_preview_png)
        print(f"[{tag}] [grid preview] saved to {grid_preview_png}")

    # meta to save
    training_meta = {
        "input_image_count": len(input_img_list),
        "input_images": [os.path.basename(p) for p in input_img_list],
        "training_samples": int(r.size),
        "az_rmse_deg": az_rmse,
        "r_rmse_px": r_rmse,
    }

    saved_path = None
    if slot.need_new_astrometry == 1:
        saved_path = save_calibration(slot, u0, v0, a_coeffs, az_offset,
                                      training_meta, az_fit_png, r_fit_png, grid_preview_png)
        print(f"[{tag}] [saved] calibration -> {saved_path}")
    else:
        print(f"[{tag}] Slot {slot.start_file} has need_new_astrometry=0; not saving a new file.")

    return saved_path

def main():
    saved_files = []
    # Process every tested set
    for tag, paths in ALL_IMAGE_SETS.items():
        try:
            saved = run_one_set(tag, paths, force_visual_preview=True, overwrite_existing=False)
            if saved:
                saved_files.append(saved)
        except Exception as e:
            print(f"[{tag}] ERROR: {e}")

    # Build/load index only from files that actually exist
    existing = []
    for ts in HISTORY:
        p = os.path.join(CAL_DIR, f"{ts.start_file}_calibration.json")
        if os.path.exists(p):
            existing.append(p)

    index_path = os.path.join(CAL_DIR, "calibration_index.json")
    build_index(existing, index_path)
    print(f"[index] wrote {index_path}")

if __name__ == "__main__":
    main()
