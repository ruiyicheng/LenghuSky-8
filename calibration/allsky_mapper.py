#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
allsky_mapper.py

Loads calibration files saved by `calibrate_and_save.py` and provides
a mapper class with:
  - altaz_to_pixel(date, alt, az, resolution=4096)
  - pixel_to_altaz(date, u, v, resolution=4096)

Inputs can be scalars, Python lists, or NumPy arrays.
Resolution can be an int (N for NxN) or a 2-tuple (W, H).
For non-square resolutions, an isotropic scale using min(W, H) is applied.

Coordinate conventions are consistent with the fit pipeline:
  - alt in degrees (0..90), az in degrees (0..360, north-through-east).
  - pixel (u, v): origin at top-left, u increases to right, v increases downward.
"""

import os
import json
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta

import numpy as np

# Match the base choices used in the fit
UTC_PLUS_8 = timezone(timedelta(hours=8))
BASE_IMG_SIZE = 4096

@dataclass
class CalibFit:
    start_file: str
    special_class: Optional[str]
    img_size: int
    u0: float
    v0: float
    a: np.ndarray         # polynomial coefficients [a1, a3, a5, ...] in pixels at base scale
    az_offset: float      # radians

class AllSkyMapper:
    def __init__(self, cal_dir: str):
        """
        cal_dir: where calibrations and calibration_index.json are stored
                 (TEMP_DIR/calibrations from the fitter).
        """
        self.cal_dir = cal_dir
        self._fits: List[CalibFit] = []
        self._load_index_and_fits()

    @staticmethod
    def _parse_local(ts: str) -> datetime:
        return datetime.strptime(ts, "%Y-%m-%d-%H-%M-%S").replace(tzinfo=UTC_PLUS_8)

    def _load_index_and_fits(self) -> None:
        idx_path = os.path.join(self.cal_dir, "calibration_index.json")
        if not os.path.exists(idx_path):
            raise FileNotFoundError(f"Missing index file: {idx_path}")

        with open(idx_path, "r", encoding="utf-8") as f:
            index = json.load(f)

        calibs = index.get("calibrations", [])
        if not calibs:
            raise RuntimeError("No calibrations listed in the index.")

        self._fits.clear()
        for entry in calibs:
            fname = entry["file"]
            fpath = os.path.join(self.cal_dir, fname)
            with open(fpath, "r", encoding="utf-8") as f:
                blob = json.load(f)

            slot = blob["time_slot"]
            fit = blob["fit"]
            cf = CalibFit(
                start_file=slot["start_file"],
                special_class=slot.get("special_class"),
                img_size=int(fit["img_size"]),
                u0=float(fit["u0"]),
                v0=float(fit["v0"]),
                a=np.array(fit["a_coeffs"], dtype=float),
                az_offset=float(fit["az_offset_rad"]),
            )
            self._fits.append(cf)

        # sort by slot time
        self._fits.sort(key=lambda cf: self._parse_local(cf.start_file))

    # ---------------------- utilities ----------------------
    def _select_fit_for_date(self, date_or_stamp: Union[str, datetime]) -> CalibFit:
        """
        Choose the most recent calibration with start_file <= given date.
        date_or_stamp: "YYYY-MM-DD-HH-MM-SS" (local UTC+8) or datetime tz-aware in UTC+8.
        """
        if isinstance(date_or_stamp, str):
            t = self._parse_local(date_or_stamp)
        elif isinstance(date_or_stamp, datetime):
            if date_or_stamp.tzinfo is None:
                # assume local
                t = date_or_stamp.replace(tzinfo=UTC_PLUS_8)
            else:
                t = date_or_stamp.astimezone(UTC_PLUS_8)
        else:
            raise TypeError("date_or_stamp must be string or datetime.")

        chosen = None
        for cf in self._fits:
            if self._parse_local(cf.start_file) <= t:
                chosen = cf
            else:
                break
        if chosen is None:
            chosen = self._fits[0]
        return chosen

    @staticmethod
    def _eval_r_of_z(z: np.ndarray, a: np.ndarray) -> np.ndarray:
        r = np.zeros_like(z)
        k = 1
        for ai in a:
            r += ai * (z**k)
            k += 2
        return r

    @staticmethod
    def _invert_r_to_z(r: np.ndarray, a: np.ndarray,
                       newton_max_iters: int = 30, newton_tol: float = 1e-9) -> np.ndarray:
        # Newton solve for z
        r = np.asarray(r, dtype=float)
        a1 = a[0] if a.size >= 1 else 1.0
        z = np.clip(r / max(a1, 1e-9), 0.0, np.pi/2)
        for _ in range(newton_max_iters):
            rz = AllSkyMapper._eval_r_of_z(z, a)
            f = rz - r
            k = 1
            drdz = np.zeros_like(z)
            for ai in a:
                drdz += ai * k * (z**(k-1))
                k += 2
            drdz = np.where(np.abs(drdz) < 1e-12, 1e-12, drdz)
            z_new = np.clip(z - f/drdz, 0.0, np.pi/2)
            if np.nanmax(np.abs(z_new - z)) < newton_tol:
                z = z_new
                break
            z = z_new
        return z

    @staticmethod
    def _uv2rphi(u: np.ndarray, v: np.ndarray, u0: float, v0: float) -> Tuple[np.ndarray, np.ndarray]:
        du = u - u0
        dv = v - v0
        r = np.sqrt(du**2 + dv**2)
        phi = np.arctan2(dv, du) % (2*np.pi)
        return r, phi

    @staticmethod
    def _rphi2uv(r: np.ndarray, phi: np.ndarray, u0: float, v0: float) -> Tuple[np.ndarray, np.ndarray]:
        u = r*np.cos(phi) + u0
        v = r*np.sin(phi) + v0
        return u, v

    @staticmethod
    def _resolve_resolution(resolution: Union[int, Tuple[int, int]]) -> Tuple[int, int, float]:
        """
        Returns (W, H, scale) where scale = isotropic scale vs BASE_IMG_SIZE.
        For tuple (W,H) with W!=H, uses scale = min(W,H)/BASE_IMG_SIZE.
        """
        if isinstance(resolution, int):
            W = H = resolution
        elif isinstance(resolution, (tuple, list)) and len(resolution) == 2:
            W, H = int(resolution[0]), int(resolution[1])
        else:
            raise TypeError("resolution must be int or (W,H).")
        s = min(W, H) / float(BASE_IMG_SIZE)
        return W, H, s

    # ---------------------- public mapping ----------------------
    def altaz_to_pixel(self,
                       date_or_stamp: Union[str, datetime],
                       alt_deg: Union[float, List[float], np.ndarray],
                       az_deg: Union[float, List[float], np.ndarray],
                       resolution: Union[int, Tuple[int, int]] = BASE_IMG_SIZE
                       ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Maps alt/az (deg) to pixel (u,v) at the requested output resolution.
        Accepts scalar or list/array. Returns NumPy arrays (u, v).
        """
        cf = self._select_fit_for_date(date_or_stamp)
        W, H, s = self._resolve_resolution(resolution)

        alt = np.asarray(alt_deg, dtype=float)
        az  = np.asarray(az_deg, dtype=float)
        # promote shapes
        alt, az = np.broadcast_arrays(alt, az)

        # scaled center and coefficients
        u0 = cf.u0 * s
        v0 = cf.v0 * s
        a_scaled = cf.a * s   # since r scales linearly with resize

        # z = pi/2 - alt
        z = np.radians(90.0 - alt)
        # r via polynomial at scaled size
        r = self._eval_r_of_z(z, a_scaled)

        # relationship: az = img_angle_from_y + az_offset
        # img_angle_from_y = (az - az_offset) mod 2π
        img_angle_from_y = (np.radians(az) - cf.az_offset) % (2*np.pi)
        # phi = π/2 - img_angle_from_y (angle from +x)
        phi = (np.pi/2 - img_angle_from_y) % (2*np.pi)

        u, v = self._rphi2uv(r, phi, u0, v0)
        return u, v

    def pixel_to_altaz(self,
                       date_or_stamp: Union[str, datetime],
                       u: Union[float, List[float], np.ndarray],
                       v: Union[float, List[float], np.ndarray],
                       resolution: Union[int, Tuple[int, int]] = BASE_IMG_SIZE
                       ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Maps pixel (u,v) to (alt, az) in degrees at requested output resolution.
        Accepts scalar or list/array. Returns NumPy arrays (alt_deg, az_deg).
        """
        cf = self._select_fit_for_date(date_or_stamp)
        W, H, s = self._resolve_resolution(resolution)

        u = np.asarray(u, dtype=float)
        v = np.asarray(v, dtype=float)
        u, v = np.broadcast_arrays(u, v)

        u0 = cf.u0 * s
        v0 = cf.v0 * s
        a_scaled = cf.a * s

        r, phi = self._uv2rphi(u, v, u0, v0)
        z = self._invert_r_to_z(r, a_scaled)
        alt = 90.0 - np.degrees(z)

        img_angle_from_y = (np.pi/2 - (phi % (2*np.pi))) % (2*np.pi)
        az = (img_angle_from_y + cf.az_offset) % (2*np.pi)
        az_deg = np.degrees(az)

        # wrap az to [0, 360)
        az_deg = az_deg % 360.0
        return alt, az_deg

# ---------------------- quick example (commented) ----------------------
if __name__ == "__main__":
    mapper = AllSkyMapper("/mnt/hgfs/cloud/data/calibration/out/calibrations")
    # Example A: alt/az -> pixel
    u, v = mapper.altaz_to_pixel("2024-04-08-01-30-30", [45, 30], [0, 180], resolution=4096)
    print(u, v)
    # Example B: pixel -> alt/az
    alt, az = mapper.pixel_to_altaz("2024-04-08-01-30-10", [2048, 1000], [2048, 1500], resolution=4096)
    print(alt, az)
