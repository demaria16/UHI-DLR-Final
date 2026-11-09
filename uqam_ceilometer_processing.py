#!/usr/bin/env python3
"""UQAM ceilometer (SkyVUE PRO / CS135 CS-messages) binned into 8-minute bins.

Output (UQAM_Clouds.nc): xarray Dataset on dim `time` (left-edge 8-minute bins, tz-naive UTC,
midnight-anchored so the bins line up with the AERI and the other ceilometer outputs).

Variables
    cbh             mean cloud base height (m) over samples with a detected cloud; NaN if none,
                    or if the bin is mostly obscured (obscured_frac > OBSCURED_THRESHOLD)
    cloud           1 if any sample in the bin had a cloud base, 0 if none did, NaN if no data
                    or the bin is mostly obscured
    cloud_fraction  cloudy samples / samples in the bin (not masked)
    obscured_frac   fraction of samples with status 5 (full obscuration)
    precip_frac     fraction of samples with the precipitation flag set
    n_samples       samples in the bin

Requires: numpy, pandas, xarray, netCDF4 (plus aeri_utils.py alongside).
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from aeri_utils import read_ceilometer_cbh

# ----------------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------------
UQAM_DIR = Path("/Users/lukemartin/Documents/UQAM Ceilometer Data/2024_08/raw")
DIR_SAVE = Path(".")
STEP = np.timedelta64(8, "m")
# A bin is treated as obscured (cbh and cloud -> NaN) if MORE than this fraction of its samples
# are obscured. The MATLAB comment said 1/16 (0.0625) but the code used 0.1; the code's value is kept.
OBSCURED_THRESHOLD = 0.1
# Optional: path to the AERI output (all_8_minute.nc) to use exactly its time grid.
AERI_FILE = None


# ----------------------------------------------------------------------------
# Reading
# ----------------------------------------------------------------------------
def load_raw(paths) -> tuple[pd.DataFrame, list]:
    """Read all files into one DataFrame (tz-naive UTC index): cbh (-1 = no cloud), obscured, precip."""
    parts, failed = [], []
    for p in paths:
        try:
            df = read_ceilometer_cbh(p)
            if len(df):
                parts.append(df)
        except Exception as exc:
            failed.append(str(p))
            warnings.warn(f"Skipped {p}: {exc}")
    if not parts:
        raise RuntimeError("No ceilometer data could be read")
    df = pd.concat(parts)
    df = df[df.index.notna()]                       # messages before the first timestamp have no time
    df.index = df.index.tz_convert(None)            # tz-naive UTC, like the AERI times
    # Identical rows from overlapping files are dropped; distinct rows at one time are kept.
    df = df.reset_index().drop_duplicates().set_index("time").sort_index()
    return df, failed


# ----------------------------------------------------------------------------
# Binning
# ----------------------------------------------------------------------------
def make_grid(tmin, tmax, step=STEP, grid=None):
    """Left-edge bin times, anchored at midnight UTC (1440 min is a whole number of 8-min steps)."""
    if grid is not None:
        grid = np.asarray(grid, dtype="datetime64[ns]")
        if not np.all(np.diff(grid) == step):
            raise ValueError("Supplied grid is not regular with the requested step")
        return grid
    origin = np.datetime64(tmin, "D").astype("datetime64[ns]")
    nbin = int((np.datetime64(tmax, "ns") - origin) // step) + 1
    return origin + np.arange(nbin) * step


def bin_clouds(df: pd.DataFrame, step=STEP, grid=None,
               obscured_threshold=OBSCURED_THRESHOLD) -> xr.Dataset:
    t = df.index.to_numpy("datetime64[ns]")
    time_vector = make_grid(t.min(), t.max(), step, grid)
    origin, nbin = time_vector[0], time_vector.size

    # Samples whose height could not be parsed (NaN) carry no information
    valid = np.isfinite(df["cbh"].to_numpy(float))
    cbh = df["cbh"].to_numpy(float)
    b = (t - origin) // step
    in_range = valid & (b >= 0) & (b < nbin)
    b = b[in_range]
    cbh = cbh[in_range]
    obscured = df["obscured"].to_numpy(bool)[in_range]
    precip = df["precip"].to_numpy(bool)[in_range]

    cloudy = cbh >= 0                                # -1 means "no cloud base"; it is never averaged
    n = np.bincount(b, minlength=nbin).astype(float)
    n_cloudy = np.bincount(b[cloudy], minlength=nbin).astype(float)
    sum_cbh = np.bincount(b[cloudy], weights=cbh[cloudy], minlength=nbin)
    n_obs = np.bincount(b[obscured], minlength=nbin).astype(float)
    n_prec = np.bincount(b[precip], minlength=nbin).astype(float)

    with np.errstate(divide="ignore", invalid="ignore"):
        cbh_mean = sum_cbh / n_cloudy                # NaN where no cloudy samples
        cloud_fraction = n_cloudy / n
        obscured_frac = n_obs / n
        precip_frac = n_prec / n
    mostly_obscured = obscured_frac > obscured_threshold

    cbh_out = np.where(mostly_obscured, np.nan, cbh_mean)
    cloud = np.where((n > 0) & ~mostly_obscured, (n_cloudy > 0).astype(float), np.nan)

    return xr.Dataset(
        {
            "cbh": ("time", cbh_out),
            "cloud": ("time", cloud),
            "cloud_fraction": ("time", cloud_fraction),
            "obscured_frac": ("time", obscured_frac),
            "precip_frac": ("time", precip_frac),
            "n_samples": ("time", n),
        },
        coords={"time": time_vector},
    )


def main(uqam_dir=None, aeri_file=None, dir_save=None, tag=None):
    """Run the whole step. Arguments override UQAM_DIR / AERI_FILE / DIR_SAVE (used by run_pipeline.py).

    tag (e.g. a site name) is added to the output name: UQAM_Clouds_<tag>.nc
    """
    uqam_dir = UQAM_DIR if uqam_dir is None else Path(uqam_dir)
    aeri_file = AERI_FILE if aeri_file is None else aeri_file
    dir_save = DIR_SAVE if dir_save is None else Path(dir_save)

    paths = sorted(uqam_dir.rglob("*.txt"))
    if not paths:
        raise FileNotFoundError(f"No .txt files found in {uqam_dir}")
    df, failed = load_raw(paths)
    grid = None
    if aeri_file:
        with xr.open_dataset(aeri_file) as a:
            grid = a["time"].values.copy()
    ds = bin_clouds(df, grid=grid)
    ds.attrs["failed_files"] = "; ".join(failed)
    ds.attrs["description"] = ("8-minute UQAM ceilometer bins (left edge, UTC). cbh = mean over samples with "
                               "a detected cloud base; bins with obscured fraction above "
                               f"{OBSCURED_THRESHOLD} are NaN in cbh and cloud.")
    dir_save.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(dir_save / (f"UQAM_Clouds_{tag}.nc" if tag else "UQAM_Clouds.nc"))
    return ds


if __name__ == "__main__":
    main()
