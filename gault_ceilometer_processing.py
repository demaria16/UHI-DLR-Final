#!/usr/bin/env python3
"""Ceilometer data binned into 8-minute bins that line up with the AERI bins.
Inputs
    Ceilometer data stored in .nc

Variables (.nc): time, warning alarm, transmission, height1..height4 (cloud base heights)

Outputs
    Ceilometer.nc   xarray Dataset on dim `time` (left-edge 8-minute bins, tz-naive UTC) for joining with AERI
    AND/OR Ceilometer.csv  the same 8 columns as the MATLAB version, for the cloud-QC workflow

Variables (.nc): transmission, height1..height4, n_samples, n_samples_qc, cloud, cloud_fraction

Requires: numpy, pandas, xarray, netCDF4.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from netCDF4 import Dataset

# ----------------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------------
BASE_DIR = Path(r"C:\Users\lmarti96\Documents\MATLAB\ceil_summer_2024")
DIR_SAVE = Path(r"C:\Users\lmarti96\Documents\MATLAB")
STEP = np.timedelta64(8, "m")
# Optional: path to the AERI output (all_8_minute.nc). If given, the ceilometer uses exactly
# the same time grid, which guarantees the bins line up.
AERI_FILE = None
HEIGHT_VARS = ["height1", "height2", "height3", "height4"]


# ----------------------------------------------------------------------------
# Reading
# ----------------------------------------------------------------------------
def _read(nc: Dataset, name: str) -> np.ndarray:
    """1-D float64 array with fill values as NaN (like MATLAB's ncread)."""
    return np.ma.filled(np.ma.asarray(nc.variables[name][:]).astype(float), np.nan).ravel()


def read_times(path: Path) -> np.ndarray:
    """POSIX seconds -> tz-naive datetime64[ns] (UTC); invalid times become NaT."""
    with Dataset(path) as nc:
        units = getattr(nc.variables["time"], "units", "")
        if "1970-01-01" not in units:
            warnings.warn(f"{path.name}: time units are '{units}', not POSIX seconds")
        x = _read(nc, "time")
    return pd.to_datetime(x, unit="s", errors="coerce").to_numpy()


def read_file(path: Path):
    t = read_times(path)
    with Dataset(path) as nc:
        data = {name: _read(nc, name) for name in ["warning_alarm", "transmission"] + HEIGHT_VARS}
    for name, arr in data.items():
        if arr.shape != t.shape:
            raise ValueError(f"{name} has {arr.size} values but time has {t.size}")
    return t, data

# ----------------------------------------------------------------------------
# Main processing
# ----------------------------------------------------------------------------
def make_grid(paths, step=STEP, grid=None):
    """Get the left-edge bin times to create a time vector. Anchored at midnight UTC unless a grid is given.

    1440 minutes is a whole number of 8-minute steps, so midnight-anchored bins fall on the
    same lattice on every day. (Anchoring at the first sample or the first hour does not:
    60 minutes is not a multiple of 8.)
    """
    if grid is not None:
        grid = np.asarray(grid, dtype="datetime64[ns]")
        if not np.all(np.diff(grid) == step):
            raise ValueError("Supplied grid is not regular with the requested step")
        return grid
    tmin, tmax = None, None
    for p in paths:
        t = read_times(p)
        t = t[~np.isnat(t)]
        if t.size:
            tmin = t.min() if tmin is None else min(tmin, t.min())
            tmax = t.max() if tmax is None else max(tmax, t.max())
    # Define the origin of each file as the first time
    origin = tmin.astype("datetime64[D]").astype("datetime64[ns]")
    # Get the range of each file
    nbin = int((tmax - origin) // step) + 1 
    # Return an array of times in whole number steps
    return origin + np.arange(nbin) * step


def process(base_dir=BASE_DIR, step=STEP, grid=None) -> xr.Dataset:
    ''' 
    Process each file entirely.
    Inputs: the directory of all files, the time-step required for each bin, whether or not to use the grid method.
    Output: a .nc file with the transmission inputs.
        Variables: (.nc) coordinates (time (s)), transmission of the ceilometer (time, 1), cloud base heights (time, 4),
        number of samples (time, 1), number of quality controlled samples ("time", 1), presence/absence of clouds (time, 1)
        cloud_fraction (time, 1) (clouds as a fraction of the number of quality controlled samples)
    '''
    
    paths = sorted(Path(base_dir).rglob("*.nc"))
    if not paths:
        raise FileNotFoundError(f"No .nc files found in {base_dir}")

    time_vector = make_grid(paths, step, grid)
    origin, nbin = time_vector[0], time_vector.size

    # Running sums and counts, accumulated across files so bins that straddle files are handled
    z = lambda: np.zeros(nbin)
    sum_trans, n_trans_qc, n_raw = z(), z(), z()
    sum_h = {h: z() for h in HEIGHT_VARS}      # sum of heights over samples where a cloud was detected
    n_cloudy = {h: z() for h in HEIGHT_VARS}   # number of such samples
    n_valid_h1 = z()                           # QC-passing samples with a valid height1 (cloud or clear)
    failed = []

    with Dataset(paths[0]) as nc:
        altitude = float(np.ma.filled(np.ma.asarray(nc.variables["altitude"][:]), np.nan).ravel()[0]) \
            if "altitude" in nc.variables else np.nan

    def acc(total, n, values, mask, b):
        total += np.bincount(b[mask], weights=values[mask], minlength=nbin)
        n += np.bincount(b[mask], minlength=nbin)

    for i, path in enumerate(paths, start=1):
        try:
            t, d = read_file(path)
            ok_t = ~np.isnat(t)
            b = np.full(t.shape, -1, dtype=np.int64)
            b[ok_t] = (t[ok_t] - origin) // step
            in_range = ok_t & (b >= 0) & (b < nbin)

            # Samples with any warning/alarm are dropped entirely (NaN flag also counts as bad)
            qc_ok = d["warning_alarm"] == 0

            trans = d["transmission"]
            m_raw = in_range & np.isfinite(trans)
            n_raw += np.bincount(b[m_raw], minlength=nbin)
            acc(sum_trans, n_trans_qc, trans, m_raw & qc_ok, b)

            for h in HEIGHT_VARS:
                v = d[h]
                # Cloud detected = valid, non-negative height. -1 means "no cloud base" and
                # must not be averaged into the height.
                cloudy = in_range & qc_ok & np.isfinite(v) & (v >= 0)
                acc(sum_h[h], n_cloudy[h], v, cloudy, b)
            valid1 = in_range & qc_ok & np.isfinite(d["height1"])
            n_valid_h1 += np.bincount(b[valid1], minlength=nbin)

            print(f"Loop {i} / {len(paths)}")
        except Exception as exc:
            failed.append(str(path))
            warnings.warn(f"Skipped {path}: {exc}")

    # ------------------------------------------------------------------
    # Finalize
    # ------------------------------------------------------------------
    with np.errstate(divide="ignore", invalid="ignore"):
        data = {"transmission": sum_trans / n_trans_qc}
        for h in HEIGHT_VARS:
            data[h] = sum_h[h] / n_cloudy[h]                  # mean over cloudy samples; NaN if none
        cloud_fraction = n_cloudy["height1"] / n_valid_h1      # NaN where no valid samples
    # 1 if any valid sample in the bin had a cloud, 0 if valid samples but none cloudy, NaN if no valid data
    cloud = np.where(n_valid_h1 > 0, (n_cloudy["height1"] > 0).astype(float), np.nan)

    ds = xr.Dataset(
        {
            **{k: ("time", v) for k, v in data.items()}, # "Data" includes the mean of cloud base heights at each height, NaN if there are no cloudy samples
            "n_samples": ("time", n_raw),
            "n_samples_qc": ("time", n_trans_qc),
            "cloud": ("time", cloud),
            "cloud_fraction": ("time", cloud_fraction),
        },
        coords={"time": time_vector},
    )
    ds.attrs["altitude"] = altitude
    ds.attrs["failed_files"] = "; ".join(failed)
    ds.attrs["description"] = ("8-minute ceilometer bins (left edge, UTC). Heights are means over samples "
                               "with a detected cloud base only; cloud = any cloud among QC-passing samples.")
    return ds


def to_legacy_dataframe(ds: xr.Dataset) -> pd.DataFrame:
    """The 8 columns of the MATLAB Ceilometer.csv, same names and order."""
    df = pd.DataFrame({
        "transmission": ds["transmission"].values,
        **{h: ds[h].values for h in HEIGHT_VARS},
        "Counts": ds["n_samples"].values,
        "QCCounts": ds["n_samples_qc"].values,
        "Clouds": ds["cloud"].values,
    }, index=pd.DatetimeIndex(ds["time"].values, name="Time"))
    return df


def main():
    grid = xr.open_dataset(AERI_FILE)["time"].values if AERI_FILE else None
    ds = process(grid=grid)
    DIR_SAVE.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(DIR_SAVE / "Ceilometer.nc")
    # to_legacy_dataframe(ds).to_csv(DIR_SAVE / "Ceilometer.csv", na_rep="NaN")
    if ds.attrs["failed_files"]:
        warnings.warn("Some files failed; see ds.attrs['failed_files'].")
    return ds


if __name__ == "__main__":
    main()
