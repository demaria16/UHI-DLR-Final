#!/usr/bin/env python3
"""Cloud-detection features from the 8-minute AERI radiances.

F1-F12 follow Liu et al. (2022); B1-B15 follow Riot-Brêtcher (2025).

Reads    all_8_minute.nc                   from aeri_8min_weighted.py (variable `radiance`)
Writes   Feature Table.csv                 F1..F12, B1..B15 (no time column), rows with any missing feature dropped
         Feature Time.csv                  header `Time`; one row per row of Feature Table.csv (UTC, ISO format)
         Features.nc                       the same features with a `time` coordinate
         (with main(tag="Gault") the names become "Feature Table Gault.csv", "Features_Gault.nc", ...)
         (optional) Radiances.csv etc.     CSV copies of the AERI products, as the MATLAB script wrote

Row i of Feature Table.csv corresponds to row i of Feature Time.csv. Keep them together.

Requires: numpy, pandas, xarray, netCDF4 (plus aeri_config.py alongside).
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from aeri_config import names_and_aesthetics

# ----------------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------------
AERI_FILE = Path("all_8_minute.nc")
DIR_SAVE = Path(".")
WRITE_AERI_CSVS = False     # also write Radiances.csv, QCCount.csv, Radiance Error.csv (large)

# ----------------------------------------------------------------------------
# Feature definitions (cm^-1)
# ----------------------------------------------------------------------------
# Liu et al. 2022
LIU_F1F2 = (740, 760)                       # linear fit: F1 = intercept, F2 = slope
LIU_F3F4 = [(780, 783), (786, 790), (815, 820), (830, 835),
            (842, 846), (857, 864), (895, 900), (915, 920)]   # one fit over all eight windows
LIU_F5F6 = (1000, 1040)                     # F5 = intercept, F6 = slope
LIU_F7 = (1050, 1070)                       # F7 = intercept only
LIU_F8_BAND, LIU_F8_POINT = (781.5, 782.5), 784.5      # F8 = radiance(point) / mean(band)
LIU_F9_BAND, LIU_F9_POINT = (789.2, 790.2), 791.5      # F9 = radiance(point) / mean(band)
# F10-F12 = radiance(first) / radiance(second), in the order listed.
# CHECK THIS ORDER: the MATLAB code took the two channels in ascending wavenumber order, which
# is the reverse of these lists (and also matched ~2 channels per wavenumber, see notes).
LIU_F10 = (1174, 1170)
LIU_F11 = (1187, 1185)
LIU_F12 = (1198, 1195)

# Riot-Bretcher 2025: eight narrow bands (lo, hi)
RB_BANDS = [(558.5, 562), (675, 680), (700, 705), (809, 812.9),
            (828.3, 834.6), (898.2, 905.4), (985, 998), (1092.1, 1098.8)]
ROLL_WINDOW = 9             # centred window: the sample plus 4 either side (+/- 32 minutes)

# Only these wavenumbers are needed, so the rest are never loaded
WN_MIN, WN_MAX = 500.0, 1250.0


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def band_idx(wn, lo, hi):
    return np.flatnonzero((wn >= lo) & (wn <= hi))


def nearest_idx(wn, target):
    """Index of the single channel closest to `target` (warns if it is further than one channel spacing)."""
    i = int(np.argmin(np.abs(wn - target)))
    if abs(wn[i] - target) > np.median(np.diff(wn)):
        warnings.warn(f"Nearest channel to {target} is {wn[i]:.2f}")
    return i


def linear_fit(x, Y):
    """Row-wise least squares y = a + b*x, ignoring NaNs (as fitlm does).

    x : (n,) wavenumbers; Y : (T, n) radiances. Returns intercept (T,), slope (T,).
    The intercept is the value extrapolated to x = 0, as in fitlm's Estimate(1). Rows with
    fewer than 2 valid points give NaN.
    """
    x = np.asarray(x, float)[None, :]
    m = np.isfinite(Y)
    n = m.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        xbar = np.where(m, x, 0.0).sum(axis=1) / n
        ybar = np.where(m, Y, 0.0).sum(axis=1) / n
        dx = np.where(m, x - xbar[:, None], 0.0)
        dy = np.where(m, Y - ybar[:, None], 0.0)
        slope = (dx * dy).sum(axis=1) / (dx**2).sum(axis=1)
        intercept = ybar - slope * xbar
    bad = n < 2
    slope[bad] = np.nan
    intercept[bad] = np.nan
    return intercept, slope


# ----------------------------------------------------------------------------
# Liu et al. 2022: F1-F12
# ----------------------------------------------------------------------------
def liu_features(r, wn) -> np.ndarray:
    T = r.shape[0]
    F = np.full((T, 12), np.nan)

    i = band_idx(wn, *LIU_F1F2)
    F[:, 0], F[:, 1] = linear_fit(wn[i], r[:, i])

    i = np.concatenate([band_idx(wn, lo, hi) for lo, hi in LIU_F3F4])
    F[:, 2], F[:, 3] = linear_fit(wn[i], r[:, i])

    i = band_idx(wn, *LIU_F5F6)
    F[:, 4], F[:, 5] = linear_fit(wn[i], r[:, i])

    i = band_idx(wn, *LIU_F7)
    F[:, 6], _ = linear_fit(wn[i], r[:, i])

    with np.errstate(invalid="ignore", divide="ignore"):
        for col, (band, point) in zip((7, 8), [(LIU_F8_BAND, LIU_F8_POINT), (LIU_F9_BAND, LIU_F9_POINT)]):
            b = band_idx(wn, *band)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)     # all-NaN rows
                band_mean = np.nanmean(r[:, b], axis=1)
            F[:, col] = r[:, nearest_idx(wn, point)] / band_mean
        for col, (num, den) in zip((9, 10, 11), [LIU_F10, LIU_F11, LIU_F12]):
            F[:, col] = r[:, nearest_idx(wn, num)] / r[:, nearest_idx(wn, den)]
    return F


# ----------------------------------------------------------------------------
# Riot-Bretcher 2025: B1-B15
# ----------------------------------------------------------------------------
def riot_features(r, wn) -> np.ndarray:
    T = r.shape[0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)             # all-NaN rows
        band = np.column_stack([np.nanmean(r[:, band_idx(wn, lo, hi)], axis=1) for lo, hi in RB_BANDS])

    # Centred rolling statistics over consecutive 8-minute bins; NaNs skipped, window truncated at the edges.
    # (MATLAB padded with NaNs and skipped them, which is the same thing.)
    roll = pd.DataFrame(band).rolling(ROLL_WINDOW, center=True, min_periods=1)
    mean8 = roll.mean().to_numpy()
    std = roll.std().to_numpy()                  # N-1; NaN (not 0) when only one valid value

    B = np.full((T, 15), np.nan)
    B[:, 0:8] = mean8                            # B1-B8   rolling mean of each band
    B[:, 8] = band[:, 0]                         # B9      band 1, this bin only
    B[:, 9] = band[:, 4]                         # B10     band 5, this bin only
    B[:, 10] = std[:, 0]                         # B11     rolling std, band 1
    B[:, 11] = std[:, 4]                         # B12     rolling std, band 5
    with np.errstate(invalid="ignore", divide="ignore"):
        B[:, 12] = mean8[:, 0] / mean8[:, 4]     # B13     B1 / B5
        B[:, 13] = mean8[:, 5] / mean8[:, 4]     # B14     B6 / B5
        B[:, 14] = mean8[:, 7] / mean8[:, 4]     # B15     B8 / B5
    return B


def compute_features(r, wn, times) -> pd.DataFrame:
    titles = names_and_aesthetics(apply_plot_defaults=False).feature_titles
    F = np.column_stack([liu_features(r, wn), riot_features(r, wn)])
    return pd.DataFrame(F, index=pd.DatetimeIndex(times, name="Time"), columns=list(titles))


def drop_missing(features: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with any missing feature. Infinities (ratios with a zero denominator) count as
    missing; MATLAB's rmmissing only removed NaN."""
    return features[np.isfinite(features.to_numpy()).all(axis=1)]


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main(aeri_file=None, dir_save=None, tag=None):
    """Run the whole step. Arguments override AERI_FILE / DIR_SAVE (used by run_pipeline.py).

    tag (e.g. a site name) is added to the output names: Feature Table <tag>.csv, Feature Time <tag>.csv,
    Features_<tag>.nc
    """
    aeri_file = AERI_FILE if aeri_file is None else Path(aeri_file)
    dir_save = DIR_SAVE if dir_save is None else Path(dir_save)
    dir_save.mkdir(parents=True, exist_ok=True)
    usfx, sfx = (f"_{tag}", f" {tag}") if tag else ("", "")

    # The file is closed as soon as the data are read, so a later step can overwrite it (Windows locks open files)
    with xr.open_dataset(aeri_file) as ds:
        wn_all = ds["wavenumber"].values
        sel = np.flatnonzero((wn_all >= WN_MIN) & (wn_all <= WN_MAX))
        r = ds["radiance"].isel(wavenumber=sel).values
        wn = wn_all[sel]
        times = ds["time"].values

        if WRITE_AERI_CSVS:
            t = times
            pd.DataFrame(ds["radiance"].values, index=t, columns=wn_all).to_csv(dir_save / f"Radiances{sfx}.csv")
            ds["n_spectra_qc"].to_dataframe().to_csv(dir_save / f"QCCount{sfx}.csv")
            total = np.sqrt(ds["noise_uncertainty"].values**2 + ds["systematic_uncertainty"].values**2)
            pd.DataFrame(total, index=t, columns=wn_all).to_csv(dir_save / f"Radiance Error{sfx}.csv")

    features = compute_features(r, wn, times)
    kept = drop_missing(features)
    print(f"{len(features)} bins, {len(kept)} kept, {len(features) - len(kept)} removed")

    kept.to_csv(dir_save / f"Feature Table{sfx}.csv", index=False)
    kept.index.to_series(name="Time").to_csv(dir_save / f"Feature Time{sfx}.csv", index=False)
    xr.Dataset({"features": (("time", "feature"), kept.to_numpy())},
               coords={"time": kept.index.to_numpy(), "feature": list(kept.columns)}
               ).to_netcdf(dir_save / f"Features{usfx}.nc")
    return kept


if __name__ == "__main__":
    main()
