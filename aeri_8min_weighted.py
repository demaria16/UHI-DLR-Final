#!/usr/bin/env python3
"""AERI 8-minute weighted averages, with noise and systematic uncertainty kept separate.

Each 8-minute bin gets an inverse-variance weighted mean radiance and two
uncertainties that behave differently when averaging:

  noise        random, independent between spectra  -> averages down
  systematic   calibration, shared by every spectrum
               in the bin (assumed fully correlated) -> does NOT average down

Per spectrum i in a bin (one wavenumber), with sigma_n = RADIANCE.SKY_NOISE and
sigma_s = RADIANCE.SKY_ERROR (assumed to be the calibration/systematic part):

    w_i = 1 / (sigma_n,i^2 + sigma_s,i^2)        weights (inverse total variance)
    W   = sum_i w_i
    mean       = sum_i w_i x_i / W
    noise unc  = sqrt( sum_i w_i^2 sigma_n,i^2 ) / W      (independent errors)
    syst. unc  = sum_i w_i sigma_s,i / W                   (fully correlated errors)

Total uncertainty = sqrt(noise^2 + systematic^2); it is not stored, so you can
combine the parts however the later averaging requires (see channel_correlation_demo.py).

Requires: numpy, scipy, netCDF4, xarray (pandas only for optional CSV output).
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import xarray as xr
from netCDF4 import Dataset, chartostring
from scipy import sparse

# ----------------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------------
BASE_DIR = Path("2024")
DIR_SAVE = Path(r"C:\Users\lmarti96\Documents\MATLAB")
BIN_SIZE = np.timedelta64(8, "m")
DO_SCATTER_CHECK = False   # also test the noise against the scatter between spectra (2 more big arrays)
MIN_N_FOR_SCALING = 5      # scatter check only inflates bins with at least this many good spectra
WRITE_CSV = False          # CSVs of the full radiance matrices are huge; the .nc file is usually enough

# DATETIME is days since 2000-01-01 (MJD 51544), matching your MJD_to_DT
DATETIME_EPOCH = np.datetime64("2000-01-01T00:00:00", "ns")


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def mjd_to_datetime(mjd) -> np.ndarray:
    """File DATETIME (days since 2000-01-01 00:00 UTC) -> datetime64[ns].

    Equivalent to MJD_to_DT: x + 51544 as an MJD is the same instant.
    """
    mjd = np.asarray(mjd, dtype=float).ravel()
    return DATETIME_EPOCH + np.round(mjd * 86400e9).astype("timedelta64[ns]")


def _read_float(nc: Dataset, name: str) -> np.ndarray:
    """Read a variable as float64, with fill values turned into NaN (like MATLAB's ncread)."""
    return np.ma.filled(np.ma.asarray(nc.variables[name][:]).astype(float), np.nan)


def _time_first(a: np.ndarray, nt: int) -> np.ndarray:
    """Force a 2-D array to have time along axis 0."""
    a = np.atleast_2d(a)
    return a if a.shape[0] == nt else a.T


def read_times(path: Path) -> np.ndarray:
    with Dataset(path) as nc:
        return mjd_to_datetime(_read_float(nc, "DATETIME"))


def read_aeri(path: Path):
    """Return time (nt,), rad/noise/err (nt, nwn), and bool flags (nt, nflags)."""
    with Dataset(path) as nc:
        time = mjd_to_datetime(_read_float(nc, "DATETIME"))
        nt = time.size
        rad = _time_first(_read_float(nc, "RADIANCE.SKY"), nt)
        noise = _time_first(_read_float(nc, "RADIANCE.SKY_NOISE"), nt)
        err = _time_first(_read_float(nc, "RADIANCE.SKY_ERROR"), nt)
        flags = np.ma.filled(np.ma.asarray(nc.variables["FLAG.MEASUREMENT.QUALITY"][:]), 0)
    flags = _time_first(flags, nt) != 0
    return time, rad, noise, err, flags


def read_flag_names(path: Path):
    try:
        with Dataset(path) as nc:
            v = np.ma.filled(nc.variables["FLAG.NAMES"][:], b"")
        if v.dtype.kind in "SU" and v.ndim == 2:
            return [str(s).strip() for s in chartostring(v)]
        return [str(s).strip() for s in np.atleast_1d(v)]
    except Exception:
        return None


# ----------------------------------------------------------------------------
# Main processing
# ----------------------------------------------------------------------------
def process(base_dir=BASE_DIR, bin_size=BIN_SIZE, do_scatter_check=DO_SCATTER_CHECK,
            min_n_for_scaling=MIN_N_FOR_SCALING) -> xr.Dataset:
    paths = sorted(Path(base_dir).rglob("*.nc"))
    if not paths:
        raise FileNotFoundError(f"No .nc files found in {base_dir}")

    # Time grid (left edge of each bin), anchored at midnight UTC. 1440 minutes is a whole number of
    # 8-minute steps, so every site and instrument lands on the same lattice whatever time its data start.
    # (Anchoring at the first hour does not: 60 minutes is not a multiple of 8.)
    t_first = read_times(paths[0]).min()
    t_last = read_times(paths[-1]).max()
    first = t_first.astype("datetime64[D]").astype("datetime64[ns]")                 # floor to midnight
    last = t_last.astype("datetime64[h]").astype("datetime64[ns]") + np.timedelta64(1, "h")
    nbin = int((last - first) // bin_size) + 1
    time_vector = first + np.arange(nbin) * bin_size

    with Dataset(paths[0]) as nc:
        wavenumbers = _read_float(nc, "WAVENUMBER").ravel()
    flag_names = read_flag_names(paths[0])
    nwn = wavenumbers.size

    # Running sums, accumulated across files so bins that straddle files are handled
    shape = (nbin, nwn)
    sum_w = np.zeros(shape)       # sum w
    sum_wx = np.zeros(shape)      # sum w x
    sum_w2n2 = np.zeros(shape)    # sum w^2 sigma_n^2   (noise variance numerator)
    sum_ws = np.zeros(shape)      # sum w sigma_s       (systematic numerator)
    if do_scatter_check:
        sum_wx2 = np.zeros(shape)   # sum w x^2
        sum_wn2 = np.zeros(shape)   # sum w sigma_n^2
    n_raw = np.zeros(nbin)        # all spectra in the bin
    n_good = np.zeros(nbin)       # spectra passing the quality flags
    flag_totals = None
    failed = []

    for i, path in enumerate(paths, start=1):
        try:
            time, rad, noise, err, flags = read_aeri(path)
            nt = time.size

            # Bin index of each spectrum on the global grid
            bin_idx = (time - first) // bin_size
            in_range = (bin_idx >= 0) & (bin_idx < nbin)

            # Quality flags: count per flag, drop spectra with any flag raised
            if flag_totals is None:
                flag_totals = np.zeros(flags.shape[1], dtype=np.int64)
            flag_totals += flags.sum(axis=0)
            bad = flags.any(axis=1)

            # Weights from the total variance; noise and systematic parts kept for later sums
            with np.errstate(divide="ignore", invalid="ignore"):
                w = 1.0 / (noise**2 + err**2)

            # Unusable values get zero weight (and zero value so 0*NaN can't leak in)
            ok = ((~bad & in_range)[:, None] & np.isfinite(rad) & np.isfinite(w)
                  & np.isfinite(noise) & np.isfinite(err))
            w = np.where(ok, w, 0.0)
            x = np.where(ok, rad, 0.0)
            sn = np.where(ok, noise, 0.0)
            ss = np.where(ok, err, 0.0)

            if not in_range.any():
                continue

            # Sparse indicator: rows = bins present in this file, cols = spectra.
            # S @ M sums the rows of M falling in each bin, all wavenumbers at once.
            cols = np.flatnonzero(in_range)
            ub, g = np.unique(bin_idx[in_range], return_inverse=True)
            S = sparse.csr_matrix((np.ones(cols.size), (g, cols)), shape=(ub.size, nt))

            sum_w[ub] += S @ w
            sum_wx[ub] += S @ (w * x)
            sum_w2n2[ub] += S @ (w**2 * sn**2)
            sum_ws[ub] += S @ (w * ss)
            if do_scatter_check:
                sum_wx2[ub] += S @ (w * x**2)
                sum_wn2[ub] += S @ (w * sn**2)
            n_raw[ub] += S @ np.ones(nt)
            n_good[ub] += S @ (~bad).astype(float)

            print(f"Loop {i} / {len(paths)}")
        except Exception as exc:  # keep going, but record what failed
            failed.append(str(path))
            warnings.warn(f"Skipped {path}: {exc}")

    # ------------------------------------------------------------------
    # Finalize (in place to save memory). Bins with no valid data -> NaN.
    # ------------------------------------------------------------------
    with np.errstate(divide="ignore", invalid="ignore"):
        rad_mean = sum_wx
        np.divide(sum_wx, sum_w, out=rad_mean)

        scale = None
        if do_scatter_check:
            # Observed weighted scatter: sum w (x - mu)^2 = sum w x^2 - mu^2 W
            observed = sum_wx2 - rad_mean**2 * sum_w
            del sum_wx2
            # Expected if the scatter were pure noise: sum w sigma_n^2 - sum w^2 sigma_n^2 / W
            expected = sum_wn2 - sum_w2n2 / sum_w
            del sum_wn2
            ratio = observed / expected                # ~1 if the quoted noise is right
            del observed, expected
            scale = np.sqrt(np.fmax(ratio, 1.0))       # fmax ignores NaN -> no inflation there
            scale[n_good < min_n_for_scaling] = 1.0
            del ratio

        noise_unc = sum_w2n2
        np.sqrt(sum_w2n2, out=noise_unc)
        np.divide(noise_unc, sum_w, out=noise_unc)     # sqrt(sum w^2 sn^2) / W

        syst_unc = sum_ws
        np.divide(sum_ws, sum_w, out=syst_unc)         # sum w ss / W
    del sum_w

    # ------------------------------------------------------------------
    # Assemble output
    # ------------------------------------------------------------------
    ds = xr.Dataset(
        {
            "radiance": (("time", "wavenumber"), rad_mean),
            "noise_uncertainty": (("time", "wavenumber"), noise_unc),
            "systematic_uncertainty": (("time", "wavenumber"), syst_unc),
            "n_spectra": (("time",), n_raw),
            "n_spectra_qc": (("time",), n_good),
        },
        coords={"time": time_vector, "wavenumber": wavenumbers},
    )
    if scale is not None:
        ds["noise_uncertainty_scaled"] = (("time", "wavenumber"), noise_unc * scale)
    if flag_totals is not None:
        ds["flag_totals"] = (("flag",), flag_totals)
        if flag_names is not None and len(flag_names) == flag_totals.size:
            ds = ds.assign_coords(flag=("flag", flag_names))
    ds.attrs["failed_files"] = "; ".join(failed)
    ds.attrs["description"] = ("8-minute inverse-variance weighted mean AERI radiance; "
                               "noise_uncertainty assumes independent errors between spectra, "
                               "systematic_uncertainty assumes fully correlated errors; "
                               "total = sqrt(noise^2 + systematic^2)")
    return ds


def main(base_dir=None, dir_save=None, tag=None):
    """Run the whole step. Arguments override BASE_DIR / DIR_SAVE (used by run_pipeline.py).

    tag (e.g. a site name) is added to the output names: all_8_minute_<tag>.nc
    """
    base_dir = BASE_DIR if base_dir is None else Path(base_dir)
    dir_save = DIR_SAVE if dir_save is None else Path(dir_save)

    usfx, sfx = (f"_{tag}", f" {tag}") if tag else ("", "")
    ds = process(base_dir=base_dir)
    dir_save.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(dir_save / f"all_8_minute{usfx}.nc")

    if WRITE_CSV:
        import pandas as pd
        idx, cols = ds["time"].values, ds["wavenumber"].values
        for name, fname in [("radiance", f"radiances{sfx}.csv"),
                            ("noise_uncertainty", f"noise_uncertainty{sfx}.csv"),
                            ("systematic_uncertainty", f"systematic_uncertainty{sfx}.csv")]:
            pd.DataFrame(ds[name].values, index=idx, columns=cols).to_csv(dir_save / fname)
        ds[["n_spectra", "n_spectra_qc"]].to_dataframe().to_csv(dir_save / f"counts{sfx}.csv")
        pd.Series(cols, name="wavenumber").to_csv(dir_save / f"wavenumbers{sfx}.csv", index=False)

    if ds.attrs["failed_files"]:
        warnings.warn("Some files failed; see ds.attrs['failed_files'].")
    return ds


if __name__ == "__main__":
    main()
