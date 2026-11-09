#!/usr/bin/env python3
"""Ceilometer truth labels at the AERI feature times, per site, plus one combined NetCDF.

For each site, the feature rows (from aeri_features.py) are labelled from that site's ceilometer:
    Clouds = 1 (any cloud among the bin's samples), 0 (clear), NaN (unknown)
    CBH    = cloud base height (m): the mean lowest cloud base over the bin's cloudy samples.
             Only defined where Clouds = 1; NaN for clear and for unknown rows.
    CloudFraction = fraction of the bin's ceilometer samples that saw a cloud (0-1). NaN where Clouds is NaN, so
             Clouds = 1 exactly when CloudFraction > 0. Not available from the legacy Ceilometer.csv (NaN).

Outputs (out_dir)
    Ceilometer Truth <site>.csv               one column `Clouds` (plus any TRUTH_CSV_EXTRA_COLUMNS), one row per row
                                              of `Feature Table <site>.csv`, SAME ORDER (the MATLAB-style truth file)
    Ceilometer Truth <site> (with times).csv  Time, Clouds, CBH, CloudFraction, QCCounts and the reason for each NaN
    Truth_and_Features.nc                     both sites together, dims (site, time, feature):
        features(site, time, feature)    the feature tables; NaN where the site has no feature row at that time
        truth(site, time)                1 / 0 / NaN at that site's feature times
        cbh(site, time)                  cloud base height (m) where truth = 1, else NaN
        cloud_fraction(site, time)       fraction of ceilometer samples with a cloud (NaN where truth is NaN)
        truth_n_samples(site, time)      ceilometer samples behind each label
        has_features(site, time)         True where the site has a feature row
        coords: site, time, feature, ceilometer(site) = which ceilometer labelled the site

NaN truth means "no usable ceilometer data for that feature row". Drop those rows (from the features and
the truth together) before training or scoring.

Requires: numpy, pandas, xarray, netCDF4.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

# ----------------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------------
# Used when this file is run on its own. run_pipeline.py passes its own `sites` to main().
#   kind: "ceilometer" = Ceilometer_<site>.nc  (ceilometer_processing.py)
#         "uqam"       = UQAM_Clouds_<site>.nc (uqam_ceilometer_processing.py)
#         "csv"        = legacy Ceilometer.csv (columns Clouds, QCCounts)
SITES = {
    "Gault": {"features": "Features_Gault.nc", "ceilometer": "Ceilometer_Gault.nc", "kind": "ceilometer"},
    "Burnside": {"features": "Features_Burnside.nc", "ceilometer": "UQAM_Clouds_Burnside.nc", "kind": "uqam"},
}
OUT_NC = "Truth_and_Features.nc"
MIN_QC_SAMPLES = 1      # a bin needs at least this many ceilometer samples to be labelled
TRUTH_CSV_EXTRA_COLUMNS = ()   # e.g. ("CBH", "CloudFraction"): extra columns for `Ceilometer Truth <site>.csv`.
                               # Empty by default so code that reads that file as a single column keeps working.
STEP = pd.Timedelta(minutes=8)
KIND_NAMES = {"ceilometer": "Ceilometer (netCDF)", "uqam": "UQAM CS135", "csv": "Ceilometer.csv"}


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------
def load_features(path) -> pd.DataFrame:
    """Features_<site>.nc -> DataFrame indexed by Time, one column per feature."""
    with xr.open_dataset(path) as ds:
        return pd.DataFrame(ds["features"].values,
                            index=pd.DatetimeIndex(ds["time"].values.copy(), name="Time"),
                            columns=[str(f) for f in ds["feature"].values])


def load_ceilometer(path, kind="ceilometer") -> pd.DataFrame:
    """Any of the ceilometer outputs -> DataFrame indexed by Time with `Clouds`, `CBH` and `QCCounts`."""
    if kind == "csv":
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        out = df[["Clouds", "QCCounts"]].copy()
        out["CBH"] = df["height1"].where(df["height1"] >= 0)       # old MATLAB CSVs hold -1 for "no cloud"
        out["CloudFraction"] = np.nan                              # the legacy CSV has no cloud fraction
        return out
    cbh_var, counts = {"ceilometer": ("height1", "n_samples_qc"), "uqam": ("cbh", "n_samples")}[kind]
    with xr.open_dataset(path) as ds:
        return pd.DataFrame({"Clouds": ds["cloud"].values.astype(float),
                             "CBH": ds[cbh_var].values.astype(float),
                             "CloudFraction": ds["cloud_fraction"].values.astype(float),
                             "QCCounts": ds[counts].values.astype(float)},
                            index=pd.DatetimeIndex(ds["time"].values.copy(), name="Time"))


def check_lattice(times, name=""):
    """All times must sit on the midnight-anchored 8-minute lattice that every step now uses."""
    t = pd.DatetimeIndex(times).values
    off = (t - t.astype("datetime64[D]")) % np.timedelta64(STEP)
    if (off != np.timedelta64(0, "ns")).any():
        raise ValueError(f"{name}: {(off != np.timedelta64(0, 'ns')).sum()} times are not on the midnight-anchored "
                         "8-minute lattice. Re-run the AERI step (it now anchors its bins at midnight).")


# ----------------------------------------------------------------------------
# Labelling
# ----------------------------------------------------------------------------
def sync(ceil: pd.DataFrame, feature_times: pd.DatetimeIndex, min_qc_samples=MIN_QC_SAMPLES) -> pd.DataFrame:
    """Label each feature time: Clouds = 1 (any cloud), 0 (clear), or NaN (unknown), with the cloud base height."""
    # A feature time off the ceilometer's 8-minute lattice means the two grids are out of phase.
    # (The MATLAB version indexed ceil(gaultTime.Time, :), which also errors on any missing time.)
    step = np.timedelta64(STEP)
    phase = (feature_times.values - ceil.index[0].to_datetime64()) % step
    off = phase != np.timedelta64(0, "ns")
    if off.any():
        raise ValueError(
            f"{off.sum()} of {len(feature_times)} feature times are off the ceilometer's 8-minute lattice "
            f"(first: {feature_times[off][0]}, {pd.Timedelta(phase[off][0])} out of phase). "
            "Re-run the ceilometer step, ideally with the AERI grid shared (SHARE_AERI_GRID in run_pipeline.py)."
        )

    # On the lattice but outside the ceilometer's period (e.g. a full-year AERI run and a summer
    # ceilometer record) is simply unknown.
    in_period = feature_times.isin(ceil.index)
    sel = ceil.reindex(feature_times)
    n_qc = sel["QCCounts"].fillna(0)
    label = sel["Clouds"].astype(float)
    # The MATLAB version used `height1 ~= -1`, which is true for NaN, so bins with no ceilometer data
    # (gaps, all samples flagged) were labelled cloudy. Those are unknown, not cloudy.
    label[n_qc < min_qc_samples] = np.nan
    reason = np.select(
        [~in_period, n_qc == 0, n_qc < min_qc_samples, label.isna()],
        ["outside ceilometer period", "no QC-passing samples", "too few samples",
         "no label (obscured, or no valid cloud height)"],
        default="",
    )
    # A cloud base only exists where a cloud was seen: NaN for clear and unknown rows.
    cbh = sel["CBH"].astype(float)
    cbh[label != 1] = np.nan
    # Cloud fraction follows the label: unknown rows (no data, or obscured) stay NaN, so it never contradicts it.
    # (For a mostly obscured UQAM bin the raw fraction would only be a lower bound.)
    frac = sel["CloudFraction"].astype(float)
    frac[label.isna()] = np.nan
    return pd.DataFrame({"Clouds": label, "CBH": cbh, "CloudFraction": frac, "QCCounts": n_qc, "note": reason},
                        index=feature_times)


# ----------------------------------------------------------------------------
# Combined dataset
# ----------------------------------------------------------------------------
def build_dataset(results: dict) -> xr.Dataset:
    """results[site] = {"features": DataFrame, "truth": DataFrame from sync(), "source": str}."""
    sites = list(results)
    columns = list(results[sites[0]]["features"].columns)
    for s in sites:
        if list(results[s]["features"].columns) != columns:
            raise ValueError(f"Feature columns differ between sites ({s})")

    idx = results[sites[0]]["features"].index
    for s in sites[1:]:
        idx = idx.union(results[s]["features"].index)

    feats = np.stack([results[s]["features"].reindex(idx).to_numpy(float) for s in sites])
    truth = np.stack([results[s]["truth"]["Clouds"].reindex(idx).to_numpy(float) for s in sites])
    cbh = np.stack([results[s]["truth"]["CBH"].reindex(idx).to_numpy(float) for s in sites])
    frac = np.stack([results[s]["truth"]["CloudFraction"].reindex(idx).to_numpy(float) for s in sites])
    nsamp = np.stack([results[s]["truth"]["QCCounts"].reindex(idx).to_numpy(float) for s in sites])
    has = np.stack([idx.isin(results[s]["features"].index) for s in sites])

    ds = xr.Dataset(
        {
            "features": (("site", "time", "feature"), feats),
            "truth": (("site", "time"), truth),
            "cbh": (("site", "time"), cbh),
            "cloud_fraction": (("site", "time"), frac),
            "truth_n_samples": (("site", "time"), nsamp),
            "has_features": (("site", "time"), has),
        },
        coords={"site": sites, "time": idx.to_numpy(), "feature": columns,
                "ceilometer": ("site", [results[s]["source"] for s in sites])},
    )
    ds["cbh"].attrs.update(units="m", long_name="lowest cloud base height, mean over the bin's cloudy samples "
                                               "(NaN unless truth = 1)")
    ds["cloud_fraction"].attrs.update(units="1", long_name="fraction of the bin's ceilometer samples with a cloud "
                                                           "(NaN where truth is NaN)")
    ds.attrs["description"] = ("AERI cloud-detection features and ceilometer truth labels per site, on 8-minute "
                               "left-edge bins (UTC). truth: 1 cloud, 0 clear, NaN unknown.")
    return ds


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main(sites=None, out_dir=None, min_qc_samples=MIN_QC_SAMPLES) -> xr.Dataset:
    """Run the whole step. `sites` has the same layout as SITES above (used by run_pipeline.py)."""
    sites = SITES if sites is None else sites
    out_dir = Path(".") if out_dir is None else Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for site, cfg in sites.items():
        kind = cfg.get("kind", "ceilometer")
        feats = load_features(cfg["features"])
        check_lattice(feats.index, f"{site} features")
        truth = sync(load_ceilometer(cfg["ceilometer"], kind), feats.index, min_qc_samples)

        #legacy = truth[["Clouds"]].astype("Int64")
        #for col in TRUTH_CSV_EXTRA_COLUMNS:
        #    legacy[col] = truth[col]
        #legacy.to_csv(out_dir / f"Ceilometer Truth {site}.csv", index=False, na_rep="NaN")
        truth.assign(Clouds=truth["Clouds"].astype("Int64")).to_csv(
            out_dir / f"Ceilometer Truth {site} (with times).csv", na_rep="NaN")
        print(f"{site}: {len(truth)} feature rows: {int((truth.Clouds == 1).sum())} cloudy, "
              f"{int((truth.Clouds == 0).sum())} clear, {int(truth.Clouds.isna().sum())} unknown (NaN); "
              f"cloud base height for {int(truth.CBH.notna().sum())} (median {truth.CBH.median():.0f} m); "
              f"mean cloud fraction {truth.CloudFraction.mean():.2f}")
        results[site] = {"features": feats, "truth": truth, "source": KIND_NAMES.get(kind, kind)}

    ds = build_dataset(results)
    ds.to_netcdf(out_dir / OUT_NC)
    return ds


if __name__ == "__main__":
    main()
