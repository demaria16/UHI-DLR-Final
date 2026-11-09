#!/usr/bin/env python3
"""Run the AERI + ceilometer pipeline for both sites, in order.

    1. aeri         aeri_8min_weighted.py          each site -> all_8_minute_<site>.nc
    2. features     aeri_features.py               each site -> Feature Table <site>.csv, Feature Time <site>.csv, Features_<site>.nc
    3. uqam         uqam_ceilometer_processing.py  sites using the UQAM ceilometer -> UQAM_Clouds_<site>.nc
    4. ceilometer   ceilometer_processing.py       sites using the netCDF ceilometer -> Ceilometer_<site>.nc, Ceilometer <site>.csv
    5. sync         sync_ceilometer_truth.py       Ceilometer Truth <site>.csv for each site, and Truth_and_Features.nc (both sites)

All outputs go to OUT_DIR.
Run everything:                      python run_pipeline.py
Run some steps (in pipeline order):  python run_pipeline.py --steps features sync
Run one site:                        python run_pipeline.py --sites Gault

Before anything runs, the pipeline checks that every input it needs exists (or will be made by an earlier
selected step), and stops at the first step that fails. Steps skipped with --steps use the files already in
OUT_DIR, so you can re-run, say, only `sync` after changing a setting.

Settings that belong to one step (QC options, thresholds, column positions, ...) stay at the top of that
step's own file; only paths and the site setup are here.
"""
from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

import aeri_8min_weighted as aeri
import aeri_features as features
import gault_ceilometer_processing as gault_ceil
import sync_ceilometer_truth as sync
import uqam_ceilometer_processing as uqam_ceil

# ----------------------------------------------------------------------------
# Setup (edit these)
# ----------------------------------------------------------------------------
OUT_DIR = Path(r"")        # every output goes here

# One entry per site.
#   aeri_dir         folder of that site's AERI .nc files
#   ceilometer       "ceilometer" = netCDF ceilometer (ceilometer_processing.py)
#                    "uqam"       = UQAM CS135 .txt files (uqam_ceilometer_processing.py)
#   ceilometer_dir   folder of that ceilometer's raw files
# CHECK: Gault -> netCDF ceilometer and Burnside -> UQAM ceilometer is my reading of your setup.
SITES = {
    "Gault": {
        "aeri_dir": Path(""),
        "ceilometer": "gault",
        "ceilometer_dir": Path(""),
    },
    "Burnside": {
        "aeri_dir": None,              
        "ceilometer": "uqam",
        "ceilometer_dir": Path(""),
    },
}

# Put each ceilometer on exactly its site's AERI time grid. Every step is now anchored at midnight UTC, so the
# grids agree either way; sharing also trims the ceilometer outputs to the AERI period. Needs the `aeri` step.
SHARE_AERI_GRID = True


# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
def _require(site: str, key: str) -> Path:
    value = SITES[site].get(key)
    if value is None:
        raise ValueError(f"SITES['{site}']['{key}'] is not set in run_pipeline.py")
    return Path(value)


def aeri_file(site):
    return OUT_DIR / f"all_8_minute_{site}.nc"


def features_file(site):
    return OUT_DIR / f"Features_{site}.nc"


def ceilometer_file(site):
    return OUT_DIR / (f"UQAM_Clouds_{site}.nc" if SITES[site]["ceilometer"] == "uqam" else f"Ceilometer_{site}.nc")


def _sites_using(kind):
    return [s for s, c in SITES.items() if c["ceilometer"] == kind]


def _grid_file(site):
    return aeri_file(site) if SHARE_AERI_GRID else None


# ----------------------------------------------------------------------------
# Steps
# ----------------------------------------------------------------------------
def step_aeri():
    for site in SITES:
        print(f"[{site}]", flush=True)
        aeri.main(base_dir=_require(site, "aeri_dir"), dir_save=OUT_DIR, tag=site)


def step_features():
    for site in SITES:
        print(f"[{site}]", flush=True)
        features.main(aeri_file=aeri_file(site), dir_save=OUT_DIR, tag=site)


def step_uqam():
    for site in _sites_using("uqam"):
        print(f"[{site}]", flush=True)
        uqam_ceil.main(uqam_dir=_require(site, "ceilometer_dir"), aeri_file=_grid_file(site), dir_save=OUT_DIR, tag=site)


def step_ceilometer():
    for site in _sites_using("ceilometer"):
        print(f"[{site}]", flush=True)
        gault_ceil.main(base_dir=_require(site, "ceilometer_dir"), aeri_file=_grid_file(site),
                        dir_save=OUT_DIR, tag=site)


def step_sync():
    sync.main(
        sites={s: {"features": features_file(s), "ceilometer": ceilometer_file(s), "kind": SITES[s]["ceilometer"]}
               for s in SITES},
        out_dir=OUT_DIR,
    )


# Each step: (function, inputs it needs from outside, files it makes, hint if an input is missing).
# These are functions so they see the current SITES / OUT_DIR when they are called.
def _spec(name):
    if name == "aeri":
        return step_aeri, [_require(s, "aeri_dir") for s in SITES], [aeri_file(s) for s in SITES], ""
    if name == "features":
        return step_features, [aeri_file(s) for s in SITES], [features_file(s) for s in SITES], "run the `aeri` step first"
    if name == "uqam":
        ss = _sites_using("uqam")
        needs = [_require(s, "ceilometer_dir") for s in ss] + ([aeri_file(s) for s in ss] if SHARE_AERI_GRID else [])
        return step_uqam, needs, [ceilometer_file(s) for s in ss], "run the `aeri` step first, or set SHARE_AERI_GRID = False"
    if name == "ceilometer":
        ss = _sites_using("ceilometer")
        needs = [_require(s, "ceilometer_dir") for s in ss] + ([aeri_file(s) for s in ss] if SHARE_AERI_GRID else [])
        return step_ceilometer, needs, [ceilometer_file(s) for s in ss], "run the `aeri` step first, or set SHARE_AERI_GRID = False"
    if name == "sync":
        needs = [features_file(s) for s in SITES] + [ceilometer_file(s) for s in SITES]
        return step_sync, needs, [OUT_DIR / sync.OUT_NC], "run the `features`, `uqam` and `ceilometer` steps first"
    raise KeyError(name)


ORDER = ["aeri", "features", "uqam", "ceilometer", "sync"]


def run(selected=None) -> dict:
    """Run the selected steps in pipeline order; returns {step: seconds}. Raises on the first failure."""
    selected = ORDER if not selected else [s for s in ORDER if s in selected]

    # Every input must exist already or be made by an earlier selected step
    produced, problems = set(), []
    for name in selected:
        _, needs, makes, hint = _spec(name)
        for p in needs:
            if Path(p) not in produced and not Path(p).exists():
                problems.append(f"step '{name}' needs {p}" + (f" ({hint})" if hint else ""))
        produced |= {Path(p) for p in makes}
    if problems:
        raise FileNotFoundError("Cannot start:\n  " + "\n  ".join(problems))

    timings = {}
    for name in selected:
        func = _spec(name)[0]
        print(f"\n=== [{ORDER.index(name) + 1}/{len(ORDER)}] {name} " + "=" * 40, flush=True)
        t0 = time.time()
        func()
        timings[name] = time.time() - t0
        print(f"--- {name} finished in {timings[name]:.1f} s", flush=True)
    return timings


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run the AERI processing pipeline.")
    parser.add_argument("--steps", nargs="+", choices=ORDER, metavar="STEP",
                        help=f"steps to run, any of: {', '.join(ORDER)} (default: all)")
    parser.add_argument("--sites", nargs="+", choices=list(SITES), metavar="SITE",
                        help=f"sites to process, any of: {', '.join(SITES)} (default: all)")
    args = parser.parse_args(argv)
    if args.sites:
        for s in [s for s in SITES if s not in args.sites]:
            del SITES[s]
    try:
        timings = run(args.steps)
    except Exception:
        print("\nPipeline stopped:\n", file=sys.stderr)
        traceback.print_exc()
        return 1
    print("\nDone: " + ", ".join(f"{k} {v:.0f}s" for k, v in timings.items()) + f"  (outputs in {OUT_DIR})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
