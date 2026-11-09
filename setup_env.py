#!/usr/bin/env python3
"""Create a virtual environment with everything the AERI / ceilometer pipeline needs.

Works the same on Windows, macOS and Linux, from any terminal, using only the standard library:

    python setup_env.py                 # creates .venv next to this file and installs requirements.txt
    python setup_env.py --venv myenv    # a different folder name or path
    python setup_env.py --recreate      # delete and rebuild an existing environment

Then activate it and run the pipeline (the script prints the exact command for your system):

    macOS / Linux :  source .venv/bin/activate
    Windows (cmd) :  .venv\\Scripts\\activate.bat
    PowerShell    :  .venv\\Scripts\\Activate.ps1
    then          :  python run_pipeline.py

Use `python3` instead of `python` if that is what your system calls it.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import venv
from pathlib import Path

HERE = Path(__file__).resolve().parent
MIN_PYTHON = (3, 9)
PACKAGES = ["numpy", "pandas", "scipy", "xarray", "netCDF4", "matplotlib"]


def venv_python(env_dir: Path) -> Path:
    return env_dir / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def run(cmd, **kwargs):
    print("  $ " + " ".join(str(c) for c in cmd), flush=True)
    subprocess.check_call([str(c) for c in cmd], **kwargs)


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a virtual environment for the AERI / ceilometer pipeline.")
    parser.add_argument("--venv", default=str(HERE / ".venv"), help="environment folder (default: .venv next to this file)")
    parser.add_argument("--requirements", default=str(HERE / "requirements.txt"), help="requirements file to install")
    parser.add_argument("--recreate", action="store_true", help="delete and rebuild the environment if it exists")
    args = parser.parse_args()

    if sys.version_info < MIN_PYTHON:
        print(f"Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]} or newer is required (this is {sys.version.split()[0]}).")
        return 1
    requirements = Path(args.requirements)
    if not requirements.exists():
        print(f"Cannot find {requirements}. Keep requirements.txt next to setup_env.py.")
        return 1

    env_dir = Path(args.venv).resolve()
    if env_dir.exists() and args.recreate:
        print(f"Removing {env_dir}")
        shutil.rmtree(env_dir)

    print(f"\n[1/4] Creating virtual environment in {env_dir}")
    try:
        venv.EnvBuilder(with_pip=True, upgrade_deps=False).create(env_dir)
    except Exception as exc:
        print(f"Could not create the environment: {exc}")
        if sys.platform.startswith("linux"):
            print("On Debian/Ubuntu you may need:  sudo apt install python3-venv")
        return 1
    py = venv_python(env_dir)

    print("\n[2/4] Upgrading pip")
    run([py, "-m", "pip", "install", "--upgrade", "pip"])

    print(f"\n[3/4] Installing {requirements.name}")
    try:
        run([py, "-m", "pip", "install", "-r", requirements])
    except subprocess.CalledProcessError:
        print("\nInstallation failed. Check your internet connection (or proxy settings) and try again.")
        return 1

    print("\n[4/4] Checking the installation")
    check = ("import importlib; "
             + "; ".join(f"m = importlib.import_module('{p}'); print('  {p:<11}', m.__version__)" for p in PACKAGES))
    try:
        run([py, "-c", check])
        # Importing run_pipeline imports every script in the project, so this catches a missing file or package.
        run([py, "run_pipeline.py", "--help"], cwd=HERE, env={**os.environ, "MPLBACKEND": "Agg"},
            stdout=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        print("\nThe packages installed, but the check failed. Make sure all the project's .py files "
              "are in the same folder as setup_env.py.")
        return 1

    if os.name == "nt":
        activate = f"{env_dir}\\Scripts\\activate.bat   (cmd)   or   {env_dir}\\Scripts\\Activate.ps1   (PowerShell)"
    else:
        activate = f"source {env_dir}/bin/activate"
    print(f"""
Done. To use the environment:

    {activate}
    python run_pipeline.py

To leave it again, type:  deactivate
""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
