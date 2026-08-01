#!/bin/bash
module load python/3.10 gcc hdf5
source $HOME/PYTHON/GUAC/jupyter_env/bin/activate

pip install --quiet cmasher 2>/dev/null || true

python -u disk_analysis/plot_chemical_timescales.py 2>&1 | tee chemical_timescales.log
