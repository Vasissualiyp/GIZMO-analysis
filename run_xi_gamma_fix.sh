#!/bin/bash
# Recompute frames18/mass_evolution.npz with the density-peak center fix, then
# regenerate the paper figures (incl. xi_gamma_combined.pdf).
export LD_LIBRARY_PATH="/nix/store/ab3753m6i7isgvzphlar0a8xb84gl96i-gcc-15.2.0-lib/lib:/nix/store/2kdz3m7ic8w226pcvkz1dlg169v91p6a-zlib-1.3.2/lib:${LD_LIBRARY_PATH:-}"

if [ -x /tmp/sci_venv/bin/python ]; then
    PYTHON=/tmp/sci_venv/bin/python
else
    echo "ERROR: no /tmp/sci_venv" >&2
    exit 1
fi

cd "$(dirname "$0")"

echo "=== Step 1: recompute frames18/mass_evolution.npz ==="
$PYTHON -u disk_analysis/compute_mass_evolution.py --overwrite 2>&1 | tee mass_evolution_recompute.log
echo "Step 1 exit: ${PIPESTATUS[0]}"

echo "=== Step 2: regenerate paper figures ==="
$PYTHON -u generate_paper_plots.py 2>&1 | tee paper_plots.log
echo "Step 2 exit: ${PIPESTATUS[0]}"
