#!/bin/bash
# Recompute frames18/mass_evolution.npz (density-peak center fix).
export LD_LIBRARY_PATH="/nix/store/ab3753m6i7isgvzphlar0a8xb84gl96i-gcc-15.2.0-lib/lib:/nix/store/2kdz3m7ic8w226pcvkz1dlg169v91p6a-zlib-1.3.2/lib:${LD_LIBRARY_PATH:-}"

if [ -x /tmp/sci_venv/bin/python ]; then
    PYTHON=/tmp/sci_venv/bin/python
else
    echo "ERROR: no /tmp/sci_venv" >&2
    exit 1
fi

cd "$(dirname "$0")"
$PYTHON -u disk_analysis/compute_mass_evolution.py --overwrite 2>&1 | tee mass_evolution_recompute.log
echo "Exit code: ${PIPESTATUS[0]}"
