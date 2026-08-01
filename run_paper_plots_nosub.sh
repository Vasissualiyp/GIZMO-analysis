#!/bin/bash
# Local execution wrapper — no sbatch, no module system required.
# Uses /tmp/sci_venv if available; falls back to cluster jupyter_env.
export LD_LIBRARY_PATH="/nix/store/ab3753m6i7isgvzphlar0a8xb84gl96i-gcc-15.2.0-lib/lib:/nix/store/2kdz3m7ic8w226pcvkz1dlg169v91p6a-zlib-1.3.2/lib:${LD_LIBRARY_PATH:-}"

if [ -x /tmp/sci_venv/bin/python ]; then
    PYTHON=/tmp/sci_venv/bin/python
elif [ -f "$HOME/PYTHON/GUAC/jupyter_env/bin/activate" ]; then
    source "$HOME/PYTHON/GUAC/jupyter_env/bin/activate"
    PYTHON=python
else
    echo "ERROR: no usable Python env found" >&2
    exit 1
fi

cd "$(dirname "$0")"
$PYTHON -u generate_paper_plots.py 2>&1 | tee paper_plots.log
echo "Exit code: ${PIPESTATUS[0]}" >> paper_plots.log
