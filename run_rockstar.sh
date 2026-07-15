#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=1:00:00
#SBATCH -p debug
#SBATCH --account=rrg-rbond-ac
#SBATCH --job-name=rockstar_m12f
#SBATCH --output=/scratch/vasissua/SHIVAN/analysis/slurm-%j.out

set -euo pipefail

ROCKSTAR_DIR="/scratch/vasissua/SHIVAN/rockstar-gizmo"
ROCKSTAR_BIN="$ROCKSTAR_DIR/rockstar"
CONFIG="/scratch/vasissua/SHIVAN/analysis/rockstar_m12f.cfg"
SNAPSHOT="/scratch/vasissua/COPY/2026-03/m12f/output_jeans_refinement/snapshot_270.hdf5"
OUTDIR="/scratch/vasissua/SHIVAN/rockstar_output"

echo "=== Running Rockstar on m12f snapshot 270 ==="
echo "Date: $(date)"
echo "Node: $(hostname)"

# Load modules (need HDF5 runtime libraries)
module purge
module load StdEnv/2023 gcc/12.3 openmpi/4.1.5 hdf5/1.14.2

# Verify binary exists
if [ ! -f "$ROCKSTAR_BIN" ]; then
    echo "ERROR: Rockstar binary not found at $ROCKSTAR_BIN"
    echo "Run build_rockstar.sh first"
    exit 1
fi

# Verify snapshot exists
if [ ! -f "$SNAPSHOT" ]; then
    echo "ERROR: Snapshot not found at $SNAPSHOT"
    exit 1
fi

# Create output directory
mkdir -p "$OUTDIR"

# Copy config to output dir for record keeping
cp "$CONFIG" "$OUTDIR/rockstar_m12f.cfg"

echo "Rockstar binary: $ROCKSTAR_BIN"
echo "Config: $CONFIG"
echo "Snapshot: $SNAPSHOT"
echo "Output dir: $OUTDIR"
echo "Snapshot size: $(ls -lh "$SNAPSHOT" | awk '{print $5}')"
echo ""

# Run Rockstar in single-snapshot mode
# -c config file, then pass the snapshot as argument
cd "$OUTDIR"
"$ROCKSTAR_BIN" -c "$CONFIG" "$SNAPSHOT"

echo ""
echo "=== Rockstar finished ==="
echo "Output files:"
ls -la "$OUTDIR"/
echo "Date: $(date)"
