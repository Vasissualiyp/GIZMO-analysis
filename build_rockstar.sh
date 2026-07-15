#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=0:30:00
#SBATCH -p debug
#SBATCH --account=rrg-rbond-ac
#SBATCH --job-name=build_rockstar
#SBATCH --output=/scratch/vasissua/SHIVAN/analysis/slurm-%j.out

set -euo pipefail

ROCKSTAR_DIR="/scratch/vasissua/SHIVAN/rockstar-gizmo"

echo "=== Building Rockstar-GIZMO on Trillium ==="
echo "Date: $(date)"
echo "Node: $(hostname)"

cd "$ROCKSTAR_DIR"
echo "Git commit: $(git rev-parse --short HEAD)"

# ---- Load modules ----
module purge
module load StdEnv/2023 gcc/12.3 openmpi/4.1.5 hdf5/1.14.2

echo "HDF5 root: $EBROOTHDF5"
echo "CC: $(which gcc)"
echo "gcc version: $(gcc --version | head -1)"

# ---- Build with HDF5 support ----
echo ""
echo "Building Rockstar with HDF5 support..."
make clean
make with_hdf5

if [ -f rockstar ]; then
    echo ""
    echo "=== BUILD SUCCESSFUL ==="
    ls -la rockstar
    file rockstar
    echo ""
    echo "Library linkage:"
    ldd rockstar | grep -i hdf5
else
    echo "=== BUILD FAILED ==="
    exit 1
fi

echo ""
echo "Done at $(date)"
