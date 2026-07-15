#!/bin/bash
#SBATCH -N 1 -n 1 --time=0:30:00 -p debug --account=rrg-rbond-ac --mail-type=ALL
#SBATCH --job-name=sink_last

cd $SLURM_SUBMIT_DIR

module load python/3.10
source $HOME/PYTHON/GUAC/jupyter_env/bin/activate

echo "=== Last-snapshot sink classification ==="
python -u disk_analysis/sink_last_snap.py 2>&1

echo ""
echo "=== Gas radial distribution plot ==="
python -u disk_analysis/plot_gas_radial_dist.py 2>&1

scontrol show job $SLURM_JOB_ID
sacct -j $SLURM_JOB_ID
