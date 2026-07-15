#!/bin/bash
#SBATCH --nodes=1
#SBATCH -p debug
#SBATCH --ntasks-per-node=1
#SBATCH --time=1:00:00
#SBATCH --account=rrg-rbond-ac
#SBATCH --mail-type=ALL

cd $SLURM_SUBMIT_DIR

module load python/3.10
source $HOME/PYTHON/GUAC/jupyter_env/bin/activate

python -u disk_analysis/compute_phase_stats.py --overwrite "$@" 2>&1 | tee phase_stats.log
