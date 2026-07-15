#!/bin/bash
#SBATCH --nodes=1
#SBATCH -p debug
#SBATCH --ntasks-per-node=1
#SBATCH --time=0:30:00
#SBATCH --account=rrg-rbond-ac

cd $SLURM_SUBMIT_DIR

module load python/3.10
source $HOME/PYTHON/GUAC/jupyter_env/bin/activate

python -u disk_analysis/plot_dm_profile.py 2>&1

scontrol show job $SLURM_JOB_ID
sacct -j $SLURM_JOB_ID
