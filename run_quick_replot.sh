#!/bin/bash
#SBATCH --nodes=1
#SBATCH -p debug
#SBATCH --ntasks-per-node=1
#SBATCH --time=0:15:00
#SBATCH --account=rrg-rbond-ac
#SBATCH --mail-type=ALL

cd $SLURM_SUBMIT_DIR

module load python/3.10
source $HOME/PYTHON/GUAC/jupyter_env/bin/activate
pip install --quiet cmasher 2>/dev/null || true

python -u run_quick_replot.py 2>&1
