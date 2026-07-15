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

# Ensure cmasher is available; eclipse colormap also registered as inline fallback
pip install --quiet cmasher 2>/dev/null || true

python -u generate_paper_plots.py 2>&1 | tee paper_plots.log
