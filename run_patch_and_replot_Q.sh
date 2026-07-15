#!/bin/bash
#SBATCH -N 1 -n 1 --time=1:00:00 -p debug --account=rrg-rbond-ac --mail-type=ALL

cd $SLURM_SUBMIT_DIR

module load python/3.10
source $HOME/PYTHON/GUAC/jupyter_env/bin/activate

echo "=== Step 1: Patch qprofiles with Q_combined (thermal+turbulent) ==="
python -u disk_analysis/patch_qprofiles_combined_Q.py --overwrite 2>&1

echo ""
echo "=== Step 2: Replot merged Toomre Q figure ==="
python -u run_replot_toomre_Q.py 2>&1

echo ""
echo "=== Done ==="
scontrol show job $SLURM_JOB_ID
sacct -j $SLURM_JOB_ID
