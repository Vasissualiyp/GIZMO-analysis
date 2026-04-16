#!/bin/bash
#SBATCH --nodes=1
#SBATCH -p debug
#SBATCH --account=rrg-rbond-ac
##SBATCH --account=rrg-murray-ac
#SBATCH --ntasks-per-node=192
#SBATCH	--time=1:00:00
##SBATCH --job-name=GIZMO_m12b_FIRE3
##SBATCH --output=gizmo_m12b_fire3.txt
##SBATCH --dependency=afterany:14592839
#SBATCH --mail-type=ALL

#cd /scratch/vasissua/SHIVAN/analysis
cd $SLURM_SUBMIT_DIR

module load python/3.10
source $HOME/PYTHON/GUAC/jupyter_env/bin/activate

plotter_file="jupytertest3.py"

echo "" > output.log
python -u "$plotter_file" >> output.log
