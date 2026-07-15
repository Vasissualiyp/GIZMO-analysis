#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=0:10:00
#SBATCH -p debug
#SBATCH --account=rrg-rbond-ac
#SBATCH --job-name=check_h2
#SBATCH --output=check_h2_%j.log

module load python/3.10
source $HOME/PYTHON/GUAC/jupyter_env/bin/activate
cd /scratch/vasissua/SHIVAN/analysis
python check_h2_fields.py
