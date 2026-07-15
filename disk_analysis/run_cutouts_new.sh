#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=1:00:00
#SBATCH -p debug
#SBATCH --account=rrg-rbond-ac
#SBATCH --job-name=cutouts_new
#SBATCH --mail-type=ALL
#SBATCH --output=cutouts_new_%j.log

module load python/3.10
source $HOME/PYTHON/GUAC/jupyter_env/bin/activate

cd /scratch/vasissua/SHIVAN/analysis

# Run without --overwrite: skips existing cutouts, only creates missing ones
python disk_analysis/create_cutouts.py \
    --input-dir /scratch/vasissua/COPY/2026-03/m12f_cutout/output_jeans_refinement \
    --output-dir /scratch/vasissua/COPY/2026-03/m12f_cutout/output_cutout \
    --cutout-radius 0.005

echo "Cutout creation done."
