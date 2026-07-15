#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=1:00:00
#SBATCH -p debug
#SBATCH --account=rrg-rbond-ac
#SBATCH --job-name=cutouts_paper
#SBATCH --mail-type=ALL
#SBATCH --output=cutouts_paper_%j.log

module load python/3.10
source $HOME/PYTHON/GUAC/jupyter_env/bin/activate

cd /scratch/vasissua/SHIVAN/analysis

echo "=== Step 1: Creating cutouts for new snapshots ==="
python disk_analysis/create_cutouts.py \
    --input-dir /scratch/vasissua/COPY/2026-03/m12f_cutout/output_jeans_refinement \
    --output-dir /scratch/vasissua/COPY/2026-03/m12f_cutout/output_cutout \
    --cutout-radius 0.005
echo "Cutout step done."

echo "=== Step 2: Generating paper plots ==="
python -u generate_paper_plots.py 2>&1 | tee paper_plots.log
echo "Paper plots step done."
