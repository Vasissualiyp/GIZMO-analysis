#!/bin/bash
#SBATCH -N 1 -n 1 --time=1:00:00 -p debug --account=rrg-rbond-ac --mail-type=ALL

cd $SLURM_SUBMIT_DIR

module load python/3.10
source $HOME/PYTHON/GUAC/jupyter_env/bin/activate

python -u disk_analysis/plot_sink_history.py \
    --cutout-dir /scratch/vasissua/COPY/2026-03/m12f_cutout/output_jeans_refinement \
    --fullsim-dir /scratch/vasissua/COPY/2026-03/m12f/output_jeans_refinement \
    --outdir /scratch/vasissua/SHIVAN/analysis/paper_plots 2>&1

scontrol show job $SLURM_JOB_ID
sacct -j $SLURM_JOB_ID
