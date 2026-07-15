#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -p debug
#SBATCH --time=1:00:00
#SBATCH --account=rrg-rbond-ac
#SBATCH --mail-type=ALL

cd $SLURM_SUBMIT_DIR

module load python/3.10

# Movies for cutout (frames18)
./disk_analysis/make_movie_from_frames.sh frames18 disk_movie_cutout 10

# Movies for full sim (frames17)
./disk_analysis/make_movie_from_frames.sh frames17 disk_movie_full 10

mkdir -p frames/movies
mv disk_movie_*.mp4 frames/movies/ 2>/dev/null || true

echo "Done. Movies in frames/movies/"
ls -lh frames/movies/disk_movie_*.mp4
