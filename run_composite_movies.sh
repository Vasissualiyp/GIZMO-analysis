#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -p debug
#SBATCH --time=2:00:00
#SBATCH --account=rrg-rbond-ac
#SBATCH --mail-type=ALL

cd $SLURM_SUBMIT_DIR

module load python/3.10

# --- Step 1: Composite frames ---
for fdir in frames18 frames17; do
    for variant in light dark; do
        if [ -d "${fdir}/${variant}/individual_frames" ]; then
            echo "=== Compositing ${fdir} ${variant} ==="
            python make_composite_frames.py "${fdir}" --variant "${variant}"
        fi
    done
done

# --- Step 2: Movies from composites ---
for fdir_out in "frames18:disk_movie_cutout" "frames17:disk_movie_full"; do
    IFS=: read -r fdir outpfx <<< "$fdir_out"
    for variant in light dark; do
        SUFFIX=""
        [ "$variant" = "dark" ] && SUFFIX="_dark"
        COMPDIR="${fdir}/${variant}/composite"
        if [ -d "$COMPDIR" ] && ls "$COMPDIR"/frame_composite_*.png &>/dev/null; then
            FILELIST="${fdir}/filelist_composite${SUFFIX}.txt"
            ls "$COMPDIR"/frame_composite_*.png | sort | while IFS= read -r f; do
                printf "file '%s'\n" "$(realpath "$f")"
            done > "$FILELIST"

            if [ -s "$FILELIST" ]; then
                ffmpeg -y \
                    -f concat -safe 0 \
                    -r 10 \
                    -i "$FILELIST" \
                    -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" \
                    -c:v libx264 -crf 18 -pix_fmt yuv420p \
                    "${outpfx}_composite${SUFFIX}.mp4"
                echo "Movie: ${outpfx}_composite${SUFFIX}.mp4"
            fi
        else
            echo "No composite frames in $COMPDIR, skipping"
        fi
    done
done

mkdir -p frames/movies
mv disk_movie_*_composite*.mp4 frames/movies/ 2>/dev/null || true

echo "Done. Composite movies:"
ls -lh frames/movies/*composite* 2>/dev/null || echo "(none)"
