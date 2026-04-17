#!/usr/bin/env bash
# Assemble disk movie from frames using the concat demuxer.
# This handles non-sequential frame numbers (skipped snapshots) correctly.
#
# Usage (run from the analysis root, i.e. the dir that contains frames/):
#   bash disk_analysis/make_movie_from_frames.sh [frames_dir] [output.mp4] [framerate]
#
# Defaults:
#   frames_dir = ./frames
#   output     = disk_movie.mp4
#   framerate  = 10

set -euo pipefail

FRAMES_DIR="${1:-./frames}"
OUTPUT="${2:-disk_movie.mp4}"
FRAMERATE="${3:-10}"
FILELIST="${FRAMES_DIR}/filelist.txt"

if ! ls "${FRAMES_DIR}"/frame_*.png &>/dev/null; then
    echo "Error: no frame_*.png files found in ${FRAMES_DIR}" >&2
    exit 1
fi

# Build the concat file list (sorted by filename = sorted by snap number)
ls "${FRAMES_DIR}"/frame_*.png | sort | while IFS= read -r f; do
    printf "file '%s'\n" "$(basename "$f")"
done > "${FILELIST}"

ffmpeg -y \
    -f concat -safe 0 \
    -r "${FRAMERATE}" \
    -i "${FILELIST}" \
    -c:v libx264 -crf 18 -pix_fmt yuv420p \
    "${OUTPUT}"

echo "Movie saved to: ${OUTPUT}"
