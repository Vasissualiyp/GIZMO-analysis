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

FRAMES_DIR="${1:-./frames18}"
OUTPUT="${2:-disk_movie}"
FRAMERATE="${3:-10}"
FILELIST_BASE="${FRAMES_DIR}/filelist"

if [ ! -d "${FRAMES_DIR}" ]; then
    echo "Error: frames directory ${FRAMES_DIR} does not exist" >&2
    exit 1
fi


plot_all_frames() {
    fname="$1"
    dirname="$2"
    outname="$3"
    local FILELIST="${FILELIST_BASE}_${outname}.txt"
    # Build the concat file list (sorted by filename = sorted by snap number)
    ls "${FRAMES_DIR}"/"${dirname}"/"${fname}"_*.png 2>/dev/null | sort | while IFS= read -r f; do
        printf "file '%s'\n" "$(realpath "$f")"
    done > "${FILELIST}"

    if [ ! -s "${FILELIST}" ]; then
        echo "No ${fname} frames found in ${dirname}, skipping."
        return 1
    fi

    ffmpeg -y \
        -f concat -safe 0 \
        -r "${FRAMERATE}" \
        -i "${FILELIST}" \
        -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" \
        -c:v libx264 -crf 18 -pix_fmt yuv420p \
        "${OUTPUT}_${outname}.mp4"

    echo "Movie saved to: ${OUTPUT}_${outname}.mp4"
}

for variant in light dark; do
    SUFFIX=""
    [ "$variant" = "dark" ] && SUFFIX="_dark"

    plot_all_frames "frame" "${variant}/master_frames" "masterplots${SUFFIX}" || echo "Warning: ${variant} masterplots movie failed"
    plot_all_frames "phase" "${variant}/T_H2_rho_phase_plots" "T_H2_rho_phaseplots${SUFFIX}" || echo "Warning: ${variant} phase movie failed"
    plot_all_frames "vps" "${variant}/velocity_power_spectra/plots" "vps_plots${SUFFIX}" || echo "Warning: ${variant} VPS movie failed"

    # Individual frame type movies (non-fatal — some types may not exist)
    for pair in \
        "frame_density:${variant}/individual_frames:density${SUFFIX}" \
        "frame_velocity:${variant}/individual_frames:velocity${SUFFIX}" \
        "frame_toomre:${variant}/individual_frames:toomre${SUFFIX}" \
        "frame_profiles_log:${variant}/individual_frames:profiles_log${SUFFIX}" \
        "frame_profiles_lin:${variant}/individual_frames:profiles_lin${SUFFIX}" \
        "frame_velprof:${variant}/individual_frames:velprof${SUFFIX}" \
        "frame_velprof_wide:${variant}/individual_frames:velprof_wide${SUFFIX}" \
        "frame_kinematics:${variant}/individual_frames:kinematics${SUFFIX}" \
        "frame_stability:${variant}/individual_frames:stability${SUFFIX}" \
        "frame_derotated:${variant}/individual_frames:derotated${SUFFIX}" \
        "frame_Q_combo:${variant}/Q_combo:Q_combo${SUFFIX}"; do
        IFS=: read -r fname dirname outname <<< "$pair"
        plot_all_frames "$fname" "$dirname" "$outname" || echo "Warning: no ${variant} $fname frames found, skipping movie"
    done
done
