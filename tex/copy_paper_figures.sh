#!/usr/bin/env bash
# copy_paper_figures.sh
# Copies only the plots needed by paper_draft_human.tex into tex/figures/
# Run from the tex/ directory: bash copy_paper_figures.sh

set -euo pipefail

# ── Source directories (frames13 = full sim, frames14 = cutout) ──
BASE="/home/vasilii/research/trillium/scratch/SHIVAN/analysis"
BASE="/scratch/vasissua/SHIVAN/analysis"
FRAMES_FULL="$BASE/frames17"
FRAMES_CUT="$BASE/frames18"
COLOR="dark"
OUT="figures_dark"

# ── Destination ──
DEST="$(cd "$(dirname "$0")" && pwd)/$OUT"

# ── Representative snapshots per epoch ──
SNAPS_FULL=(0000 0050 0206 0276)
SNAPS_CUT=(0000 0050 0193 0429)

# ── Per-snapshot individual frame types ──
FRAME_TYPES=(frame_density frame_velocity frame_toomre frame_kinematics frame_profiles_log frame_profiles_lin frame_velprof frame_velprof_wide frame_stability frame_derotated)
FRAME_TYPES_OPTIONAL=(frame_bfield)

# ── Counters ──
copied=0
skipped=0

copy_file() {
    local src="$1" dst="$2"
    if [ -f "$src" ]; then
        cp "$src" "$dst"
        copied=$((copied + 1))
    else
        echo "  MISSING: $src"
        skipped=$((skipped + 1))
    fi
}

# ── Clean and create destination ──
rm -rf "$DEST"
mkdir -p "$DEST/full" "$DEST/cutout"

echo "=== Copying full-sim plots (frames13) ==="
for snap in "${SNAPS_FULL[@]}"; do
    echo "  Snap $snap:"
    for ftype in "${FRAME_TYPES[@]}"; do
        copy_file "$FRAMES_FULL/$COLOR/individual_frames/${ftype}_${snap}.png" "$DEST/full/"
    done
    for ftype in "${FRAME_TYPES_OPTIONAL[@]}"; do
        copy_file "$FRAMES_FULL/$COLOR/individual_frames/${ftype}_${snap}.png" "$DEST/full/" 2>/dev/null || true
    done
    # Phase diagram
    copy_file "$FRAMES_FULL/$COLOR/T_H2_rho_phase_plots/phase_${snap}.png" "$DEST/full/"
    # Velocity power spectrum
    copy_file "$FRAMES_FULL/$COLOR/velocity_power_spectra/plots/vps_${snap}.png" "$DEST/full/"
    # Q combo
    copy_file "$FRAMES_FULL/$COLOR/Q_combo/frame_Q_combo_${snap}.png" "$DEST/full/" 2>/dev/null || true
done

# Global evolution plots (full sim)
echo "  Global plots:"
for gplot in mass_evolution.png mass_evolution_rates.png energy_evolution.png Q_heatmap.png sigma_heatmap.png sigma_r_heatmap.png; do
    copy_file "$FRAMES_FULL/$COLOR/$gplot" "$DEST/full/"
done

echo ""
echo "=== Copying cutout plots (frames14) ==="
for snap in "${SNAPS_CUT[@]}"; do
    echo "  Snap $snap:"
    for ftype in "${FRAME_TYPES[@]}"; do
        copy_file "$FRAMES_CUT/$COLOR/individual_frames/${ftype}_${snap}.png" "$DEST/cutout/"
    done
    for ftype in "${FRAME_TYPES_OPTIONAL[@]}"; do
        copy_file "$FRAMES_CUT/$COLOR/individual_frames/${ftype}_${snap}.png" "$DEST/cutout/" 2>/dev/null || true
    done
    # Phase diagram
    copy_file "$FRAMES_CUT/$COLOR/T_H2_rho_phase_plots/phase_${snap}.png" "$DEST/cutout/"
    # Velocity power spectrum
    copy_file "$FRAMES_CUT/$COLOR/velocity_power_spectra/plots/vps_${snap}.png" "$DEST/cutout/"
    # Q combo
    copy_file "$FRAMES_CUT/$COLOR/Q_combo/frame_Q_combo_${snap}.png" "$DEST/cutout/" 2>/dev/null || true
done

# Global evolution plots (cutout)
echo "  Global plots:"
for gplot in mass_evolution.png mass_evolution_rates.png energy_evolution.png Q_heatmap.png sigma_heatmap.png sigma_r_heatmap.png; do
    copy_file "$FRAMES_CUT/$COLOR/$gplot" "$DEST/cutout/"
done

echo ""
echo "=== Summary ==="
echo "  Copied:  $copied files"
echo "  Missing: $skipped files"
echo "  Destination: $DEST"
