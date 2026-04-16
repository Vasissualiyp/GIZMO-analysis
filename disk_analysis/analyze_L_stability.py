#!/usr/bin/env python
"""
Analyze angular momentum vector stability for different sphere radii.

This script loads the saved angular momentum vectors for different radii
and calculates the angular separation between consecutive snapshots to
determine which radius gives the most stable L vector.

Usage:
    python analyze_L_stability.py
"""

import sys, os
import numpy as np
import matplotlib.pyplot as plt

# Setup paths
scratch_analysis_path = "/scratch/vasissua/SHIVAN/analysis/"
sys.path.insert(0, scratch_analysis_path)
import meshoid_plotting.starforge_plot as sfp

from vasthemer import set_theme
plt.style.use('dark_background')

# Configuration
scratch_path = "/scratch/vasissua/"
out_path = os.path.join(scratch_path, "SHIVAN", "analysis")
run_out_path = os.path.join(scratch_path, "COPY/2026-03/m12f/output_jeans_refinement")
run_subdir = "8x_zoom_m12f"

# Extract snapshot numbers
snap_nos = sorted([a.split("_")[1].split(".")[0]
                   for a in os.listdir(run_out_path)
                   if "snapshot_" in a and a[-4:] == "hdf5"])
snap_nos = snap_nos[::-1]  # Reverse

# Load angular momentum vectors
# Load vectors (now returns (n_snaps, n_radii, 3))
L_vectors, radii = sfp.load_angular_momentum_vectors(out_path, snap_nos, run_subdir)

# --- NEW CALCULATION LOGIC ---
# Check for actual NaN values and zero-norm vectors
print(f"\nVector magnitude statistics:")
norms_all = np.linalg.norm(L_vectors, axis=2)
print(f"  Min norm: {np.nanmin(norms_all):.6e}")
print(f"  Max norm: {np.nanmax(norms_all):.6e}")
print(f"  Median norm: {np.nanmedian(norms_all):.6e}")

# Only filter out actual NaN values and truly zero vectors (norm < machine epsilon * median)
true_zero_threshold = 1e-30  # Very conservative threshold
nan_mask = np.isnan(L_vectors).any(axis=2)
zero_norm_mask = norms_all < true_zero_threshold

invalid_mask = nan_mask | zero_norm_mask
n_invalid = np.sum(invalid_mask)
print(f"  Invalid vectors (NaN or truly zero): {n_invalid}/{L_vectors.size//3}")

# Keep all data, just mark invalid entries
L_vectors_filtered = L_vectors.copy()
L_vectors_filtered[invalid_mask] = np.nan

# Don't remove entire rows - work with what we have
# Only remove radii that are ALL invalid
valid_radii = ~invalid_mask.all(axis=0)
L_vectors_clean = L_vectors_filtered[:, valid_radii, :]
radii_clean = radii[valid_radii]

print(f"\nRemoved {np.sum(~valid_radii)} radii with all invalid values")
print(f"Using all {len(snap_nos)} snapshots and {len(radii_clean)} radii")

if len(radii_clean) == 0:
    print("\nERROR: No valid radii remaining!")
    sys.exit(1)

# Find the snapshot with the most valid vectors to use as reference
# Count valid vectors per snapshot (only for clean radii)
valid_counts_per_snap = (~invalid_mask[:, valid_radii]).sum(axis=1)
print(f"\nValid vectors per snapshot range: {valid_counts_per_snap.min()} to {valid_counts_per_snap.max()}")

# Try to use the latest snapshot with at least one valid vector
# Search backwards from the end
ref_snap_idx = None
for i in range(len(snap_nos)-1, -1, -1):
    if valid_counts_per_snap[i] > 0:
        ref_snap_idx = i
        break

if ref_snap_idx is None:
    print("\nERROR: No snapshots have valid L vectors!")
    sys.exit(1)

# Find first valid radius in the reference snapshot
mask_ref = ~invalid_mask[ref_snap_idx, valid_radii]
ref_radius_idx = np.where(mask_ref)[0][0]
ref_vec = L_vectors_clean[ref_snap_idx, ref_radius_idx]
ref_norm = np.linalg.norm(ref_vec)

print(f"\nReference vector: Snapshot {ref_snap_idx} (snap {snap_nos[ref_snap_idx]}), radius index {ref_radius_idx} (R={radii_clean[ref_radius_idx]:.2e})")
print(f"  Reference vector: {ref_vec}")
print(f"  Reference vector norm: {ref_norm:.6e}")
print(f"  This snapshot has {valid_counts_per_snap[ref_snap_idx]} valid radii")

# Vectorized angular separation from ref_vec
dot_prod = np.einsum('ijk,k->ij', L_vectors_clean, ref_vec)
norms = np.linalg.norm(L_vectors_clean, axis=2)

# Calculate angles (N_snaps, N_radii), handling any remaining edge cases
# Calculate raw angles: [0, 180]
with np.errstate(divide='ignore', invalid='ignore'):
    cos_angle = dot_prod / (norms * ref_norm)
    raw_angles = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))

# Fold angles to [0, 90] because V and -V are the same
angular_separations = np.minimum(raw_angles, 180 - raw_angles)

# Now calculate stats on the folded angles
mean_separations = np.nanmean(angular_separations, axis=0)
std_separations = np.nanstd(angular_separations, axis=0)
max_separations = np.nanmax(angular_separations, axis=0)

# Identify best radius
best_radius_idx = np.nanargmin(mean_separations)

# Check if we have any valid data
n_valid_per_radius = np.sum(~np.isnan(angular_separations), axis=0)
print(f"\nValid angular separation measurements per radius:")
for i, (r, n) in enumerate(zip(radii_clean, n_valid_per_radius)):
    print(f"  Radius {i} (R={r:.2e}): {n}/{len(snap_nos)} valid")

if np.all(np.isnan(mean_separations)):
    print("\nERROR: All angular separations are NaN!")
    print("This suggests all L vectors are zero or invalid.")
    sys.exit(1)

# --- UPDATED PLOTTING ---
fig, axes = plt.subplots(2, 1, figsize=(12, 10))

# Plot 1: Evolution relative to final snapshot
ax1 = axes[0]
n_snaps = len(snap_nos)

for j in range(len(radii_clean)):
    ax1.plot(range(n_snaps), angular_separations[:, j],
             label=f"R={radii_clean[j]:.2e} {'(Best)' if j==best_radius_idx else ''}",
             alpha=1.0 if j==best_radius_idx else 0.5,
             lw=2 if j==best_radius_idx else 1)

ax1.set_xlabel("Snapshot Index")
ax1.set_ylabel("Angle from Reference Vector [deg]")
ax1.set_title(f"Deviation from Reference (Snap {snap_nos[ref_snap_idx]}, R={radii_clean[ref_radius_idx]:.2e})")
ax1.legend(ncol=2, fontsize=8)
ax1.grid(True, alpha=0.3)

# Plot 2: Stats
ax2 = axes[1]
x = np.arange(len(radii_clean))
ax2.bar(x - 0.2, mean_separations, 0.4, label='Mean Dev')
ax2.bar(x + 0.2, std_separations, 0.4, label='Std Dev')
ax2.set_xlabel("Sphere Radius")
ax2.set_ylabel("Angular Deviation [deg]")
ax2.set_xticks(x)
ax2.set_xticklabels([f"{r:.1e}" for r in radii_clean], rotation=45)
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
output_filename = "L_stability_from_reference.png"
plt.savefig(os.path.join(out_path, run_subdir, output_filename))

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
print(f"\nReference: Snapshot {ref_snap_idx} (snap {snap_nos[ref_snap_idx]}), R={radii_clean[ref_radius_idx]:.2e}")
print(f"\nBest radius (minimum mean deviation): R = {radii_clean[best_radius_idx]:.2e}")
print(f"  Mean deviation: {mean_separations[best_radius_idx]:.3f} deg")
print(f"  Std deviation:  {std_separations[best_radius_idx]:.3f} deg")
print(f"  Max deviation:  {max_separations[best_radius_idx]:.3f} deg")
print(f"\nPlot saved to: {os.path.join(out_path, run_subdir, output_filename)}")
print("="*80)
