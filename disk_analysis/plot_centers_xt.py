#!/usr/bin/env python
"""
Plot x(t) for box centers from the 8x_zoom_m12f simulation.
Uses functions from starforge_plot.py to load and smooth center data.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display

# Setup paths (following jupytertest2.py style)
scratch_analysis_path = "/scratch/vasissua/SHIVAN/analysis/"
sys.path.insert(0, scratch_analysis_path)
import meshoid_plotting.starforge_plot as sfp

plt.style.use('dark_background')

# Configuration
out_path = scratch_analysis_path
run_subdir = "8x_zoom_m12f"
box_path = os.path.join(out_path, run_subdir)

# Get all snapshot numbers from directory
snap_dirs = sorted([d for d in os.listdir(box_path)
                   if os.path.isdir(os.path.join(box_path, d))
                   and d.isdigit()])
print(f"Found {len(snap_dirs)} snapshots")

# Load centers using the function from starforge_plot.py
centers, velocities, times, _ = sfp.load_centers_from_files(out_path, snap_dirs, run_subdir)

# Convert times from seconds to Myr for better readability
times_myr = times / (1e6 * 365.25 * 24 * 3600)

# Filter out NaN values
valid_mask = ~np.isnan(centers[:, 0])
centers_valid = centers[valid_mask]
times_valid = times_myr[valid_mask]

print(f"Valid data points: {np.sum(valid_mask)} out of {len(centers)}")

# Smooth the centers using the function from starforge_plot.py
smoothing_sigma = 1.0
if len(centers_valid) > 1:
    centers_smoothed = sfp.smooth_vector_evolution(centers_valid, sigma=smoothing_sigma)
else:
    centers_smoothed = centers_valid.copy()

# Calculate dx/dt (velocity) using finite differences
# Use smoothed centers for cleaner derivatives
dt = np.diff(times_valid)  # Time differences in Myr
dx_raw = np.diff(centers_valid, axis=0)  # Position differences (raw)
dx_smoothed = np.diff(centers_smoothed, axis=0)  # Position differences (smoothed)

# Convert to velocity: kpc/Myr
vx_raw = dx_raw[:, 0] / dt
vy_raw = dx_raw[:, 1] / dt
vz_raw = dx_raw[:, 2] / dt

vx_smoothed = dx_smoothed[:, 0] / dt
vy_smoothed = dx_smoothed[:, 1] / dt
vz_smoothed = dx_smoothed[:, 2] / dt

# Midpoint positions for plotting (since velocity is between two points)
x_mid_raw = 0.5 * (centers_valid[:-1, 0] + centers_valid[1:, 0])
x_mid_smoothed = 0.5 * (centers_smoothed[:-1, 0] + centers_smoothed[1:, 0])

# Create the phase space plot: dx/dt vs x
fig, ax = plt.subplots(figsize=(10, 8))

# Plot raw dx/dt vs x
ax.plot(centers_valid[:-1, 0], vx_raw, 'c-', alpha=0.3, linewidth=1,
        label='raw')

# Plot smoothed dx/dt vs x
ax.plot(centers_smoothed[:-1, 0], vx_smoothed, 'c-', linewidth=2,
        label='smoothed')

# Add arrows to show direction of time evolution
n_arrows = 10
arrow_indices = np.linspace(0, len(vx_smoothed)-2, n_arrows, dtype=int)
for idx in arrow_indices:
    ax.annotate('', xy=(centers_smoothed[idx+1, 0], vx_smoothed[idx+1]),
                xytext=(centers_smoothed[idx, 0], vx_smoothed[idx]),
                arrowprops=dict(arrowstyle='->', color='yellow', lw=1.5))

# Labels and formatting
ax.set_xlabel('x position [kpc]', fontsize=14)
ax.set_ylabel('dx/dt [kpc/Myr]', fontsize=14)
ax.set_title('Phase Space: dx/dt vs x (8x_zoom_m12f)', fontsize=16)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)

plt.tight_layout()

# Save the plot
output_file = os.path.join(scratch_analysis_path, 'meshoid_plotting', 'centers_dxdt_vs_x.png')
fig.savefig(output_file, dpi=150)
print(f"Plot saved to: {output_file}")
display(fig)

# Also create phase space plots for all three coordinates
fig2, axes = plt.subplots(1, 3, figsize=(15, 5))

labels = ['x', 'y', 'z']
colors = ['cyan', 'lime', 'red']
velocities_raw = [vx_raw, vy_raw, vz_raw]
velocities_smoothed = [vx_smoothed, vy_smoothed, vz_smoothed]

for i, (ax, label, color) in enumerate(zip(axes, labels, colors)):
    ax.plot(np.arange(len(velocities_raw[0])), velocities_raw[i], color=color, alpha=0.3,
            linewidth=1, label='raw')
    ax.plot(np.arange(len(velocities_raw[0])), velocities_smoothed[i], color=color,
            linewidth=2, label='smoothed')
    #ax.plot(centers_valid[:-1, i], velocities_raw[i], color=color, alpha=0.3,
    #        linewidth=1, label='raw')
    #ax.plot(centers_smoothed[:-1, i], velocities_smoothed[i], color=color,
    #        linewidth=2, label='smoothed')
    ax.set_xlabel(f'{label} [kpc]', fontsize=12)
    ax.set_ylabel(f'd{label}/dt [kpc/Myr]', fontsize=12)
    ax.set_title(f'd{label}/dt vs {label}', fontsize=14)
    ax.set_ylim([-2, 2])
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

fig2.suptitle('Phase Space: Velocity vs Position (8x_zoom_m12f)', fontsize=16)

plt.tight_layout()

output_file_all = os.path.join(scratch_analysis_path, 'meshoid_plotting', 'centers_phase_space_xyz.png')
fig2.savefig(output_file_all, dpi=150)
print(f"All coordinates phase space plot saved to: {output_file_all}")
display(fig2)
