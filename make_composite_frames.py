#!/usr/bin/env python3
"""
Composite all existing individual frames + Q_combo + phase into one
big master image per snapshot. No re-rendering — just PIL paste.

Layout (3 columns, top-to-bottom):
  Col 0 (left):      density, derotated, velprof, velprof_wide
  Col 1 (center):    velocity, toomre, bfield, stability
  Col 2 (right):     kinematics, profiles_log, profiles_lin, Q_combo, phase

Each image is scaled to a common column width; aspect ratio preserved.
"""

import os
import sys
import glob
from PIL import Image

COL_WIDTH = 1200  # px per column
MARGIN = 4        # px between tiles
BG_COLOR = (0, 0, 0)  # black background


def find_frame(frames_dir, variant, subdir, prefix, snap):
    """Try to find a frame PNG.  Returns path or None."""
    patterns = [
        os.path.join(frames_dir, variant, subdir, f"{prefix}_{snap:04d}.png"),
    ]
    for p in patterns:
        if os.path.isfile(p):
            return p
    return None


def resize_to_width(img, target_w):
    """Resize preserving aspect ratio."""
    w, h = img.size
    if w == target_w:
        return img
    scale = target_w / w
    return img.resize((target_w, int(h * scale)), Image.LANCZOS)


def stack_images(images, col_width, margin):
    """Stack images vertically, return (composite_image, height)."""
    resized = [resize_to_width(im, col_width) for im in images]
    total_h = sum(im.size[1] for im in resized) + margin * (len(resized) - 1)
    col_img = Image.new('RGB', (col_width, total_h), BG_COLOR)
    y = 0
    for im in resized:
        col_img.paste(im, (0, y))
        y += im.size[1] + margin
    return col_img


def make_composite(frames_dir, variant, snap, outpath):
    """Build composite for one snapshot. Returns True if created."""

    # Define layout: (subdir, prefix)
    col0_spec = [
        ("individual_frames", "frame_density"),
        ("individual_frames", "frame_derotated"),
        ("individual_frames", "frame_velprof"),
        ("individual_frames", "frame_velprof_wide"),
    ]
    col1_spec = [
        ("individual_frames", "frame_velocity"),
        ("individual_frames", "frame_toomre"),
        ("individual_frames", "frame_bfield"),
        ("individual_frames", "frame_stability"),
    ]
    col2_spec = [
        ("individual_frames", "frame_kinematics"),
        ("individual_frames", "frame_profiles_log"),
        ("individual_frames", "frame_profiles_lin"),
        ("Q_combo", "frame_Q_combo"),
        ("T_H2_rho_phase_plots", "phase"),
    ]

    def load_col(spec):
        imgs = []
        for subdir, prefix in spec:
            path = find_frame(frames_dir, variant, subdir, prefix, snap)
            if path:
                imgs.append(Image.open(path).convert('RGB'))
        return imgs

    c0 = load_col(col0_spec)
    c1 = load_col(col1_spec)
    c2 = load_col(col2_spec)

    if not c0 and not c1 and not c2:
        return False

    cols = []
    for images in [c0, c1, c2]:
        if images:
            cols.append(stack_images(images, COL_WIDTH, MARGIN))
        else:
            cols.append(Image.new('RGB', (COL_WIDTH, 1), BG_COLOR))

    max_h = max(c.size[1] for c in cols)
    total_w = COL_WIDTH * 3 + MARGIN * 2
    composite = Image.new('RGB', (total_w, max_h), BG_COLOR)

    x = 0
    for col_img in cols:
        composite.paste(col_img, (x, 0))
        x += COL_WIDTH + MARGIN

    composite.save(outpath, optimize=True)
    return True


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("frames_dir", help="e.g. frames18")
    parser.add_argument("--variant", default="light", choices=["light", "dark"])
    args = parser.parse_args()

    frames_dir = args.frames_dir
    variant = args.variant

    outdir = os.path.join(frames_dir, variant, "composite")
    os.makedirs(outdir, exist_ok=True)

    # Discover snapshots from density frames (most common)
    pattern = os.path.join(frames_dir, variant, "individual_frames", "frame_density_*.png")
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"No density frames in {frames_dir}/{variant}/individual_frames/")
        sys.exit(1)

    snaps = []
    for f in files:
        base = os.path.basename(f)
        num = int(base.split('_')[-1].replace('.png', ''))
        snaps.append(num)

    print(f"Found {len(snaps)} snapshots, compositing {variant} frames...")

    count = 0
    for snap in snaps:
        outpath = os.path.join(outdir, f"frame_composite_{snap:04d}.png")
        if os.path.isfile(outpath):
            continue
        if make_composite(frames_dir, variant, snap, outpath):
            count += 1
            if count % 50 == 0:
                print(f"  {count} composites done...")

    print(f"Done. Created {count} new composites in {outdir}/")


if __name__ == "__main__":
    main()
