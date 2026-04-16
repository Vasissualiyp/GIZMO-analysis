#!/usr/bin/env bash

ffmpeg -framerate 10 -pattern_type glob -i './frames/frame_*.png' -c:v libx264 -crf 18 -pix_fmt yuv420p disk_movie.mp^C

