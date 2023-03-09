#!/bin/bash

source ~/research/pyenv/python3/bin/activate

python snapshots_to_plots.py
python plots_to_movie.py
