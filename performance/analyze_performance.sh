#!/bin/bash
module load python; source env/bin/activate

while true; do
    python analyze_performance.py
done
