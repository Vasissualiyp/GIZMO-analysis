# SLURM Job Submission and Monitoring Guide

This document explains how to submit and monitor SLURM jobs for this project
from the **local machine** (NixOS workstation), where `sbatch`/`squeue`/`sacct`
are NOT available.

---

## Key facts

- The local machine and the cluster share a filesystem mounted at:
  - **Local path**: `/home/vasilii/research/trillium/scratch/`
  - **Cluster path**: `/scratch/vasissua/`
  - These are the same data — just different mount points.
- The analysis directory:
  - Local: `/home/vasilii/research/trillium/scratch/SHIVAN/analysis/`
  - Cluster: `/scratch/vasissua/SHIVAN/analysis/`
- `sbatch`, `squeue`, `sacct`, `scontrol` only exist on the cluster login node.
- **Never run `sbatch` locally** — it does not exist and will fail silently.

---

## How job submission works

A daemon called `queue_runner.sh` runs on the **cluster login node**. It watches
`scripts.txt` in the analysis directory and executes one command at a time every
~10 seconds. To submit a job, append a line to that file from the local machine.

---

## Submitting a job

### Step 1 — Check queue_runner is not running locally (common mistake)

```bash
ps aux | grep queue_runner | grep -v grep
```

If a local process is found, kill it immediately. It will silently consume lines
from `scripts.txt` but fail because `sbatch` doesn't exist locally. The daemon
must run only on the cluster.

### Step 2 — Note the current latest slurm output file

```bash
ls -lt /home/vasilii/research/trillium/scratch/SHIVAN/analysis/slurm-*.out | head -1
```

Record this job ID so you can detect when a new one appears.

### Step 3 — Append the sbatch command to scripts.txt

```bash
echo "sbatch run_paper_plots.sh" >> /home/vasilii/research/trillium/scratch/SHIVAN/analysis/scripts.txt
```

Replace `run_paper_plots.sh` with whatever SLURM script you want to submit.
Common scripts:
- `run_paper_plots.sh` — regenerate all paper figures (~4 min)
- `run_plotter.sh` — full movie frame pipeline (hours)

### Step 4 — Wait for the daemon to pick it up (~10–30 seconds)

Check for a new `slurm-*.out` file:

```bash
ls -lt /home/vasilii/research/trillium/scratch/SHIVAN/analysis/slurm-*.out | head -3
```

A new file with a higher job ID means the job was queued successfully.

---

## Monitoring a job

### Check if still running (file size growing)

```bash
wc -c /home/vasilii/research/trillium/scratch/SHIVAN/analysis/slurm-<JOBID>.out
# wait a few seconds
wc -c /home/vasilii/research/trillium/scratch/SHIVAN/analysis/slurm-<JOBID>.out
```

If the byte count increases, the job is still running.

### Check for completion

```bash
tail -40 /home/vasilii/research/trillium/scratch/SHIVAN/analysis/slurm-<JOBID>.out
```

Look for the `sacct` block appended at the end:
- `ExitCode=0:0` → **success**
- `ExitCode=N:0` (N > 0) → **script error**
- `JobState=TIMEOUT` → job exceeded time limit
- `JobState=CANCELLED` → job was cancelled

The sacct block looks like:
```
sacct -j <JOBID>
JobID    JobName    Account    Elapsed  ...  ExitCode
-------- ---------- ---------- -------- ...  --------
1728491  run_paper+ rrg-rbond+  00:03:56 ...      0:0
```

### Check for errors in output

```bash
grep -i "error\|traceback\|failed\|warning" \
  /home/vasilii/research/trillium/scratch/SHIVAN/analysis/slurm-<JOBID>.out | head -20
```

Note: `Warning: Field Molecular_Fraction not found` warnings are expected and
harmless — the full simulation doesn't have H2 fields.

### Check latest job (no ID known)

```bash
tail -40 "$(ls -t /home/vasilii/research/trillium/scratch/SHIVAN/analysis/slurm-*.out | head -1)"
```

---

## Other useful commands

```bash
# See what's queued in scripts.txt (pending commands)
cat /home/vasilii/research/trillium/scratch/SHIVAN/analysis/scripts.txt

# Check queue_runner daemon log
tail -20 /home/vasilii/research/trillium/scratch/SHIVAN/analysis/queue_runner.log

# Stop the daemon gracefully (run this on the cluster)
touch /scratch/vasissua/SHIVAN/analysis/queue_runner.stop

# Restart the daemon (run on the cluster login node)
cd /scratch/vasissua/SHIVAN/analysis && nohup bash queue_runner.sh &
```

---

## Output locations after a successful paper plots job

- `paper_plots/light/` — light-background PNGs
- `paper_plots/dark/` — dark-background PNGs
- Key files: `combined_density.png`, `combined_Btor.png`, `combined_Bpol.png`,
  `toomre_Q_merged.png`, `Q_heatmap.png`, `phase_combined.png`, etc.
