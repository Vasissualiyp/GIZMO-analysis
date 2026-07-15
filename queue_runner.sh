#!/bin/bash
# queue_runner.sh — Command queue daemon
#
# Watches a file (default: scripts.txt) for commands, runs them one at a time,
# and removes each command after executing it.
#
# Usage:
#   nohup bash queue_runner.sh &               # start in background, log to nohup.out
#   nohup bash queue_runner.sh myqueue.txt &   # custom queue file
#
# To enqueue a command from anywhere:
#   echo "sbatch run_plotter.sh" >> scripts.txt
#
# To stop the daemon:
#   touch queue_runner.stop          # clean stop after current sleep
#   # or: kill <pid from nohup.out>

QUEUE_FILE="${1:-scripts.txt}"
STOP_FLAG="queue_runner.stop"
INTERVAL=10   # seconds between polls

echo "[queue_runner] started — watching '$QUEUE_FILE' (PID $$)"
echo "[queue_runner] stop: touch $STOP_FLAG"

while true; do
    # Clean-stop mechanism
    if [[ -f "$STOP_FLAG" ]]; then
        echo "[queue_runner] stop flag found — exiting"
        rm -f "$STOP_FLAG"
        exit 0
    fi

    # Only act if the queue file exists and has at least one non-blank line
    if [[ -f "$QUEUE_FILE" ]]; then
        # Read the first non-blank line
        CMD=$(grep -m1 '.' "$QUEUE_FILE")

        if [[ -n "$CMD" ]]; then
            echo "[queue_runner] $(date '+%Y-%m-%d %H:%M:%S')  running: $CMD"

            # Remove the first non-blank line atomically
            # sed -i '1{/./d}' would delete line 1 only if non-empty;
            # using a tmp file avoids races on NFS/Lustre.
            TMPFILE=$(mktemp "${QUEUE_FILE}.XXXXXX")
            tail -n +2 "$QUEUE_FILE" > "$TMPFILE" && mv "$TMPFILE" "$QUEUE_FILE"

            # Execute the command (inherit the current environment)
            eval "$CMD"
            echo "[queue_runner] $(date '+%Y-%m-%d %H:%M:%S')  done (exit $?)"
        fi
    fi

    sleep "$INTERVAL"
done
