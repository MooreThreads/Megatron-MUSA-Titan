#!/bin/bash

# Default hostfile path
HOSTFILE=./hostfile

# Check if hostfile path is provided as argument
if [ $# -ge 1 ]; then
  HOSTFILE="$1"
fi

# Check if hostfile exists
if [ ! -f "$HOSTFILE" ]; then
  echo "Error: Hostfile $HOSTFILE not found!"
  exit 1
fi

# Count number of non-comment, non-empty lines in hostfile
NUM_NODES=$(grep -v '^#\|^$' "$HOSTFILE" | wc -l)
echo "NUM_NODES: $NUM_NODES"

# Extract hostnames from hostfile (ignoring comments and empty lines)
hostlist=$(grep -v '^#\|^$' "$HOSTFILE" | awk '{print $1}' | xargs)

# Kill torchrun processes on all hosts in parallel
for host in ${hostlist[@]}; do
    (
        # Try killing torchrun processes with different possible paths
        ssh -n $host "pkill -f '/opt/conda/envs/py310/bin/torchrun'" || true
        ssh -n $host "pkill -f '/usr/local/bin/torchrun'" || true
        ssh -n $host "pkill -f torchrun" || true
        ssh -n $host "pkill -f /usr/bin/python" || true
        ssh -n $host "pkill -f /opt/conda/envs/py310/bin/python" || true
        ssh -n $host "pkill -f /usr/local/musa/mccl_test/all_reduce_perf" || true
        echo "$host: Successfully killed torchrun processes"
    ) &  # Run each host's commands in a background subshell for parallelism
done

# Wait for all background processes to complete
wait

echo "Completed processing all hosts"