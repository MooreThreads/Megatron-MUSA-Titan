#!/usr/bin/env bash
# Bind one torchrun child to CPU cores and memory local to its GPU NUMA node.
set -euo pipefail

case "${LOCAL_RANK:-}" in
    0) numa_node=0; cpu_list=0-7,64-71 ;;
    1) numa_node=0; cpu_list=8-15,72-79 ;;
    2) numa_node=0; cpu_list=16-23,80-87 ;;
    3) numa_node=0; cpu_list=24-31,88-95 ;;
    4) numa_node=1; cpu_list=32-39,96-103 ;;
    5) numa_node=1; cpu_list=40-47,104-111 ;;
    6) numa_node=1; cpu_list=48-55,112-119 ;;
    7) numa_node=1; cpu_list=56-63,120-127 ;;
    *) echo "Unsupported LOCAL_RANK: ${LOCAL_RANK:-unset}" >&2; exit 1 ;;
esac

echo "[numa-bind] local_rank=${LOCAL_RANK} numa_node=${numa_node} cpus=${cpu_list}" >&2
if command -v numactl >/dev/null 2>&1; then
    exec numactl --membind="${numa_node}" --physcpubind="${cpu_list}" "$@"
fi

# The cluster image provides libnuma but not the numactl executable. The
# fallback applies both CPU affinity and a strict NUMA memory policy.
exec python3 "$(dirname -- "$0")/numa_bind_exec.py" "${numa_node}" "${cpu_list}" "$@"
