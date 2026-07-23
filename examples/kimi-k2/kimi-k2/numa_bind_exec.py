#!/usr/bin/env python3
"""Apply CPU/NUMA affinity, then replace this process with the target command."""

import ctypes
import os
import sys


def parse_cpu_list(value):
    cpus = set()
    for item in value.split(","):
        if "-" in item:
            start, end = (int(part) for part in item.split("-", 1))
            cpus.update(range(start, end + 1))
        else:
            cpus.add(int(item))
    return cpus


def bind_memory(numa_node):
    try:
        libnuma = ctypes.CDLL("libnuma.so.1", use_errno=True)
    except OSError as error:
        print(f"[numa-bind] warning: cannot load libnuma.so.1: {error}", file=sys.stderr)
        return False

    libnuma.numa_allocate_nodemask.restype = ctypes.c_void_p
    libnuma.numa_get_membind.restype = ctypes.c_void_p
    libnuma.numa_bitmask_clearall.argtypes = [ctypes.c_void_p]
    libnuma.numa_bitmask_setbit.argtypes = [ctypes.c_void_p, ctypes.c_uint]
    libnuma.numa_bitmask_isbitset.argtypes = [ctypes.c_void_p, ctypes.c_uint]
    libnuma.numa_bitmask_isbitset.restype = ctypes.c_int
    libnuma.numa_set_membind.argtypes = [ctypes.c_void_p]
    libnuma.numa_bitmask_free.argtypes = [ctypes.c_void_p]

    mask = libnuma.numa_allocate_nodemask()
    if not mask:
        raise RuntimeError("libnuma failed to allocate a node mask")
    try:
        libnuma.numa_bitmask_clearall(mask)
        libnuma.numa_bitmask_setbit(mask, numa_node)
        libnuma.numa_set_membind(mask)
    finally:
        libnuma.numa_bitmask_free(mask)

    current = libnuma.numa_get_membind()
    if not current or not libnuma.numa_bitmask_isbitset(current, numa_node):
        raise RuntimeError(f"failed to bind memory policy to NUMA node {numa_node}")
    return True


def main():
    if len(sys.argv) < 4:
        raise SystemExit(f"usage: {sys.argv[0]} NUMA_NODE CPU_LIST COMMAND [ARG ...]")

    numa_node = int(sys.argv[1])
    cpu_list = sys.argv[2]
    os.sched_setaffinity(0, parse_cpu_list(cpu_list))
    memory_bound = bind_memory(numa_node)
    print(
        f"[numa-bind] applied cpus={cpu_list} memory_node={numa_node} "
        f"strict_memory_bind={memory_bound}",
        file=sys.stderr,
        flush=True,
    )
    os.execvp(sys.argv[3], sys.argv[3:])


if __name__ == "__main__":
    main()
