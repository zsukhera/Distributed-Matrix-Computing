#!/usr/bin/env bash
# run_q2.sh — run every Question 2 experiment on the cluster.
# Output goes to the terminal and to results/q2_outputs.txt.
#
# Run on the master node:   ./run_q2.sh
#
# Cluster: master, slave1, slave2 with slots=1 each, so p > 3 uses
# --oversubscribe --map-by node (ranks go master, slave1, slave2, master, ...).
# HYBRID_THREADS sets the OpenMP threads for the Task 2.4 hybrid runs
# (default: the number of cores on this node).

set -u
cd "$(dirname "$0")"

LOG=results/q2_outputs.txt
HYBRID_THREADS=${HYBRID_THREADS:-$(nproc)}
N_SMALL=100000
N_LARGE=1000000

mkdir -p results

# run <command...> : print the command, run it, print exit code and wall time
run() {
    echo "================================================================"
    echo "\$ $*"
    echo "================================================================"
    local t0=$(date +%s)
    bash -c "$*"
    local rc=$?
    echo "[exit=$rc  wall=$(( $(date +%s) - t0 ))s]"
    echo
}

# mpi <p> <threads> <task2 args...>
mpi() {
    local p=$1 t=$2
    shift 2
    local extra=""
    [ "$p" -gt 3 ] && extra="--oversubscribe --map-by node "
    run "OMP_NUM_THREADS=$t mpirun ${extra}-n $p --hostfile hosts ./task2 $*"
}

section() { echo "######## $* ########"; }

{
    echo "Q2 run log — $(date)  host=$(hostname)  commit=$(git rev-parse --short HEAD 2>/dev/null)"
    echo "Cluster: master, slave1, slave2 (slots=1 each). p > 3 uses --oversubscribe --map-by node."
    echo

    section "BUILD + DEPLOY"
    run make task2
    run make deploy

    section "CORRECTNESS — n=$N_SMALL, every version (p=3 and 7 do not divide n)"
    for p in 1 2 3 7 8; do mpi $p 1 $N_SMALL all; done

    section "TASK 2.1 — serial vs OpenMP static vs dynamic,64 (one node)"
    mpi 1 1 $N_SMALL 2.1
    for t in 1 2 4; do mpi 1 $t $N_LARGE 2.1; done

    section "TASK 2.2 — Version A (Allgatherv)"
    for p in 1 2 4 8; do mpi $p 1 $N_LARGE 2.2; done

    section "TASK 2.3 — Version B (halo exchange, with / without overlap)"
    for p in 1 2 4 8; do mpi $p 1 $N_LARGE 2.3; done

    section "TASK 2.4 — equal rows vs equal links (one process per core)"
    for p in 4 8; do mpi $p 1 $N_LARGE 2.4; done

    section "TASK 2.4 — balanced Version B, one process per node, $HYBRID_THREADS thread(s) each"
    mpi 3 "$HYBRID_THREADS" $N_LARGE 2.4
    [ "$HYBRID_THREADS" -ne 4 ] && mpi 3 4 $N_LARGE 2.4

    echo "ALL DONE $(date)"
} 2>&1 | tee "$LOG"
