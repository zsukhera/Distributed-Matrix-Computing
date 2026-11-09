#!/usr/bin/env bash
# run_q2.sh — run every Question 2 experiment on the cluster.
# Output goes to the terminal and to results/q2_outputs.txt.
#
# Run on the master node:   ./run_q2.sh          (REPEATS=1 by default)
#
# ./task2 <n> <2.1|2.2|2.3|2.4|all>

set -u
cd "$(dirname "$0")"
source ./run_common.sh

LOG=results/q2_outputs.txt
N_SMALL=100000
N_LARGE=1000000
mkdir -p results

{
    header "Q2"

    section "BUILD + DEPLOY"
    run make task2
    run make deploy

    section "CORRECTNESS — n=$N_SMALL, every version (p=3 and 7 do not divide n)"
    for p in 1 2 3 7 8; do mpi task2 $p 1 $N_SMALL all; done

    for r in $(seq 1 "$REPEATS"); do
        rep "$r"
        section "TASK 2.1 — serial vs OpenMP static vs dynamic,64 (one node)"
        mpi task2 1 1 $N_SMALL 2.1
        for t in 1 2 4; do mpi task2 1 $t $N_LARGE 2.1; done

        section "TASK 2.2 — Version A (Allgatherv)"
        for p in 1 2 4 8; do mpi task2 $p 1 $N_LARGE 2.2; done

        section "TASK 2.3 — Version B (halo exchange, with / without overlap)"
        for p in 1 2 4 8; do mpi task2 $p 1 $N_LARGE 2.3; done

        section "TASK 2.4 — equal rows vs equal links (one process per core)"
        for p in 4 8; do mpi task2 $p 1 $N_LARGE 2.4; done

        section "TASK 2.4 — balanced Version B, one process per node + OpenMP"
        mpi task2 3 "$HYBRID_THREADS" $N_LARGE 2.4
        [ "$HYBRID_THREADS" -ne 4 ] && mpi task2 3 4 $N_LARGE 2.4
    done

    echo "ALL DONE $(date)"
} 2>&1 | tee "$LOG"
