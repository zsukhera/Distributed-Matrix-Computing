#!/usr/bin/env bash
# run_q1.sh — run every Question 1 experiment on the cluster.
# Output goes to the terminal and to results/q1_outputs.txt.
#
# Run on the master node:   ./run_q1.sh          (REPEATS=1 by default)
#                           REPEATS=5 ./run_q1.sh (best of five)
#
# ./task1 <N> <1.1|1.2|1.3|1.4>

set -u
cd "$(dirname "$0")"
source ./run_common.sh

LOG=results/q1_outputs.txt
mkdir -p results

{
    header "Q1"

    section "BUILD + DEPLOY"
    run make task1
    run make deploy

    section "CORRECTNESS — N=8 full matrix (Table 1a), p=1,2,3,8"
    mpi task1 1 1 8 1.1
    for v in 1.2 1.3 1.4; do
        for p in 1 2 3 8; do mpi task1 $p 1 8 $v; done
    done

    section "TASK 1.1 — N=4000 serial i-j-k / i-k-j / OpenMP (single run, for Table 2)"
    mpi task1 1 1 4000 1.1
    # (4-thread N=4000 run omitted: 1 core per node, it only repeats the 8-min i-j-k pass)

    for r in $(seq 1 "$REPEATS"); do
        rep "$r"
        section "TASK 1.1 — i-j-k vs i-k-j vs OpenMP, one node"
        for N in 1000 2000; do
            for t in 1 2 4; do mpi task1 1 $t $N 1.1; done
        done

        section "TASK 1.2 — Version A (Scatterv / Bcast / Gatherv)"
        for N in 2000 4000; do for p in 1 2 4 8; do mpi task1 $p 1 $N 1.2; done; done

        section "TASK 1.3 — Version B (blocking ring)"
        for N in 2000 4000; do for p in 1 2 4 8; do mpi task1 $p 1 $N 1.3; done; done

        section "TASK 1.4 — Version C (non-blocking ring), one process per core"
        for N in 2000 4000; do for p in 1 2 4 8; do mpi task1 $p 1 $N 1.4; done; done

        section "TASK 1.4 — Version C hybrid, one process per node + OpenMP"
        mpi task1 3 "$HYBRID_THREADS" 4000 1.4
        mpi task1 2 4 4000 1.4
        mpi task1 3 4 4000 1.4
    done

    echo "ALL DONE $(date)"
} 2>&1 | tee "$LOG"
