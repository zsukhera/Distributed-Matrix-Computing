#!/usr/bin/env bash
# run_q3.sh — run every Question 3 experiment on the cluster.
# Output goes to the terminal and to results/q3_outputs.txt.
#
# Run on the master node:   ./run_q3.sh          (REPEATS=1 by default)
#
# ./task3 <m> <3.1|3.2|3.2t|3.3|all>
#   3.1  = Reduce+Bcast and Allreduce training
#   3.2  = ring all-reduce training + 1 KB / 1 MB / 64 MB benchmark (50 calls each)
#   3.2t = ring all-reduce training only
#   3.3  = scaling study (MPI_Allreduce, m=1024 fixed)

set -u
cd "$(dirname "$0")"
source ./run_common.sh

LOG=results/q3_outputs.txt
mkdir -p results

{
    header "Q3"

    section "BUILD + DEPLOY"
    run make task3
    run make deploy

    section "CORRECTNESS — m=512, Table 4, p=1,2,3,4 (all three all-reduces)"
    for p in 1 2 3 4; do mpi task3 $p 1 512 3.1;  done
    for p in 1 2 3 4; do mpi task3 $p 1 512 3.2t; done

    section "TASK 3.2 — all-reduce benchmark 1 KB / 1 MB / 64 MB (each value = mean of 50 calls)"
    for p in 2 4 8; do mpi task3 $p 1 512 3.2; done

    for r in $(seq 1 "$REPEATS"); do
        rep "$r"
        section "TASK 3.1 — m=1024, Reduce+Bcast vs Allreduce"
        for p in 1 2 4 8; do mpi task3 $p 1 1024 3.1; done

        section "TASK 3.2 — m=1024, ring all-reduce training"
        for p in 1 2 4 8; do mpi task3 $p 1 1024 3.2t; done

        section "TASK 3.3 — scaling, best all-reduce (MPI_Allreduce), m=1024"
        for p in 1 2 4 8; do mpi task3 $p 1 1024 3.3; done

        section "TASK 3.3 — one process per node + OpenMP"
        mpi task3 3 "$HYBRID_THREADS" 1024 3.3
        [ "$HYBRID_THREADS" -ne 4 ] && mpi task3 3 4 1024 3.3
    done

    echo "ALL DONE $(date)"
} 2>&1 | tee "$LOG"
