#!/usr/bin/env bash
# run_common.sh — helpers shared by run_q1.sh, run_q2.sh and run_q3.sh.
# Not meant to be run directly; each run_qN.sh sources it.
#
# Cluster: master, slave1, slave2 (slots=1 each). p > 3 uses
# --oversubscribe --map-by node (ranks go master, slave1, slave2, master, ...).
#
# Every mpirun forwards OMP_NUM_THREADS with -x: Open MPI does NOT pass the
# environment to remote nodes by itself, so without -x the ranks on slave1 and
# slave2 would silently run with their default thread count.

REPEATS=${REPEATS:-1}                    # timing runs per configuration (set REPEATS=5 for best-of-five)
HYBRID_THREADS=${HYBRID_THREADS:-$(nproc)} # cores per node, for "one process per node" runs

# run <command...> : print the command, run it, print exit code and wall time.
# The "plm:ssh: Warning: setpgid" lines are harmless launcher noise and are dropped.
run() {
    echo "================================================================"
    echo "\$ $*"
    echo "================================================================"
    local t0=$(date +%s)
    bash -c "$*" 2>&1 | grep --line-buffered -v 'plm:ssh: Warning: setpgid'
    local rc=${PIPESTATUS[0]}
    echo "[exit=$rc  wall=$(( $(date +%s) - t0 ))s]"
    echo
}

# mpi <binary> <p> <threads> <args...>
mpi() {
    local bin=$1 p=$2 t=$3
    shift 3
    local extra=""
    [ "$p" -gt 3 ] && extra="--oversubscribe --map-by node "
    run "OMP_NUM_THREADS=$t mpirun -x OMP_NUM_THREADS ${extra}-n $p --hostfile hosts ./$bin $*"
}

section() { echo "######## $* ########"; }
rep()     { echo "######## REPEAT $1/$REPEATS ########"; }

header() {
    echo "$1 run log — $(date)  host=$(hostname)  commit=$(git rev-parse --short HEAD 2>/dev/null)$(git diff --quiet 2>/dev/null || echo ' + uncommitted changes')"
    echo "Cluster: master, slave1, slave2 (slots=1 each). p > 3 uses --oversubscribe --map-by node."
    echo "REPEATS=$REPEATS (timing runs; report the best)  HYBRID_THREADS=$HYBRID_THREADS"
    echo
}
