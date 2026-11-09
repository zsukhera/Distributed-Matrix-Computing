#!/usr/bin/env bash
# run_all.sh — run Q1, Q2 and Q3 one after the other, then summarise.
#   ./run_all.sh            (one run per configuration)
#   REPEATS=5 ./run_all.sh  (best of five)
set -u
cd "$(dirname "$0")"
make all && make deploy || exit 1
./run_q1.sh > /dev/null
./run_q2.sh > /dev/null
./run_q3.sh > /dev/null
python3 summarize_results.py
