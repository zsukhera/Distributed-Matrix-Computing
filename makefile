# makefile for Distributed-Matrix-Computing (Question 1)
# Builds the dense matrix multiplication program with MPI and OpenMP.

CXX      := mpicxx
CXXFLAGS := -O2 -fopenmp -std=c++17 -Wall -Wextra
LDFLAGS  := -fopenmp

SRC := task1.cpp
BIN := task1

# Cluster run settings (override on the command line, e.g. make run NP=3 THREADS=4)
HOSTFILE ?= hosts
NP       ?= 3
THREADS  ?= 2
ARGS     ?=
# --map-by node   : deal processes round-robin across nodes (master, slave1, slave2, master, ...)
# --oversubscribe : allow more processes than slots (NP > 3 on this cluster)
MPIRUN   := mpirun --hostfile $(HOSTFILE) -np $(NP) \
            --map-by node --oversubscribe --bind-to none \
            -x OMP_NUM_THREADS=$(THREADS)

.PHONY: all clean run run-local

all: $(BIN)

$(BIN): $(SRC)
	$(CXX) $(CXXFLAGS) -o $@ $(SRC) $(LDFLAGS)

# Run across the cluster: one MPI process per node, THREADS OpenMP threads each
run: $(BIN)
	$(MPIRUN) ./$(BIN) $(ARGS)

# Run on this machine only (useful for quick tests)
run-local: $(BIN)
	mpirun --oversubscribe -np $(NP) -x OMP_NUM_THREADS=$(THREADS) ./$(BIN) $(ARGS)

clean:
	rm -f $(BIN)
