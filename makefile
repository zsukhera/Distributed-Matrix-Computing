# Makefile — Distributed Matrix Computing (Q1, Q2, Q3)
#
# Usage:
#   make           build all three tasks
#   make task1     build Question 1 only
#   make task2     build Question 2 only
#   make task3     build Question 3 only
#   make test1     run all Q1 tests (N=8,1000,2000,4000 tested inside binary)
#   make test2     run all Q2 tests
#   make test3     run all Q3 tests
#   make clean     remove all binaries

CXX      := mpicxx
CXXFLAGS := -O2 -fopenmp -std=c++17 -Wall -Wextra
LDFLAGS  := -fopenmp

SLAVES   := slave1 slave2
DEPLOY_DIR := /mirror/Distributed-Matrix-Computing

.PHONY: all clean deploy task1 task2 task3 test1 test2 test3

all: task1 task2 task3

# -----------------------------------------------------------------------
# Question 1 — Dense Matrix Multiplication
# Binary tests N = 8, 1000, 2000, 4000 internally.
# -----------------------------------------------------------------------
task1: task1.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

test1: task1
	@echo ">>> Q1 | p=1, threads=1"
	OMP_NUM_THREADS=1 mpirun -n 1 ./task1

	@echo ">>> Q1 | p=1, threads=2"
	OMP_NUM_THREADS=2 mpirun -n 1 ./task1

	@echo ">>> Q1 | p=1, threads=4"
	OMP_NUM_THREADS=4 mpirun -n 1 ./task1

	@echo ">>> Q1 | p=2, threads=1"
	OMP_NUM_THREADS=1 mpirun -n 2 ./task1

	@echo ">>> Q1 | p=4, threads=1"
	OMP_NUM_THREADS=1 mpirun -n 4 ./task1

	@echo ">>> Q1 | p=8, threads=1"
	OMP_NUM_THREADS=1 mpirun -n 8 ./task1

	@echo ">>> Q1 | p=2, threads=4 (hybrid)"
	OMP_NUM_THREADS=4 mpirun -n 2 ./task1

# -----------------------------------------------------------------------
# Question 2 — Sparse Matrix Multiplication (Social Network)
# -----------------------------------------------------------------------
task2: task2.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

test2: task2
	@echo ">>> Q2 | p=1, threads=1"
	OMP_NUM_THREADS=1 mpirun -n 1 ./task2

	@echo ">>> Q2 | p=4, threads=1"
	OMP_NUM_THREADS=1 mpirun -n 4 ./task2

	@echo ">>> Q2 | p=8, threads=1"
	OMP_NUM_THREADS=1 mpirun -n 8 ./task2

# -----------------------------------------------------------------------
# Question 3 — Adding Up Partial Results (Parallel Training)
# -----------------------------------------------------------------------
task3: task3.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

test3: task3
	@echo ">>> Q3 | p=1, threads=1"
	OMP_NUM_THREADS=1 mpirun -n 1 ./task3

	@echo ">>> Q3 | p=4, threads=1"
	OMP_NUM_THREADS=1 mpirun -n 4 ./task3

	@echo ">>> Q3 | p=8, threads=1"
	OMP_NUM_THREADS=1 mpirun -n 8 ./task3

# -----------------------------------------------------------------------
deploy: task1
	@for s in $(SLAVES); do \
		echo "Deploying to $$s..."; \
		ssh $$s "mkdir -p $(DEPLOY_DIR)" && \
		scp task1 $$s:$(DEPLOY_DIR)/.task1.tmp && \
		ssh $$s "mv -f $(DEPLOY_DIR)/.task1.tmp $(DEPLOY_DIR)/task1" || exit 1; \
	done

# -----------------------------------------------------------------------
clean:
	rm -f task1 task2 task3
