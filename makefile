# makefile for Distributed-Matrix-Computing (Question 1)
# Builds the dense matrix multiplication program with MPI and OpenMP.

CXX      := mpicxx
CXXFLAGS := -O2 -fopenmp -std=c++17 -Wall -Wextra
LDFLAGS  := -fopenmp

SRC := task1.cpp
BIN := task1

.PHONY: all clean

all: $(BIN)

$(BIN): $(SRC)
	$(CXX) $(CXXFLAGS) -o $@ $(SRC) $(LDFLAGS)

clean:
	rm -f $(BIN)