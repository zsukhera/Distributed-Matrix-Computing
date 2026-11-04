# Distributed-Matrix-Computing

Implementation of distributed and parallel computing techniques using MPI and OpenMP, covering dense and sparse matrix multiplication, neural network computations, social network recommendations, and distributed gradient aggregation using collective communication and ring-based messaging.

All programs are written in C++ using only MPI and OpenMP, and are built with a single Makefile.

---

## Contents

| Question | Problem | Techniques |
|---|---|---|
| **Q1** | Dense matrix multiplication: one neural-network layer, `C = A × B` | OpenMP loop parallelism, `MPI_Scatterv` / `MPI_Bcast` / `MPI_Gatherv`, blocking and non-blocking ring, hybrid MPI+OpenMP |
| **Q2** | Sparse matrix multiplication: "average of my friends" on a social network | CSR storage, `schedule(static)` vs `schedule(dynamic)`, `MPI_Allgatherv`, halo exchange with overlap, load balancing |
| **Q3** | Adding partial gradients in parallel training | `MPI_Reduce` + `MPI_Bcast`, `MPI_Allreduce`, custom ring all-reduce, scaling study |

---

## Repository Structure

```
Distributed-Matrix-Computing/
├── Makefile          # builds all three questions
├── run.sh            # runs the experiments
├── README.md
├── q1/               # dense matrix multiplication
├── q2/               # sparse CSR multiplication
└── q3/               # gradient all-reduce and training loop
```

---

## Requirements

- A C++ compiler with OpenMP support (e.g. `g++` 9 or newer)
- An MPI implementation (Open MPI or MPICH) providing `mpicxx` and `mpirun`
- An MPI cluster of at least two nodes for the distributed experiments

Check your setup:

```bash
mpicxx --version
mpirun --version
```

---

## Building

```bash
make            # build everything
make q1         # build a single question
make clean      # remove binaries
```

All programs are compiled with `-O2 -fopenmp`.

---

## Running

### Q1: Dense Matrix Multiplication

| Version | Description |
|---|---|
| `serial` | Single thread, i-k-j loop order |
| `omp` | OpenMP parallel over rows of C |
| `A` | Rank 0 creates A and B; Scatterv rows of A, Bcast all of B, Gatherv C |
| `B` | Ring of B blocks with blocking `MPI_Send`/`MPI_Recv` (even/odd ordering to avoid deadlock) |
| `C` | Ring with `MPI_Isend`/`MPI_Irecv`/`MPI_Waitall` and two buffers, overlapping communication with computation; OpenMP inside each process |

```bash
OMP_NUM_THREADS=4 ./q1/matmul omp 2000
mpirun -np 8 ./q1/matmul A 4000
mpirun -np 8 ./q1/matmul B 4000
mpirun -np 8 ./q1/matmul C 4000
```

### Q2: Sparse Matrix Multiplication (CSR)

| Version | Description |
|---|---|
| `omp` | CSR step parallelised with OpenMP (static or dynamic schedule) |
| `A` | Each process owns a block of users; full H shared with `MPI_Allgatherv` every step |
| `B` | Only 128 halo rows exchanged with each neighbour; interior rows computed while halos are in flight |
| `balanced` | Block boundaries chosen so each process gets about the same number of links; hybrid MPI+OpenMP |

```bash
OMP_NUM_THREADS=8 ./q2/sparse omp 1000000
mpirun -np 8 ./q2/sparse B 1000000
```

### Q3: Gradient All-Reduce

| Version | Description |
|---|---|
| `reduce_bcast` | `MPI_Reduce` to rank 0, then `MPI_Bcast` |
| `allreduce` | Single `MPI_Allreduce` |
| `ring` | Custom ring all-reduce (add phase + share phase), works for any `p` and `n` |

```bash
mpirun -np 4 ./q3/train ring 1024
```

### Hybrid MPI+OpenMP Runs

Run one process per node with OpenMP threads inside it. Disable core binding so the threads aren't confined to a single core:

```bash
export OMP_NUM_THREADS=<cores per node>
mpirun -np <nodes> --hostfile hosts --map-by ppr:1:node --bind-to none ./q1/matmul C 4000
```

---

## Verifying Results

Every version of a question must print the same values regardless of the number of processes or threads.

- **Q1:** `N = 8` prints the whole matrix C (compare with Table 1(a) in the assignment). Larger runs print C(0,0), C(0,1), C(1,0), C(N/2,N/2) and C(N−1,N−1), each printed by the process that owns that row.
- **Q2:** after T = 10 steps, prints H(0,0), H(12345,7) and H(n−1,15) to 10 decimal places.
- **Q3:** prints three entries of the summed gradients after step 0 and three weights after the last step (`%.16g`).

Correctness is tested with process counts that do not divide the problem size (e.g. `p = 3, 5, 7`).

---

## Timing Method

All MPI programs follow the same pattern: an `MPI_Barrier`, then `MPI_Wtime` around computation and communication only (no data creation, checking or printing). Each process's time is combined with `MPI_Reduce(..., MPI_MAX, ...)` so the slowest process is reported. Computation time and communication/waiting time are measured separately for each process.

Reported times are the best of five runs.

---

## Author

Zain Sukhera 
October 2026 
