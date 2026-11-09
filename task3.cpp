/*
 * Task 3 — Adding Up Partial Results in Parallel Training
 *
 * L=8 layers, each with an m×m weight matrix W_l.
 * Batch = 512 examples divided among processes.
 * Per step:
 *   1. compute  G_l = X_l x E_l  over own examples  (OpenMP i-k-j)
 *   2. all-reduce  G_l across all processes
 *   3. update  W_l = W_l - G_l / 65536
 *
 * Input formulas (s = training step, l = layer index):
 *   X_l(i, g) = ((i + 3g + 5l + s) mod 9) - 4
 *   E_l(g, j) = ((2g + j + l + 3s) mod 7) - 3
 *
 *   3.1  (a) MPI_Reduce + MPI_Bcast   (b) MPI_Allreduce
 *   3.2  Custom ring_allreduce with MPI_Isend / MPI_Irecv / MPI_Waitall
 *   3.3  Scaling study
 *
 * Verification (Table 4):
 *   (a) Summed gradients after all-reduce of step 0:
 *       m=512:  G0(1,2)=33   G0(m-1,m-1)=24   G7(m-1,m-1)=27
 *       m=1024: G0(1,2)=33   G0(m-1,m-1)=5    G7(m-1,m-1)=-20
 *   (b) Weights after S=5 steps (%.16g):
 *       m=512:  W0(0,0)=0.0001220703125  W0(1,2)=-0.000152587890625  W7(m-1,m-1)=-0.00079345703125
 *       m=1024: W0(0,0)=0.0001220703125  W0(1,2)=-0.000152587890625  W7(m-1,m-1)=0.0008697509765625
 *
 * Build:  make task3
 * Run:    mpirun -n <p> --hostfile hosts ./task3 [m] [3.1|3.2|3.2t|3.3|all]
 *         (3.2t = ring all-reduce training only, without the 1KB/1MB/64MB benchmark)
 */

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <mpi.h>
#include <omp.h>

using namespace std;

static const int L       = 8;      /* number of layers                  */
static const int BATCH   = 512;    /* total examples per step            */
static const int S       = 5;      /* training steps                     */
static const double SCALE = 65536.0; /* weight update divisor            */

static const int DEFAULT_M = 512;

/* -----------------------------------------------------------------------
 * Data initialisation
 * X_l(i, g) = ((i + 3g + 5l + s) mod 9) - 4   input matrix for layer l
 * E_l(g, j) = ((2g + j + l + 3s) mod 7) - 3   error matrix for layer l
 *
 * g here is the GLOBAL example index (0..511), not the local one.
 * Each process fills only its own slice of examples.
 * --------------------------------------------------------------------- */

static inline double val_X(int i, int g, int l, int s)
{
    return (double)(((i + 3*g + 5*l + s) % 9) - 4);
}

static inline double val_E(int g, int j, int l, int s)
{
    return (double)(((2*g + j + l + 3*s) % 7) - 3);
}

/* Fill X_l for this process's examples [g_start, g_start+my_examples).
   X is stored row-major: X[i * my_examples + k] where k is local example index. */
static void fill_X(double* X, int m, int g_start, int my_examples, int l, int s)
{
    for (int i = 0; i < m; i++)
        for (int k = 0; k < my_examples; k++)
            X[i * my_examples + k] = val_X(i, g_start + k, l, s);
}

/* Fill E_l for this process's examples [g_start, g_start+my_examples).
   E is stored row-major: E[k * m + j] where k is local example index. */
static void fill_E(double* E, int m, int g_start, int my_examples, int l, int s)
{
    for (int k = 0; k < my_examples; k++)
        for (int j = 0; j < m; j++)
            E[k * m + j] = val_E(g_start + k, j, l, s);
}

/* -----------------------------------------------------------------------
 * Matrix multiply: G += X x E  (i-k-j, OpenMP over rows of G)
 * X: m x my_examples,  E: my_examples x m,  G: m x m  (all row-major)
 * Note: accumulates INTO G (+=), so caller must zero G before the first use.
 * --------------------------------------------------------------------- */
static void multiply_add(const double* X, const double* E, double* G,
                         int m, int my_examples)
{
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < m; i++)
        for (int k = 0; k < my_examples; k++) {
            double x_ik = X[i * my_examples + k];
            for (int j = 0; j < m; j++)
                G[i * m + j] += x_ik * E[k * m + j];
        }
}

/* -----------------------------------------------------------------------
 * Example distribution: process r owns examples [ex_start(r), ex_start(r)+ex_rows(r))
 * Same even-split rule as Q1/Q2: first (BATCH % p) ranks get one extra example.
 * --------------------------------------------------------------------- */
static int ex_rows(int r, int p)  { return BATCH / p + (r < BATCH % p ? 1 : 0); }
static int ex_start(int r, int p) { return r * (BATCH / p) + (r < BATCH % p ? r : BATCH % p); }

/* -----------------------------------------------------------------------
 * All-reduce versions
 * All three operate on a flat array v[0..n) of doubles in-place,
 * replacing each element with the sum across all processes.
 * --------------------------------------------------------------------- */

/* Task 3.1a: rank 0 collects the sum with MPI_Reduce, then broadcasts back. */
//array v will be reduced in place 
static void allreduce_reduce_bcast(double* v, long n, MPI_Comm comm)
{
    int rank;
    MPI_Comm_rank(comm, &rank);
    // Rank 0 receives the sum; all others send MPI_IN_PLACE means rank 0 uses v as both input and output.
    if (rank == 0)
        MPI_Reduce(MPI_IN_PLACE, v, (int)n, MPI_DOUBLE, MPI_SUM, 0, comm);
    else
        MPI_Reduce(v, nullptr,   (int)n, MPI_DOUBLE, MPI_SUM, 0, comm);
    // Now only rank 0 has the sum — broadcast it back to everyone.
    MPI_Bcast(v, (int)n, MPI_DOUBLE, 0, comm);
}

/* Task 3.1b: single built-in collective — does reduce + broadcast in one call. */
static void allreduce_builtin(double* v, long n, MPI_Comm comm)
{
    // MPI_IN_PLACE: each process uses v as both send and receive buffer.
    MPI_Allreduce(MPI_IN_PLACE, v, (int)n, MPI_DOUBLE, MPI_SUM, comm);
}

/* Task 3.2: custom ring all-reduce.
 *
 * Cuts v[0..n) into p chunks (last chunk absorbs any remainder).
 *
 * Phase 1 (add), p-1 steps:
 *   step s: send chunk (r-s) mod p to RIGHT, recv chunk (r-s-1) mod p from LEFT,
 *            add received values into own copy.
 *   After p-1 steps, process r holds the full sum of chunk (r+1) mod p.
 *
 * Phase 2 (share), p-1 steps:
 *   step s: send chunk (r+1-s) mod p to RIGHT, recv chunk (r-s) mod p from LEFT,
 *            copy received values over own copy.
 *   After p-1 steps, every process holds the full sum of all chunks.
 *
 * Uses MPI_Isend + MPI_Irecv + MPI_Waitall in every step. */
static void ring_allreduce(double* v, long n, MPI_Comm comm)
{
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    if (size == 1) return;

    int left  = (rank - 1 + size) % size;
    int right = (rank + 1) % size;

    // Compute chunk start offsets and lengths.
    // Chunks 0..(rem-1) get one extra element to absorb the remainder.
    long base = n / size, rem = n % size;
    vector<long> cstart(size), clen(size);
    for (int c = 0; c < size; c++) {
        clen[c]   = base + (c < rem ? 1 : 0);
        cstart[c] = c * base + (c < rem ? c : rem);
    }

    // Temporary receive buffer sized for the largest chunk.
    long max_chunk = base + (rem > 0 ? 1 : 0);
    vector<double> buf(max_chunk);

    // Phase 1 (add): p-1 steps.
    for (int s = 0; s < size - 1; s++) {
        int sc = (rank - s + size) % size;         /* chunk to send */
        int rc = (rank - s - 1 + size) % size;     /* chunk to receive */
        MPI_Request reqs[2];
        MPI_Irecv(buf.data(),            (int)clen[rc], MPI_DOUBLE, left,  s, comm, &reqs[0]);
        MPI_Isend(v + cstart[sc], (int)clen[sc], MPI_DOUBLE, right, s, comm, &reqs[1]);
        MPI_Waitall(2, reqs, MPI_STATUSES_IGNORE);
        // Add the received partial sum into our local copy of this chunk.
        double* dst = v + cstart[rc];
        for (long i = 0; i < clen[rc]; i++) dst[i] += buf[i];
    }

    // Phase 2 (share): p-1 steps.
    // Use tags offset by size to avoid any collision with Phase 1 tags.
    for (int s = 0; s < size - 1; s++) {
        int sc = (rank + 1 - s + size) % size;    /* chunk to send (fully summed) */
        int rc = (rank - s + size) % size;         /* chunk to receive             */
        MPI_Request reqs[2];
        MPI_Irecv(buf.data(),            (int)clen[rc], MPI_DOUBLE, left,  size + s, comm, &reqs[0]);
        MPI_Isend(v + cstart[sc], (int)clen[sc], MPI_DOUBLE, right, size + s, comm, &reqs[1]);
        MPI_Waitall(2, reqs, MPI_STATUSES_IGNORE);
        // Overwrite our copy with the fully-summed chunk from the left neighbour.
        memcpy(v + cstart[rc], buf.data(), clen[rc] * sizeof(double));
    }
}

/* -----------------------------------------------------------------------
 * Timing struct
 * --------------------------------------------------------------------- */
struct StepStats {
    double compute   = 0.0;
    double allreduce = 0.0;
};

struct TimingResult {
    double total;        /* wall time for S steps (max across procs) */
    double compute;      /* compute time per step  (max across procs) */
    double allreduce;    /* allreduce time per step (max across procs) */
};

/* -----------------------------------------------------------------------
 * Training loop — returns timing data; prints Table 4(a) and 4(b).
 * all_procs_print: if true every rank prints Table 4(b) (used by Task 3.3)
 * --------------------------------------------------------------------- */
using AllreduceFn = void(*)(double*, long, MPI_Comm);

static TimingResult train(int m, int rank, int size, AllreduceFn allreduce_fn,
                          const char* title, bool all_procs_print = false)
{
    int my_examples = ex_rows(rank, size);
    int g_start     = ex_start(rank, size);
    long grad_n     = (long)L * m * m;

    vector<vector<double>> W(L, vector<double>((size_t)m * m, 0.0));
    vector<vector<double>> G(L, vector<double>((size_t)m * m));
    vector<double> X((size_t)m * my_examples);
    vector<double> E((size_t)my_examples * m);
    vector<double> G_flat((size_t)grad_n);

    StepStats acc;
    bool printed_step0 = false;

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    for (int s = 0; s < S; s++) {

        /* compute: G_l = X_l x E_l over own examples */
        double tc = MPI_Wtime();
        for (int l = 0; l < L; l++) {
            fill(G[l].begin(), G[l].end(), 0.0);
            fill_X(X.data(), m, g_start, my_examples, l, s);
            fill_E(E.data(), m, g_start, my_examples, l, s);
            multiply_add(X.data(), E.data(), G[l].data(), m, my_examples);
        }
        acc.compute += MPI_Wtime() - tc;

        for (int l = 0; l < L; l++)
            memcpy(G_flat.data() + (size_t)l * m * m,
                   G[l].data(), (size_t)m * m * sizeof(double));

        /* all-reduce: sum G across all processes */
        tc = MPI_Wtime();
        allreduce_fn(G_flat.data(), grad_n, MPI_COMM_WORLD);
        acc.allreduce += MPI_Wtime() - tc;

        for (int l = 0; l < L; l++)
            memcpy(G[l].data(), G_flat.data() + (size_t)l * m * m,
                   (size_t)m * m * sizeof(double));

        /* Table 4(a): print summed gradients after step 0 all-reduce */
        if (s == 0 && !printed_step0) {
            if (rank == 0) {
                printf("\n  Table 4(a) m=%d: G0(1,2)=%g  G0(%d,%d)=%g  G7(%d,%d)=%g\n",
                       m, G[0][1 * m + 2],
                       m-1, m-1, G[0][(m-1)*m + (m-1)],
                       m-1, m-1, G[7][(m-1)*m + (m-1)]);
            }
            printed_step0 = true;
        }

        /* update: W_l = W_l - G_l / 65536 */
        for (int l = 0; l < L; l++)
            for (long idx = 0; idx < (long)m * m; idx++)
                W[l][idx] -= G[l][idx] / SCALE;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t_local = MPI_Wtime() - t0;

    double t_max, tc_max, ta_max;
    MPI_Reduce(&t_local,      &t_max,  1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&acc.compute,  &tc_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&acc.allreduce,&ta_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("\n--- %s: m=%d, p=%d, threads=%d ---\n",
               title, m, size, omp_get_max_threads());
        printf("  Total:     %.6f s\n", t_max);
        printf("  Compute:   %.6f s/step  (%.1f%%)\n",
               tc_max / S, 100.0 * tc_max / t_max);
        printf("  Allreduce: %.6f s/step  (%.1f%%)\n",
               ta_max / S, 100.0 * ta_max / t_max);
        fflush(stdout);
    }

    /* Every process's own compute / all-reduce time per step, collected on
       rank 0 with MPI_Gather (one line per process). */
    {
        double mine[2] = {acc.compute / S, acc.allreduce / S};
        vector<double> all(rank == 0 ? 2 * (size_t)size : 0);
        MPI_Gather(mine, 2, MPI_DOUBLE, all.data(), 2, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            printf("  Per-process times per step:\n  rank   compute(s)  allreduce(s)\n");
            for (int r = 0; r < size; r++)
                printf("  %4d   %10.6f    %10.6f\n", r, all[2 * r], all[2 * r + 1]);
            fflush(stdout);
        }
    }

    /* Table 4(b): weights after S steps.
     * all_procs_print=true (Task 3.3) means every rank prints to verify
     * that the all-reduce left all processes with identical weights. */
    if (all_procs_print) {
        MPI_Barrier(MPI_COMM_WORLD);
        for (int r = 0; r < size; r++) {
            if (rank == r) {
                printf("  [rank %d] Table 4(b) m=%d: "
                       "W0(0,0)=%.16g  W0(1,2)=%.16g  W7(%d,%d)=%.16g\n",
                       rank, m,
                       W[0][0 * m + 0],
                       W[0][1 * m + 2],
                       m-1, m-1, W[7][(m-1)*m + (m-1)]);
                fflush(stdout);
            }
            MPI_Barrier(MPI_COMM_WORLD);
        }
    } else if (rank == 0) {
        printf("\n  Table 4(b) m=%d: W0(0,0)=%.16g  W0(1,2)=%.16g  W7(%d,%d)=%.16g\n",
               m,
               W[0][0 * m + 0],
               W[0][1 * m + 2],
               m-1, m-1, W[7][(m-1)*m + (m-1)]);
        fflush(stdout);
    }

    return {t_max, tc_max / S, ta_max / S};
}

/* -----------------------------------------------------------------------
 * Task 3.2: standalone all-reduce benchmark
 * Measures all three methods on buffers of 1 KB, 1 MB and 64 MB.
 * Requires p >= 2. Run with p = 2, 4, 8 as asked by the PDF.
 * --------------------------------------------------------------------- */
static void benchmark_allreduce(int rank, int size)
{
    if (size < 2) {
        if (rank == 0) printf("\n  (skip benchmark: needs p >= 2)\n");
        return;
    }

    struct Cfg { long n; const char* label; };
    /* 1 KB = 128 doubles, 1 MB = 131072, 64 MB = 8388608 */
    Cfg cfgs[] = {
        {1024L / 8,              "1 KB"},
        {1024L * 1024 / 8,       "1 MB"},
        {64L * 1024 * 1024 / 8,  "64 MB"},
    };

    struct Method { const char* name; AllreduceFn fn; int reps; };
    Method methods[] = {
        {"Reduce+Bcast", allreduce_reduce_bcast, 50},
        {"Allreduce",    allreduce_builtin,       50},
        {"Ring",         ring_allreduce,          50},
    };

    if (rank == 0) {
        printf("\n--- Task 3.2 all-reduce benchmark: p=%d ---\n", size);
        printf("  %-16s %14s %14s %14s\n", "Method", "1 KB (s)", "1 MB (s)", "64 MB (s)");
    }

    for (auto& meth : methods) {
        if (rank == 0) printf("  %-16s", meth.name);
        for (auto& cfg : cfgs) {
            vector<double> v(cfg.n);

            /* warmup */
            for (long i = 0; i < cfg.n; i++) v[i] = rank + 1.0;
            meth.fn(v.data(), cfg.n, MPI_COMM_WORLD);

            MPI_Barrier(MPI_COMM_WORLD);
            double t0 = MPI_Wtime();
            for (int r = 0; r < meth.reps; r++) {
                /* reset so the all-reduce always has real work to do */
                for (long i = 0; i < cfg.n; i++) v[i] = (double)(rank + 1);
                meth.fn(v.data(), cfg.n, MPI_COMM_WORLD);
            }
            MPI_Barrier(MPI_COMM_WORLD);
            double t_local = (MPI_Wtime() - t0) / meth.reps;
            double t_max;
            MPI_Reduce(&t_local, &t_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
            if (rank == 0) printf(" %14.6f", t_max);
        }
        if (rank == 0) printf("\n");
    }
    if (rank == 0) {
        printf("  (ring is slow for small data: needs 2*(p-1) messages per call)\n");
        fflush(stdout);
    }
}

/* -----------------------------------------------------------------------
 * Task 3.3: Scaling study
 * Runs with best all-reduce (MPI_Allreduce) at m=1024, batch=512 fixed.
 * All processes print Table 4(b) to confirm identical weights after all-reduce.
 * Run with p=1,2,4,8 externally; this function reports one row of the table.
 * --------------------------------------------------------------------- */
static void task3_3(int rank, int size)
{
    static const int M33 = 1024;

    if (rank == 0) {
        printf("\n=== Task 3.3: Scaling study (best=MPI_Allreduce, m=%d, p=%d, threads=%d) ===\n",
               M33, size, omp_get_max_threads());
        printf("  batch=%d fixed, S=%d steps\n", BATCH, S);
    }

    TimingResult r = train(M33, rank, size, allreduce_builtin,
                           "Task 3.3 (Scaling/Allreduce)", true);

    if (rank == 0) {
        /* Speedup and efficiency cannot be computed within a single run;
         * collect T(1) from the p=1 run and compute T(1)/T(p) in the report.
         * Print the raw wall time here so the report table can be filled in. */
        printf("\n  [3.3 table row] p=%d  total=%.6f s  compute=%.6f s/step  allreduce=%.6f s/step\n",
               size, r.total, r.compute, r.allreduce);
        fflush(stdout);
    }
}

/* -----------------------------------------------------------------------
 * main
 * --------------------------------------------------------------------- */
int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int m = DEFAULT_M;
    if (argc > 1) {
        m = atoi(argv[1]);
        if (m <= 0) {
            if (rank == 0) fprintf(stderr, "Usage: %s [m] [3.1|3.2|3.2t|3.3|all]\n", argv[0]);
            MPI_Finalize(); return 1;
        }
    }
    string task = (argc > 2) ? argv[2] : "all";

    if (rank == 0)
        printf("Processes=%d, threads=%d, m=%d, L=%d, S=%d, batch=%d\n",
               size, omp_get_max_threads(), m, L, S, BATCH);

    bool all = (task == "all");

    if (all || task == "3.1") {
        /* PDF: "let every process print the three weights of Table 4(b)" */
        train(m, rank, size, allreduce_reduce_bcast, "Task 3.1a (Reduce+Bcast)", true);
        train(m, rank, size, allreduce_builtin,      "Task 3.1b (Allreduce)",    true);
    }
    /* "3.2t" = ring training only (no benchmark), used for repeated timing runs */
    if (all || task == "3.2" || task == "3.2t") {
        train(m, rank, size, ring_allreduce, "Task 3.2 (Ring all-reduce)", true);
        if (task != "3.2t")
            benchmark_allreduce(rank, size);  /* 1KB/1MB/64MB across all three methods */
    }
    if (all || task == "3.3")
        task3_3(rank, size);  /* always uses m=1024 as per PDF spec */

    MPI_Finalize();
    return 0;
}
