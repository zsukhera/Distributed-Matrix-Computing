/*
 * Task 1 — Dense Matrix Multiplication: C = A x B
 *
 * Automatically tests N = 8, 1000, 2000, 4000.
 * N=8  : prints the full matrix  (compare Table 1a)
 * N>8  : prints five entries     (compare Table 1b)
 *
 * Build:  make task1
 * Run:    OMP_NUM_THREADS=4 mpirun -n 1 ./task1       (Task 1.1)
 *         OMP_NUM_THREADS=1 mpirun -n 4 ./task1       (Tasks 1.2-1.4)
 */

#include <iostream>
#include <iomanip>
#include <cstring>
#include <mpi.h>
#include <omp.h>

using namespace std;

/* -----------------------------------------------------------------------
 * Test sizes — all versions loop over these automatically
 * --------------------------------------------------------------------- */
static const int TEST_SIZES[] = {8, 1000, 2000, 4000};
static const int NUM_SIZES    = 4;

/* -----------------------------------------------------------------------
 * Data initialisation
 * A(i,j) = ((2i + j) mod 7) − 3
 * B(i,j) = ((i + 3j) mod 5) − 1
 * --------------------------------------------------------------------- */
static inline double val_A(int i, int j)
{
    return (double)(((2*i + j) % 7) - 3);
}

static inline double val_B(int i, int j)
{
    return (double)(((i + 3*j) % 5) - 1);
}

/* Fill rows [row_start, row_start+rows) of A into local buffer. */
static void fill_A(double* local_A, int row_start, int rows, int N)
{
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < N; j++)
            local_A[i*N + j] = val_A(row_start + i, j);
}

/* Fill the full N x N matrix B. */
static void fill_B_full(double* B, int N)
{
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            B[i*N + j] = val_B(i, j);
}

/* Fill rows [row_start, row_start+rows) of B into local buffer.
   Used in Tasks 1.3 and 1.4 where each process owns only its block. */
static void fill_B_block(double* local_B, int row_start, int rows, int N)
{
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < N; j++)
            local_B[i*N + j] = val_B(row_start + i, j);
}

/* -----------------------------------------------------------------------
 * Multiplication kernels
 * A: (rows x N),  B: (N x N),  C: (rows x N)  — all row-major
 * --------------------------------------------------------------------- */

/* i-j-k: naive. Inner loop walks down a column of B -> cache misses. */
static void multiply_ijk(const double* A, const double* B,
                         double* C, int rows, int N)
{
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < N; j++) {
            double sum = 0.0;
            for (int k = 0; k < N; k++)
                sum += A[i*N + k] * B[k*N + j];
            C[i*N + j] = sum;
        }
}

/* i-k-j: cache-friendly. Inner loop streams along rows of B and C. */
static void multiply_ikj(const double* A, const double* B,
                         double* C, int rows, int N)
{
    memset(C, 0, (size_t)rows * N * sizeof(double));
    for (int i = 0; i < rows; i++)
        for (int k = 0; k < N; k++) {
            double a_ik = A[i*N + k];
            for (int j = 0; j < N; j++)
                C[i*N + j] += a_ik * B[k*N + j];
        }
}

/* i-k-j + OpenMP: parallel over rows of C.
   Thread t owns a subset of rows, so each C element is written by
   exactly one thread — no race condition, no locks needed. */
static void multiply_ikj_omp(const double* A, const double* B,
                              double* C, int rows, int N)
{
#pragma omp parallel for schedule(static)
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < N; j++) C[i*N + j] = 0.0;
        for (int k = 0; k < N; k++) {
            double a_ik = A[i*N + k];
            for (int j = 0; j < N; j++)
                C[i*N + j] += a_ik * B[k*N + j];
        }
    }
}

/* -----------------------------------------------------------------------
 * Output helpers
 * --------------------------------------------------------------------- */
static void print_full_matrix(const double* C, int N)
{
    cout << "Full matrix C (N=" << N << "):\n";
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++)
            cout << setw(5) << (int)C[i*N + j];
        cout << "\n";
    }
}

static void print_five_entries(const double* C, int N)
{
    cout << fixed << setprecision(0)
         << "  C(0,0)="       << C[0]
         << "  C(0,1)="       << C[1]
         << "  C(1,0)="       << C[N]
         << "  C(N/2,N/2)="   << C[(N/2)*N + N/2]
         << "  C(N-1,N-1)="   << C[(N-1)*N + (N-1)]
         << "\n";
}

/* -----------------------------------------------------------------------
 * Timing: wall time of the slowest process (MPI_MAX across all ranks).
 * --------------------------------------------------------------------- */
static double elapsed_max(double t0)
{
    double t_local = MPI_Wtime() - t0;
    double t_max;
    MPI_Reduce(&t_local, &t_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    return t_max;
}

/* -----------------------------------------------------------------------
 * Task 1.1 — serial i-j-k, serial i-k-j, OpenMP i-k-j
 * Single MPI process; all three kernels run for every test size.
 * --------------------------------------------------------------------- */
static void task1_1(int rank)
{
    if (rank != 0) return;

    int nthreads = 1;
#pragma omp parallel
#pragma omp single
    nthreads = omp_get_num_threads();

    cout << "\n========== Task 1.1  (OpenMP, " << nthreads
         << " thread" << (nthreads == 1 ? "" : "s") << ") ==========\n";

    for (int t = 0; t < NUM_SIZES; t++) {
        int N = TEST_SIZES[t];

        double* A = new double[(size_t)N * N];
        double* B = new double[(size_t)N * N];
        double* C = new double[(size_t)N * N];

        fill_A(A, 0, N, N);
        fill_B_full(B, N);

        cout << "\n--- N=" << N << " ---\n";

        /* i-j-k */
        {
            double t0 = MPI_Wtime();
            multiply_ijk(A, B, C, N, N);
            double elapsed = MPI_Wtime() - t0;
            cout << "i-j-k (serial):        " << elapsed << " s\n";
            if (N == 8) print_full_matrix(C, N);
            else        print_five_entries(C, N);
        }

        /* i-k-j serial */
        {
            double t0 = MPI_Wtime();
            multiply_ikj(A, B, C, N, N);
            double elapsed = MPI_Wtime() - t0;
            cout << "i-k-j (serial):        " << elapsed << " s\n";
            if (N == 8) print_full_matrix(C, N);
            else        print_five_entries(C, N);
        }

        /* i-k-j OpenMP */
        {
            double t0 = MPI_Wtime();
            multiply_ikj_omp(A, B, C, N, N);
            double elapsed = MPI_Wtime() - t0;
            cout << "i-k-j (OpenMP x"
                 << nthreads << "): " << elapsed << " s\n";
            if (N == 8) print_full_matrix(C, N);
            else        print_five_entries(C, N);
        }

        delete[] A;
        delete[] B;
        delete[] C;
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

    if (rank == 0)
        cout << "Processes=" << size << "\n";

    /* Task 1.1 */
    task1_1(rank);

    /* Tasks 1.2, 1.3, 1.4 will be added here */

    MPI_Finalize();
    return 0;
}
