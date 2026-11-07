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
#include <cstdlib>
#include <string>
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
static void task1_1(int rank, const int* sizes, int nsizes)
{
    if (rank != 0) return;

    int nthreads = 1;
#pragma omp parallel
#pragma omp single
    nthreads = omp_get_num_threads();

    cout << "\n========== Task 1.1  (OpenMP, " << nthreads
         << " thread" << (nthreads == 1 ? "" : "s") << ") ==========\n";

    for (int t = 0; t < nsizes; t++) {
        int N = sizes[t];

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
 * Row partition: first (N mod p) ranks get one extra row.
 * --------------------------------------------------------------------- */
static int rows_of(int r, int N, int p)  { return N / p + (r < N % p ? 1 : 0); }
static int start_of(int r, int N, int p) { return r * (N / p) + (r < N % p ? r : N % p); }

/* -----------------------------------------------------------------------
 * Task 1.2 — Version A: Scatterv(A) + Bcast(B) + local multiply + Gatherv(C)
 * Rank 0 builds full A and B; every rank receives the whole of B.
 * --------------------------------------------------------------------- */
static void task1_2_one(int N, int rank, int size)
{
    int my_rows = rows_of(rank, N, size);

    /* counts / displacements in doubles (rows * N) */
    int* counts = new int[size];
    int* displs = new int[size];
    for (int r = 0; r < size; r++) {
        counts[r] = rows_of(r, N, size) * N;
        displs[r] = start_of(r, N, size) * N;
    }

    double* A_full = nullptr;
    double* C_full = nullptr;
    double* B      = new double[(size_t)N * N];          /* every rank */
    double* local_A;
    double* local_C;

    if (rank == 0) {
        A_full = new double[(size_t)N * N];
        C_full = new double[(size_t)N * N];
        fill_A(A_full, 0, N, N);
        fill_B_full(B, N);
        /* rank 0's block is the first block of the full arrays (in-place) */
        local_A = A_full;
        local_C = C_full;
    } else {
        local_A = new double[(size_t)my_rows * N];
        local_C = new double[(size_t)my_rows * N];
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t_start = MPI_Wtime();

    /* 1. Scatter rows of A */
    double t0 = MPI_Wtime();
    if (rank == 0)
        MPI_Scatterv(A_full, counts, displs, MPI_DOUBLE,
                     MPI_IN_PLACE, counts[0], MPI_DOUBLE, 0, MPI_COMM_WORLD);
    else
        MPI_Scatterv(nullptr, counts, displs, MPI_DOUBLE,
                     local_A, counts[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);
    double t_scatter = elapsed_max(t0);

    /* 2. Broadcast all of B */
    t0 = MPI_Wtime();
    MPI_Bcast(B, N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    double t_bcast = elapsed_max(t0);

    /* 3. Local multiply: my rows of A x full B */
    t0 = MPI_Wtime();
    multiply_ikj_omp(local_A, B, local_C, my_rows, N);
    double t_mult = elapsed_max(t0);

    /* 4. Gather rows of C */
    t0 = MPI_Wtime();
    if (rank == 0)
        MPI_Gatherv(MPI_IN_PLACE, counts[0], MPI_DOUBLE,
                    C_full, counts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    else
        MPI_Gatherv(local_C, counts[rank], MPI_DOUBLE,
                    nullptr, counts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    double t_gather = elapsed_max(t0);

    double t_total = elapsed_max(t_start);

    if (rank == 0) {
        cout << "\n--- N=" << N << ", p=" << size << " ---\n"
             << setprecision(6) << defaultfloat
             << "Scatterv(A): " << t_scatter << " s\n"
             << "Bcast(B):    " << t_bcast   << " s\n"
             << "Multiply:    " << t_mult    << " s\n"
             << "Gatherv(C):  " << t_gather  << " s\n"
             << "Total:       " << t_total   << " s\n"
             << "Memory for B per process: "
             << (double)N * N * sizeof(double) / (1024.0 * 1024.0)
             << " MiB (x" << size << " copies)\n";
        if (N == 8) print_full_matrix(C_full, N);
        else        print_five_entries(C_full, N);
        cout << defaultfloat;
    }

    if (rank == 0) { delete[] A_full; delete[] C_full; }
    else           { delete[] local_A; delete[] local_C; }
    delete[] B;
    delete[] counts;
    delete[] displs;
}

static void task1_2(int rank, int size, const int* sizes, int nsizes)
{
    if (rank == 0)
        cout << "\n========== Task 1.2  (Version A: Scatterv/Bcast/Gatherv, p="
             << size << ") ==========\n";
    for (int t = 0; t < nsizes; t++) {
        if (sizes[t] < size) {
            if (rank == 0)
                cout << "\nSkipping N=" << sizes[t] << " (fewer rows than processes)\n";
            continue;
        }
        task1_2_one(sizes[t], rank, size);
    }
}

/* -----------------------------------------------------------------------
 * main
 *   ./task1 [N] [task]
 *     N     matrix size (default: 8, 1000, 2000, 4000)
 *     task  1.1 | 1.2 | all   (default: all)
 * --------------------------------------------------------------------- */
int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int* sizes = TEST_SIZES;
    int nsizes = NUM_SIZES;
    int user_N = 0;
    if (argc > 1) {
        user_N = atoi(argv[1]);
        if (user_N <= 0) {
            if (rank == 0) cerr << "Usage: " << argv[0] << " [N] [1.1|1.2|all]\n";
            MPI_Finalize();
            return 1;
        }
        sizes = &user_N;
        nsizes = 1;
    }
    string task = (argc > 2) ? argv[2] : "all";

    if (rank == 0)
        cout << "Processes=" << size << "\n";
    /* show where each rank is running (proves slaves are used) */
    {
        char host[MPI_MAX_PROCESSOR_NAME];
        int len;
        MPI_Get_processor_name(host, &len);
        char* all = nullptr;
        if (rank == 0) all = new char[(size_t)size * MPI_MAX_PROCESSOR_NAME];
        MPI_Gather(host, MPI_MAX_PROCESSOR_NAME, MPI_CHAR,
                   all, MPI_MAX_PROCESSOR_NAME, MPI_CHAR, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            for (int r = 0; r < size; r++)
                cout << "  rank " << r << " on " << &all[(size_t)r * MPI_MAX_PROCESSOR_NAME] << "\n";
            delete[] all;
        }
    }

    if (task == "1.1" || task == "all") {
        /* Task 1.1 is single-process; runs on rank 0 only */
        task1_1(rank, sizes, nsizes);
    }
    if (task == "1.2" || task == "all")
        task1_2(rank, size, sizes, nsizes);

    MPI_Finalize();
    return 0;
}
