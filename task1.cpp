/*
 * Task 1 — Dense Matrix Multiplication: C = A x B
 *
 * Tests N = 8, 1000, 2000, 4000. N=8 prints full matrix; N>8 prints five entries.
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
#include <vector>
#include <mpi.h>
#include <omp.h>

using namespace std;

static const int TEST_SIZES[] = {8, 1000, 2000, 4000};
static const int NUM_SIZES    = 4;

static inline double val_A(int i, int j) { return (double)(((2*i + j) % 7) - 3); }
static inline double val_B(int i, int j) { return (double)(((i + 3*j) % 5) - 1); }

static void fill_A(double* local_A, int row_start, int rows, int N)
{
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < N; j++)
            local_A[i*N + j] = val_A(row_start + i, j);
}

static void fill_B_full(double* B, int N)
{
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            B[i*N + j] = val_B(i, j);
}

static void fill_B_block(double* local_B, int row_start, int rows, int N)
{
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < N; j++)
            local_B[i*N + j] = val_B(row_start + i, j);
}

// i-j-k: naive, inner loop walks down a column of B -> cache misses.
static void multiply_ijk(const double* A, const double* B, double* C, int rows, int N)
{
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < N; j++) {
            double sum = 0.0;
            for (int k = 0; k < N; k++)
                sum += A[i*N + k] * B[k*N + j];
            C[i*N + j] = sum;
        }
}

// i-k-j: cache-friendly, inner loop streams along rows of B and C.
static void multiply_ikj(const double* A, const double* B, double* C, int rows, int N)
{
    memset(C, 0, (size_t)rows * N * sizeof(double));
    for (int i = 0; i < rows; i++)
        for (int k = 0; k < N; k++) {
            double a_ik = A[i*N + k];
            for (int j = 0; j < N; j++)
                C[i*N + j] += a_ik * B[k*N + j];
        }
}

static void multiply_ikj_omp(const double* A, const double* B, double* C, int rows, int N)
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
         << "  C(0,0)="     << C[0]
         << "  C(0,1)="     << C[1]
         << "  C(1,0)="     << C[N]
         << "  C(N/2,N/2)=" << C[(N/2)*N + N/2]
         << "  C(N-1,N-1)=" << C[(N-1)*N + (N-1)]
         << "\n" << setprecision(6) << defaultfloat;
}

static double elapsed_max(double t0)
{
    double t_local = MPI_Wtime() - t0, t_max;
    MPI_Reduce(&t_local, &t_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    return t_max;
}

static int rows_of(int r, int N, int p)  { return N / p + (r < N % p ? 1 : 0); }
static int start_of(int r, int N, int p) { return r * (N / p) + (r < N % p ? r : N % p); }

static void setup_counts_displs(int* counts, int* displs, int N, int size)
{
    for (int r = 0; r < size; r++) {
        counts[r] = rows_of(r, N, size) * N;
        displs[r] = start_of(r, N, size) * N;
    }
}

static void gather_C(double* local_C, double* C_full,
                     const int* counts, const int* displs, int rank)
{
    if (rank == 0)
        MPI_Gatherv(MPI_IN_PLACE, counts[0], MPI_DOUBLE,
                    C_full, counts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    else
        MPI_Gatherv(local_C, counts[rank], MPI_DOUBLE,
                    nullptr, counts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
}

/* N=8: gather to rank 0 and print full matrix (for checking only).
   Large N: each process prints only the five Table 1(b) entries it owns
   without gathering — PDF: "each entry is printed by the process that owns its row". */
static void print_check(const double* local_C, double* C_full,
                        const int* counts, const int* displs,
                        int N, int rank, int my_start, int my_rows)
{
    if (N == 8) {
        gather_C(const_cast<double*>(local_C), C_full, counts, displs, rank);
        if (rank == 0) print_full_matrix(C_full, N);
        return;
    }
    int er[] = {0, 0, 1, N/2, N-1};
    int ec[] = {0, 1, 0, N/2, N-1};
    for (int e = 0; e < 5; e++) {
        MPI_Barrier(MPI_COMM_WORLD);
        int r = er[e], c = ec[e];
        if (r >= my_start && r < my_start + my_rows) {
            cout << fixed << setprecision(0)
                 << "  C(" << r << "," << c << ")="
                 << local_C[(long)(r - my_start) * N + c]
                 << "  [rank " << rank << "]\n";
            cout.flush();
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
}

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

        double t0 = MPI_Wtime();
        multiply_ijk(A, B, C, N, N);
        cout << "i-j-k (serial):        " << MPI_Wtime() - t0 << " s\n";
        if (N == 8) print_full_matrix(C, N); else print_five_entries(C, N);

        t0 = MPI_Wtime();
        multiply_ikj(A, B, C, N, N);
        cout << "i-k-j (serial):        " << MPI_Wtime() - t0 << " s\n";
        if (N == 8) print_full_matrix(C, N); else print_five_entries(C, N);

        t0 = MPI_Wtime();
        multiply_ikj_omp(A, B, C, N, N);
        cout << "i-k-j (OpenMP x" << nthreads << "): " << MPI_Wtime() - t0 << " s\n";
        if (N == 8) print_full_matrix(C, N); else print_five_entries(C, N);

        delete[] A; delete[] B; delete[] C;
    }
}

// Task 1.2: Scatterv(A) + Bcast(B) + local multiply + Gatherv(C)
static void task1_2_one(int N, int rank, int size)
{
    int my_rows = rows_of(rank, N, size);
    int* counts = new int[size];
    int* displs = new int[size];
    setup_counts_displs(counts, displs, N, size);

    double* A_full = nullptr;
    double* C_full = nullptr;
    double* B      = new double[(size_t)N * N];
    double* local_A;
    double* local_C;

    if (rank == 0) {
        A_full = new double[(size_t)N * N];
        C_full = new double[(size_t)N * N];
        fill_A(A_full, 0, N, N);
        fill_B_full(B, N);
        local_A = A_full; local_C = C_full;
    } else {
        local_A = new double[(size_t)my_rows * N];
        local_C = new double[(size_t)my_rows * N];
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t_start = MPI_Wtime();

    double t0 = MPI_Wtime();
    if (rank == 0)
        MPI_Scatterv(A_full, counts, displs, MPI_DOUBLE,
                     MPI_IN_PLACE, counts[0], MPI_DOUBLE, 0, MPI_COMM_WORLD);
    else
        MPI_Scatterv(nullptr, counts, displs, MPI_DOUBLE,
                     local_A, counts[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);
    double t_scatter = elapsed_max(t0);

    t0 = MPI_Wtime();
    MPI_Bcast(B, N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    double t_bcast = elapsed_max(t0);

    t0 = MPI_Wtime();
    multiply_ikj_omp(local_A, B, local_C, my_rows, N);
    double t_mult = elapsed_max(t0);

    t0 = MPI_Wtime();
    gather_C(local_C, C_full, counts, displs, rank);
    double t_gather = elapsed_max(t0);
    double t_total  = elapsed_max(t_start);

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
    }

    if (rank == 0) { delete[] A_full; delete[] C_full; }
    else           { delete[] local_A; delete[] local_C; }
    delete[] B; delete[] counts; delete[] displs;
}

// Task 1.3: Blocking Ring — B blocks rotate; each process owns rows of A and B.
static void task1_3_one(int N, int rank, int size)
{
    int my_rows  = rows_of(rank, N, size);
    int my_start = start_of(rank, N, size);
    int* counts  = new int[size];
    int* displs  = new int[size];
    setup_counts_displs(counts, displs, N, size);

    int max_block_rows = rows_of(0, N, size);
    double* local_A = new double[(size_t)my_rows * N];
    double* local_C = new double[(size_t)my_rows * N]();
    double* buf[2];
    buf[0] = new double[(size_t)max_block_rows * N];
    buf[1] = new double[(size_t)max_block_rows * N];

    fill_A(local_A, my_start, my_rows, N);
    fill_B_block(buf[0], my_start, my_rows, N);

    int left  = (rank - 1 + size) % size;
    int right = (rank + 1) % size;
    int cur = 0, cur_owner = rank;

    MPI_Barrier(MPI_COMM_WORLD);
    double t_start = MPI_Wtime(), t_mult_total = 0.0, t_ring_total = 0.0;

    for (int s = 0; s < size; s++) {
        int q = cur_owner, q_rows = rows_of(q, N, size), q_start = start_of(q, N, size);

        double t0 = MPI_Wtime();
        for (int i = 0; i < my_rows; i++)
            for (int k = 0; k < q_rows; k++) {
                double a_ik = local_A[i*N + q_start + k];
                for (int j = 0; j < N; j++)
                    local_C[i*N + j] += a_ik * buf[cur][k*N + j];
            }
        t_mult_total += MPI_Wtime() - t0;

        if (s < size - 1) {
            int nxt = 1 - cur, next_owner = (cur_owner + 1) % size;
            int next_rows = rows_of(next_owner, N, size);
            t0 = MPI_Wtime();
            if (rank % 2 == 0) {
                MPI_Send(buf[cur], q_rows * N,    MPI_DOUBLE, left,  0, MPI_COMM_WORLD);
                MPI_Recv(buf[nxt], next_rows * N, MPI_DOUBLE, right, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            } else {
                MPI_Recv(buf[nxt], next_rows * N, MPI_DOUBLE, right, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Send(buf[cur], q_rows * N,    MPI_DOUBLE, left,  0, MPI_COMM_WORLD);
            }
            t_ring_total += MPI_Wtime() - t0;
            cur = nxt; cur_owner = next_owner;
        }
    }

    double t_mult_max, t_ring_max;
    MPI_Reduce(&t_mult_total, &t_mult_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t_ring_total, &t_ring_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    /* Capture total time before any gather — the gather is for checking only. */
    MPI_Barrier(MPI_COMM_WORLD);
    double t_total = elapsed_max(t_start);

    if (rank == 0) {
        cout << "\n--- N=" << N << ", p=" << size << " ---\n"
             << setprecision(6) << defaultfloat
             << "Ring-comm:   " << t_ring_max << " s\n"
             << "Multiply:    " << t_mult_max << " s\n"
             << "Total:       " << t_total    << " s\n"
             << "Memory/proc (2 B-blocks): "
             << 2.0 * max_block_rows * N * sizeof(double) / (1024.0 * 1024.0)
             << " MiB (vs "
             << (double)N * N * sizeof(double) / (1024.0 * 1024.0)
             << " MiB for Version A's full B)\n";
    }

    /* N=8: gather to rank 0 (for checking); large N: owning process prints its entries. */
    double* C_full = nullptr;
    if (N == 8 && rank == 0) {
        C_full = new double[(size_t)N * N];
        memcpy(C_full, local_C, (size_t)counts[0] * sizeof(double));
    }
    print_check(local_C, C_full, counts, displs, N, rank, my_start, my_rows);
    delete[] C_full;

    delete[] local_A; delete[] local_C;
    delete[] buf[0];  delete[] buf[1];
    delete[] counts;  delete[] displs;
}

// Task 1.4: Non-blocking Ring + OpenMP — Irecv/Isend posted before multiply.
static void task1_4_one(int N, int rank, int size)
{
    int my_rows  = rows_of(rank, N, size);
    int my_start = start_of(rank, N, size);
    int* counts  = new int[size];
    int* displs  = new int[size];
    setup_counts_displs(counts, displs, N, size);

    int max_block_rows = rows_of(0, N, size);
    double* local_A = new double[(size_t)my_rows * N];
    double* local_C = new double[(size_t)my_rows * N]();
    double* buf[2];
    buf[0] = new double[(size_t)max_block_rows * N];
    buf[1] = new double[(size_t)max_block_rows * N];

    fill_A(local_A, my_start, my_rows, N);
    fill_B_block(buf[0], my_start, my_rows, N);

    int left  = (rank - 1 + size) % size;
    int right = (rank + 1) % size;
    int cur = 0, cur_owner = rank;

    MPI_Barrier(MPI_COMM_WORLD);
    double t_start = MPI_Wtime(), t_mult_total = 0.0, t_wait_total = 0.0;
    vector<double> step_mult(size, 0.0), step_wait(size, 0.0);

    for (int s = 0; s < size; s++) {
        int q = cur_owner, q_rows = rows_of(q, N, size), q_start = start_of(q, N, size);
        int nxt = 1 - cur;
        MPI_Request reqs[2];
        bool has_comm = (s < size - 1);

        if (has_comm) {
            int next_owner = (cur_owner + 1) % size;
            int next_rows  = rows_of(next_owner, N, size);
            MPI_Irecv(buf[nxt], next_rows * N, MPI_DOUBLE, right, 0, MPI_COMM_WORLD, &reqs[0]);
            MPI_Isend(buf[cur], q_rows * N,    MPI_DOUBLE, left,  0, MPI_COMM_WORLD, &reqs[1]);
        }

        double t0 = MPI_Wtime();
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < my_rows; i++)
            for (int k = 0; k < q_rows; k++) {
                double a_ik = local_A[i*N + q_start + k];
                for (int j = 0; j < N; j++)
                    local_C[i*N + j] += a_ik * buf[cur][k*N + j];
            }
        step_mult[s] = MPI_Wtime() - t0;
        t_mult_total += step_mult[s];

        if (has_comm) {
            t0 = MPI_Wtime();
            MPI_Waitall(2, reqs, MPI_STATUSES_IGNORE);
            step_wait[s] = MPI_Wtime() - t0;
            t_wait_total += step_wait[s];
            cur = nxt; cur_owner = (cur_owner + 1) % size;
        }
    }

    double t_mult_max, t_wait_max;
    MPI_Reduce(&t_mult_total, &t_mult_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t_wait_total, &t_wait_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    vector<double> step_mult_max(size), step_wait_max(size);
    MPI_Reduce(step_mult.data(), step_mult_max.data(), size, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(step_wait.data(), step_wait_max.data(), size, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    /* Capture total time before any gather — the gather is for checking only. */
    MPI_Barrier(MPI_COMM_WORLD);
    double t_total = elapsed_max(t_start);

    if (rank == 0) {
        int nthreads = 1;
        #pragma omp parallel
        #pragma omp single
        nthreads = omp_get_num_threads();

        cout << "\n--- N=" << N << ", p=" << size << ", threads=" << nthreads << " ---\n"
             << setprecision(6) << defaultfloat
             << "Multiply:    " << t_mult_max << " s\n"
             << "Wait(comm):  " << t_wait_max << " s\n"
             << "Total:       " << t_total    << " s\n"
             << "Memory/proc (2 B-blocks): "
             << 2.0 * max_block_rows * N * sizeof(double) / (1024.0 * 1024.0) << " MiB\n";
        cout << "Per-step times (max over ranks):\n"
             << "  step   multiply(s)    wait(s)\n";
        for (int s = 0; s < size; s++)
            cout << "  " << setw(4) << s
                 << fixed << setprecision(6)
                 << "   " << setw(10) << step_mult_max[s]
                 << "   " << setw(10) << step_wait_max[s] << "\n";
        cout << setprecision(6) << defaultfloat;
    }

    /* N=8: gather to rank 0 (for checking); large N: owning process prints its entries. */
    double* C_full = nullptr;
    if (N == 8 && rank == 0) {
        C_full = new double[(size_t)N * N];
        memcpy(C_full, local_C, (size_t)counts[0] * sizeof(double));
    }
    print_check(local_C, C_full, counts, displs, N, rank, my_start, my_rows);
    delete[] C_full;

    delete[] local_A; delete[] local_C;
    delete[] buf[0];  delete[] buf[1];
    delete[] counts;  delete[] displs;
}

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
            if (rank == 0) cerr << "Usage: " << argv[0] << " [N] [1.1|1.2|1.3|1.4|all]\n";
            MPI_Finalize(); return 1;
        }
        sizes = &user_N; nsizes = 1;
    }
    string task = (argc > 2) ? argv[2] : "all";

    if (rank == 0) cout << "Processes=" << size << "\n";
    {
        char host[MPI_MAX_PROCESSOR_NAME]; int len;
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

    // Helper to run a task function over all test sizes, skipping under-sized ones.
    using SizeFn = void(*)(int, int, int);
    auto run_all = [&](const char* hdr, SizeFn fn) {
        if (rank == 0)
            cout << "\n========== " << hdr << " (p=" << size << ") ==========\n";
        for (int t = 0; t < nsizes; t++) {
            if (sizes[t] < size) {
                if (rank == 0)
                    cout << "\nSkipping N=" << sizes[t] << " (fewer rows than processes)\n";
                continue;
            }
            fn(sizes[t], rank, size);
        }
    };

    if (task == "1.1" || task == "all") task1_1(rank, sizes, nsizes);
    if (task == "1.2" || task == "all")
        run_all("Task 1.2  (Version A: Scatterv/Bcast/Gatherv)", task1_2_one);
    if (task == "1.3" || task == "all")
        run_all("Task 1.3  (Version B: Blocking Ring)", task1_3_one);
    if (task == "1.4" || task == "all")
        run_all("Task 1.4  (Version C: Non-blocking Ring + OpenMP)", task1_4_one);

    MPI_Finalize();
    return 0;
}
