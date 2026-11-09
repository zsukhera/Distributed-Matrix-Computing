/*
 * Task 2 — Sparse Matrix Multiplication on a Social Network: H_new = W x H
 *
 *   2.1  serial Code 3, then OpenMP schedule(static) vs schedule(dynamic,64)
 *   2.2  Version A: every process gets all of H with MPI_Allgatherv
 *   2.3  Version B: 128-row halo exchange; interior rows computed before Waitall
 *   2.4  Version B: equal-rows vs equal-links partitioning
 *
 * Build:  make task2
 * Run:    OMP_NUM_THREADS=<t> mpirun -n <p> --hostfile hosts ./task2 [n] [task]
 */

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <mpi.h>
#include <omp.h>

using namespace std;

static const int F    = 16;
static const int T    = 10;
static const int HALO = 128;

static const long DEFAULT_SIZES[] = {100000, 1000000};
static const int  NUM_DEFAULT     = 2;

// Network generator (Code 4): users on a circle, links in [-128, 128].
// Hub users: first 10% AND hash divisible by 4 -> 64..127 links; others -> 4..12.
static inline uint32_t hash32(uint32_t x)
{
    x ^= x >> 16; x *= 0x7feb352dU;
    x ^= x >> 15; x *= 0x846ca68bU;
    x ^= x >> 16;
    return x;
}

static inline int degree(long i, long n)
{
    uint32_t h = hash32((uint32_t)i);
    if (i < n / 10 && h % 4 == 0) return 64 + (int)(h % 64);
    return 4 + (int)(h % 9);
}

static inline int link_offset(long i, int k)
{
    uint32_t h = hash32((uint32_t)(i * 131 + k * 7 + 1));
    return (int)(h % 257) - 128;
}

static inline long neighbour(long i, int k, long n)
{
    long j = (i + link_offset(i, k)) % n;
    return j < 0 ? j + n : j;
}

struct CSR {
    long           row_start = 0, rows = 0;
    vector<long>   row_ptr;
    vector<int>    col_idx;
    vector<double> val;
    long nnz() const { return row_ptr.empty() ? 0 : row_ptr[rows]; }
};

static void build_csr(CSR& A, long row_start, long rows, long n)
{
    A.row_start = row_start;
    A.rows      = rows;
    A.row_ptr.assign(rows + 1, 0);
    for (long r = 0; r < rows; r++)
        A.row_ptr[r + 1] = A.row_ptr[r] + degree(row_start + r, n);

    long nnz = A.row_ptr[rows];
    A.col_idx.resize(nnz);
    A.val.resize(nnz);

    #pragma omp parallel for schedule(dynamic, 64)
    for (long r = 0; r < rows; r++) {
        long   i = row_start + r;
        int    d = degree(i, n);
        double w = 1.0 / d;
        long   p = A.row_ptr[r];
        for (int k = 0; k < d; k++) {
            A.col_idx[p + k] = (int)neighbour(i, k, n);
            A.val[p + k]     = w;
        }
    }
}

static inline double h0(long i, int f) { return (double)((31 * i + 17 * f) % 100) / 100.0; }

static void fill_H(double* H, long row_start, long rows)
{
    #pragma omp parallel for schedule(static)
    for (long r = 0; r < rows; r++)
        for (int f = 0; f < F; f++)
            H[r * F + f] = h0(row_start + r, f);
}

// Code 3: compute one row of H_new = W x H; return link count.
static inline long csr_row(const long* row_ptr, const int* col, const double* val,
                           const double* Hin, double* Hout, long r)
{
    double acc[F] = {0};
    for (long k = row_ptr[r]; k < row_ptr[r + 1]; k++)
        for (int f = 0; f < F; f++)
            acc[f] += val[k] * Hin[(long)col[k] * F + f];
    for (int f = 0; f < F; f++)
        Hout[r * F + f] = acc[f];
    return row_ptr[r + 1] - row_ptr[r];
}

struct ThreadStats {
    vector<double> busy;
    vector<long>   links;
    explicit ThreadStats(int nt) : busy(nt, 0.0), links(nt, 0) {}
};

// Runs the OpenMP CSR step (Code 3) over rows [r_begin, r_end).
// Each thread picks up rows and accumulates: Hout[r] = sum_k val[k] * Hin[col[k]].
// dynamic_sched=true uses schedule(dynamic,64) to balance hub-user rows; false uses static.
// If ts is provided, each thread records its busy time and link count for load analysis.
static void csr_rows(const long* row_ptr, const int* col, const double* val,
                     const double* Hin, double* Hout, long r_begin, long r_end,
                     bool dynamic_sched, ThreadStats* ts = nullptr)
{
    if (r_end <= r_begin) return;
    if (dynamic_sched) omp_set_schedule(omp_sched_dynamic, 64);
    else               omp_set_schedule(omp_sched_static,  1);
    #pragma omp parallel
    {
        double t0    = omp_get_wtime();
        long   links = 0;
        #pragma omp for schedule(runtime) nowait
        for (long r = r_begin; r < r_end; r++)
            links += csr_row(row_ptr, col, val, Hin, Hout, r);
        if (ts) {
            int tid = omp_get_thread_num();
            ts->busy[tid]  += omp_get_wtime() - t0;
            ts->links[tid] += links;
        }
    }
}

struct Entry { long row; int f; };

static Entry entry_of(int e, long n)
{
    if (e == 0) return {0, 0};
    if (e == 1) return {12345, 7};
    return {n - 1, 15};
}

static long expected_links(long n)
{
    if (n == 100000)  return 1020227;
    if (n == 1000000) return 10160271;
    return -1;
}

static const char* expected_value(long n, int e)
{
    static const char* t100k[3] = {"0.493896795310", "0.497876802886", "0.497973641990"};
    static const char* t1m[3]   = {"0.500882792529", "0.499154198420", "0.492279183027"};
    if (n == 100000)  return t100k[e];
    if (n == 1000000) return t1m[e];
    return nullptr;
}

static void print_entry(long n, int e, double v, int rank)
{
    Entry en = entry_of(e, n);
    char buf[64];
    snprintf(buf, sizeof buf, "%.12f", v);
    printf("  H(%ld,%d) = %s", en.row, en.f, buf);
    const char* ref = expected_value(n, e);
    if (ref) printf("   (Table 3: %s  %s)", ref, strcmp(buf, ref) == 0 ? "OK" : "MISMATCH");
    if (rank >= 0) printf("   [printed by rank %d]", rank);
    printf("\n");
    fflush(stdout);
}

// Returns b[0..p] where process r owns users [b[r], b[r+1]).
// Divides n users as evenly as possible: first (n%p) ranks get one extra row.
static vector<long> equal_rows_bounds(long n, int p)
{
    vector<long> b(p + 1);
    for (int r = 0; r <= p; r++)
        b[r] = r * (n / p) + (r < n % p ? r : n % p);
    return b;
}

// Returns b[0..p] where each process gets roughly equal total link count (work).
// Only rank 0 does the O(n) boundary computation; MPI_Bcast distributes the result.
static vector<long> equal_links_bounds(long n, int p, int rank)
{
    vector<long> b(p + 1, 0);
    if (rank == 0) {
        // Count total links across all users.
        long total = 0;
        for (long i = 0; i < n; i++) total += degree(i, n);

        b[0] = 0; b[p] = n;
        long run = 0; int r = 1;
        for (long i = 0; i < n && r < p; i++) {
            run += degree(i, n);
            // Once run/total >= r/p, we have passed the r-th equal share — cut here.
            while (r < p && run * p >= total * r) b[r++] = i + 1;
        }
        // Forward pass: push boundaries right so every block has at least HALO rows.
        for (int q = 1;     q < p;  q++) b[q] = max(b[q], b[q - 1] + HALO);
        // Backward pass: pull boundaries left so the forward pass didn't overshoot.
        for (int q = p - 1; q >= 1; q--) b[q] = min(b[q], b[q + 1] - HALO);
    }
    // Rank 0 broadcasts the computed boundaries to all other processes.
    MPI_Bcast(b.data(), p + 1, MPI_LONG, 0, MPI_COMM_WORLD);
    return b;
}

struct RankStats {
    double compute = 0.0, comm = 0.0, wait = 0.0;
    long   rows = 0, links = 0;
    double bytes_per_step = 0.0;
};

// Collects per-process stats onto rank 0 and prints the full results table.
// 1. MPI_Reduce(MAX) finds the slowest process — that is the true wall time.
// 2. MPI_Gather brings each process's RankStats (compute, comm, wait, rows, links,
//    bytes/step) to rank 0, which prints one row per process plus the compute
//    imbalance ratio (max/avg) and total link count vs Table 3.
// 3. Each of the three Table 3 check entries is printed by whichever process owns
//    that row — avoids gathering all of H to rank 0.
static void report(const char* title, long n, int rank, int size,
                   const RankStats& s, double t_local,
                   const double* H_local, long my_start, long my_rows)
{
    double t_max;
    //finding the slowest process
    MPI_Reduce(&t_local, &t_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    const int K = 6;
    double mine[K] = {s.compute, s.comm, s.wait,
                      (double)s.rows, (double)s.links, s.bytes_per_step};
    vector<double> all(rank == 0 ? (size_t)K * size : 0);
    //gather all the stats across processes to rank 0
    MPI_Gather(mine, K, MPI_DOUBLE, all.data(), K, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        long   total_links = 0;
        double cmax = 0.0, csum = 0.0;
        printf("\n--- %s: n=%ld, p=%d, threads=%d ---\n",
               title, n, size, omp_get_max_threads());
        printf("  rank      rows      links   compute(s)    comm(s)    wait(s)   recv/step\n");
        for (int r = 0; r < size; r++) {
            const double* a = &all[(size_t)r * K];
            printf("  %4d %9.0f %10.0f   %10.6f %10.6f %10.6f   %7.1f KiB\n",
                   r, a[3], a[4], a[0], a[1], a[2], a[5] / 1024.0);
            total_links += (long)a[4];
            cmax = max(cmax, a[0]); csum += a[0];
        }
        long ref = expected_links(n);
        printf("  Total links: %ld", total_links);
        if (ref > 0) printf("  (Table 3: %ld  %s)", ref, total_links == ref ? "OK" : "MISMATCH");
        printf("\n  Compute imbalance (max/avg): %.3f\n", cmax / (csum / size));
        printf("  Total time:    %.6f s\n  Time per step: %.6f s\n", t_max, t_max / T);
        fflush(stdout);
    }

    for (int e = 0; e < 3; e++) {
        MPI_Barrier(MPI_COMM_WORLD);
        Entry en = entry_of(e, n);
        if (en.row >= my_start && en.row < my_start + my_rows)
            print_entry(n, e, H_local[(en.row - my_start) * F + en.f], rank);
    }
    MPI_Barrier(MPI_COMM_WORLD);
}

static void task2_1(long n, int rank)
{
    //will only be run by rank 0
    if (rank != 0) return;

    int nt = omp_get_max_threads();
    CSR W;
    build_csr(W, 0, n, n);
    vector<double> H((size_t)n * F), Hn((size_t)n * F);

    printf("\n--- Task 2.1: n=%ld, threads=%d ---\n", n, nt);
    long ref = expected_links(n);
    printf("  Total links: %ld", W.nnz());
    if (ref > 0) printf("  (Table 3: %ld  %s)", ref, W.nnz() == ref ? "OK" : "MISMATCH");
    printf("\n");

    const long*   rp  = W.row_ptr.data();
    const int*    ci  = W.col_idx.data();
    const double* val = W.val.data();

    // mode: 0=serial, 1=static, 2=dynamic
    auto run = [&](const char* name, int mode, ThreadStats* ts) {
        fill_H(H.data(), 0, n);
        double t0 = omp_get_wtime();
        for (int t = 0; t < T; t++) {
            if (mode == 0)
                for (long r = 0; r < n; r++) csr_row(rp, ci, val, H.data(), Hn.data(), r);
            else
                csr_rows(rp, ci, val, H.data(), Hn.data(), 0, n, mode != 1, ts);
            swap(H, Hn);
        }
        double el = omp_get_wtime() - t0;
        printf("\n%s: total %.6f s, per step %.6f s\n", name, el, el / T);
        for (int e = 0; e < 3; e++) {
            Entry en = entry_of(e, n);
            if (en.row < n) print_entry(n, e, H[en.row * F + en.f], -1);
        }
        return el;
    };

    ThreadStats st(nt), dy(nt);
    double t_ser = run("Serial (Code 3)", 0, nullptr);
    double t_sta = run("OpenMP schedule(static)", 1, &st);
    double t_dyn = run("OpenMP schedule(dynamic, 64)", 2, &dy);

    printf("\nSpeedup over serial: static %.3f, dynamic %.3f\n", t_ser / t_sta, t_ser / t_dyn);
    printf("\nPer-thread work (per step):\n");
    printf("  thread   static: links    busy(s)   dynamic: links    busy(s)\n");
    long smax = 0, ssum = 0, dmax = 0, dsum = 0;
    for (int t = 0; t < nt; t++) {
        printf("  %6d   %13ld  %9.6f   %14ld  %9.6f\n", t,
               st.links[t] / T, st.busy[t] / T, dy.links[t] / T, dy.busy[t] / T);
        smax = max(smax, st.links[t]); ssum += st.links[t];
        dmax = max(dmax, dy.links[t]); dsum += dy.links[t];
    }
    if (nt > 1)
        printf("  Link imbalance (max/avg): static %.3f, dynamic %.3f\n",
               (double)smax / ((double)ssum / nt), (double)dmax / ((double)dsum / nt));
    fflush(stdout);
}

static void version_A(long n, int rank, int size)
{
    vector<long> b = equal_rows_bounds(n, size);
    long my_start = b[rank], my_rows = b[rank + 1] - b[rank];

    CSR W;
    build_csr(W, my_start, my_rows, n);

    vector<double> H_full((size_t)n * F), H_loc((size_t)my_rows * F);
    fill_H(H_loc.data(), my_start, my_rows);

    vector<int> counts(size), displs(size);
    for (int r = 0; r < size; r++) {
        counts[r] = (int)((b[r + 1] - b[r]) * F);
        displs[r] = (int)(b[r] * F);
    }

    RankStats s;
    s.rows = my_rows; s.links = W.nnz();
    s.bytes_per_step = (double)(n - my_rows) * F * sizeof(double);

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    for (int t = 0; t < T; t++) {
        double tc = MPI_Wtime();
        MPI_Allgatherv(H_loc.data(), counts[rank], MPI_DOUBLE,
                       H_full.data(), counts.data(), displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);
        s.comm += MPI_Wtime() - tc;

        tc = MPI_Wtime();
        csr_rows(W.row_ptr.data(), W.col_idx.data(), W.val.data(),
                 H_full.data(), H_loc.data(), 0, my_rows, true);
        s.compute += MPI_Wtime() - tc;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    report("Task 2.2 Version A (Allgatherv)", n, rank, size, s, MPI_Wtime() - t0,
           H_loc.data(), my_start, my_rows);
}

static const int TAG_TO_LEFT  = 0;
static const int TAG_TO_RIGHT = 1;

// H buffer layout: [HALO left-halo | my_rows | HALO right-halo].
// Interior rows (no halo dependency) can be computed while halos travel.
static void version_B(long n, int rank, int size, const vector<long>& b,
                      bool overlap, const char* title)
{
    long my_start = b[rank], my_rows = b[rank + 1] - b[rank];
    int  left  = (rank - 1 + size) % size;
    int  right = (rank + 1) % size;

    CSR W;
    build_csr(W, my_start, my_rows, n);

    // Rewrite col indices to offsets into the extended buffer.
    vector<int> lcol(W.nnz());
    #pragma omp parallel for schedule(dynamic, 64)
    for (long r = 0; r < my_rows; r++) {
        long i = my_start + r, p = W.row_ptr[r];
        int  d = (int)(W.row_ptr[r + 1] - p);
        for (int k = 0; k < d; k++)
            lcol[p + k] = (int)(HALO + r + link_offset(i, k));
    }
    vector<int>().swap(W.col_idx);

    long ext_rows = my_rows + 2 * HALO;
    vector<double> H_ext((size_t)ext_rows * F), H_nxt((size_t)ext_rows * F);
    fill_H(H_ext.data() + (size_t)HALO * F, my_start, my_rows);

    const int cnt     = HALO * F;
    long int_begin    = HALO, int_end = my_rows - HALO;
    bool has_interior = int_end > int_begin;
    long b1_end       = min((long)HALO, my_rows);
    long b2_begin     = max((long)HALO, my_rows - HALO);

    RankStats s;
    s.rows = my_rows; s.links = W.nnz();
    s.bytes_per_step = 2.0 * cnt * sizeof(double);

    const long*   rp  = W.row_ptr.data();
    const int*    ci  = lcol.data();
    const double* val = W.val.data();

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    for (int t = 0; t < T; t++) {
        double* H  = H_ext.data();
        double* Hn = H_nxt.data() + (size_t)HALO * F;
        MPI_Request req[4];

        double tc = MPI_Wtime();
        MPI_Irecv(H,                                 cnt, MPI_DOUBLE, left,  TAG_TO_RIGHT, MPI_COMM_WORLD, &req[0]);
        MPI_Irecv(H + (size_t)(HALO + my_rows) * F, cnt, MPI_DOUBLE, right, TAG_TO_LEFT,  MPI_COMM_WORLD, &req[1]);
        MPI_Isend(H + (size_t)HALO * F,             cnt, MPI_DOUBLE, left,  TAG_TO_LEFT,  MPI_COMM_WORLD, &req[2]);
        MPI_Isend(H + (size_t)my_rows * F,          cnt, MPI_DOUBLE, right, TAG_TO_RIGHT, MPI_COMM_WORLD, &req[3]);
        s.comm += MPI_Wtime() - tc;

        if (overlap && has_interior) {
            tc = MPI_Wtime();
            csr_rows(rp, ci, val, H, Hn, int_begin, int_end, true);
            s.compute += MPI_Wtime() - tc;
        }

        tc = MPI_Wtime();
        MPI_Waitall(4, req, MPI_STATUSES_IGNORE);
        double w = MPI_Wtime() - tc;
        s.wait += w; s.comm += w;

        tc = MPI_Wtime();
        if (overlap) {
            csr_rows(rp, ci, val, H, Hn, 0,        b1_end,  true);
            csr_rows(rp, ci, val, H, Hn, b2_begin, my_rows, true);
        } else {
            csr_rows(rp, ci, val, H, Hn, 0, my_rows, true);
        }
        s.compute += MPI_Wtime() - tc;

        swap(H_ext, H_nxt);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    report(title, n, rank, size, s, MPI_Wtime() - t0,
           H_ext.data() + (size_t)HALO * F, my_start, my_rows);
}

static void task2_3(long n, int rank, int size)
{
    vector<long> b = equal_rows_bounds(n, size);
    version_B(n, rank, size, b, true,  "Task 2.3 Version B (halo, interior before Waitall)");
    version_B(n, rank, size, b, false, "Task 2.3 Version B (halo, all rows after Waitall)");
}

static void task2_4(long n, int rank, int size)
{
    // Two partitions: equal user count vs equal link count (work).
    vector<long> eq  = equal_rows_bounds(n, size);
    vector<long> bal = equal_links_bounds(n, size, rank);

    // Print both sets of boundaries side by side so the difference is visible.
    // Equal-rows gives uniform row counts; equal-links gives narrower slices for
    // the hub region (first 10%) and wider slices for the sparse normal region.
    if (rank == 0) {
        printf("\n--- Task 2.4 block boundaries (n=%ld, p=%d) ---\n", n, size);
        printf("  rank   equal rows: first user, rows   equal links: first user, rows\n");
        for (int r = 0; r < size; r++)
            printf("  %4d   %17ld %7ld   %18ld %7ld\n", r,
                   eq[r], eq[r+1] - eq[r], bal[r], bal[r+1] - bal[r]);
        fflush(stdout);
    }

    // Run Version B (with overlap) under both partitions.
    // Compare the compute imbalance (max/avg) in the report:
    // equal-rows will show a high ratio; equal-links should be close to 1.0.
    version_B(n, rank, size, eq,  true, "Task 2.4 Version B, equal ROWS");
    version_B(n, rank, size, bal, true, "Task 2.4 Version B, equal LINKS (balanced)");
}

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    vector<long> sizes(DEFAULT_SIZES, DEFAULT_SIZES + NUM_DEFAULT);
    if (argc > 1) {
        long n = atol(argv[1]);
        if (n <= 0) {
            if (rank == 0) fprintf(stderr, "Usage: %s [n] [2.1|2.2|2.3|2.4|all]\n", argv[0]);
            MPI_Finalize(); return 1;
        }
        sizes.assign(1, n);
    }
    string task = (argc > 2) ? argv[2] : "all";

    if (rank == 0)
        printf("Processes=%d, OpenMP threads per process=%d\n", size, omp_get_max_threads());
    {
        char host[MPI_MAX_PROCESSOR_NAME]; int len;
        MPI_Get_processor_name(host, &len);
        vector<char> all(rank == 0 ? (size_t)size * MPI_MAX_PROCESSOR_NAME : 0);
        MPI_Gather(host, MPI_MAX_PROCESSOR_NAME, MPI_CHAR,
                   all.data(), MPI_MAX_PROCESSOR_NAME, MPI_CHAR, 0, MPI_COMM_WORLD);
        if (rank == 0)
            for (int r = 0; r < size; r++)
                printf("  rank %d on %s\n", r, &all[(size_t)r * MPI_MAX_PROCESSOR_NAME]);
        fflush(stdout);
    }

    bool all = (task == "all");
    for (long n : sizes) {
        if (all || task == "2.1") task2_1(n, rank);
        MPI_Barrier(MPI_COMM_WORLD);

        if (!(all || task == "2.2" || task == "2.3" || task == "2.4")) continue;
        if (n / size < HALO) {
            if (rank == 0)
                printf("\nSkipping MPI versions for n=%ld: need at least %d users per process\n", n, HALO);
            continue;
        }
        if (all || task == "2.2") version_A(n, rank, size);
        if (all || task == "2.3") task2_3(n, rank, size);
        if (all || task == "2.4") task2_4(n, rank, size);
    }

    MPI_Finalize();
    return 0;
}
