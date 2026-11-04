#include <iostream>
#include <mpi.h>
#include <omp.h>

using namespace std;

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);

    int rank, size;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();

        #pragma omp critical
        {
            cout << "Process " << rank
                 << " | Thread " << thread_id
                 << " | Total Processes " << size
                 << endl;
        }
    }

    MPI_Finalize();

    return 0;
}
