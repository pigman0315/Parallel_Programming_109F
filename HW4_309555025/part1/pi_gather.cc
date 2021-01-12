#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/types.h>
#include <unistd.h>

int main(int argc, char **argv)
{
    // --- DON'T TOUCH ---
    MPI_Init(&argc, &argv);
    double start_time = MPI_Wtime();
    double pi_result;
    long long int tosses = atoi(argv[1]);
    int world_rank, world_size;
    // ---

    // TODO: MPI init
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    long long int tosses_num = tosses / world_size;
    unsigned long int count = 0;
    const int n = world_size;
    unsigned long int recv_ary[n];
    srand(time(NULL)*world_rank);
    //
    float x,y;
    for(int i = 0;i < tosses_num;i++){
	x = (float)rand()/(RAND_MAX) *2 -1;
	y = (float)rand()/(RAND_MAX) *2 -1;
	if(x*x + y*y <= 1.0){
		count++;
	}
    }
    // TODO: use MPI_Gather
    MPI_Gather(&count,1, MPI_UNSIGNED_LONG, &recv_ary, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    if (world_rank == 0)
    {
        // TODO: PI result
	count = 0;
	for(int i = 0;i<world_size;i++){
		count += recv_ary[i];
	}
	pi_result = 4*(float)count/tosses;
        // --- DON'T TOUCH ---
        double end_time = MPI_Wtime();
	printf("%lf\n", pi_result);
        printf("MPI running time: %lf Seconds\n", end_time - start_time);
        // ---
    }
    
    MPI_Finalize();
    return 0;
}
