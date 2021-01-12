#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/types.h>
#include <unistd.h>
#include <iostream>
using namespace std;
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
    unsigned long int count = 0;
    const int n = world_size;
    unsigned long int cnt_ary[n];
    long long int tosses_num = tosses / world_size;
    MPI_Request req[n];
    MPI_Status sta[n];
    srand(time(NULL)*world_rank);
    //
    //
    float x,y;
    for(int i = 0;i < tosses_num;i++){
	x = (float)rand() / (RAND_MAX) * 2 - 1;
	y = (float)rand() / (RAND_MAX) * 2 - 1;
	if(x*x + y*y <= 1.0){
		count++;
	}
    }
    if (world_rank == 0)
    {
        // TODO: non-blocking MPI communication.
        // Use MPI_Irecv, MPI_Wait or MPI_Waitall.
	for(int i = 1;i < world_size;i++){
		MPI_Irecv(&cnt_ary[i], 1, MPI_UNSIGNED_LONG, i, 0,MPI_COMM_WORLD,&req[i-1]);
	}
        for(int i = 1;i<world_size;i++){
		MPI_Wait(&req[i-1],&sta[i-1]);
		count += cnt_ary[i];
	}
    }
    else if(world_rank > 0)
    {
        // TODO: MPI workers
	MPI_Send(&count, 1, MPI_UNSIGNED_LONG, 0,0,MPI_COMM_WORLD);
    }
    if (world_rank == 0)
    {
        // --- DON'T TOUCH ---
        // TODO: PI result
	pi_result = 4*(float)count / tosses;
        double end_time = MPI_Wtime();
        printf("%lf\n", pi_result);
        printf("MPI running time: %lf Seconds\n", end_time - start_time);
        // ---
    }
    MPI_Finalize();
    return 0;
}
