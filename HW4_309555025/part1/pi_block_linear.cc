#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <time.h>
#include <sys/types.h>
#include <unistd.h>
//
int main(int argc, char **argv)
{
    // --- DON'T TOUCH ---
    MPI_Init(&argc, &argv);
    double start_time = MPI_Wtime();
    double pi_result;
    long long int tosses = atoi(argv[1]);
    unsigned long int count = 0;
    int world_rank, world_size;
    // ---

    // TODO: init MPI
    // get workd_rank
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    long long int tosses_num = tosses / world_size;
    srand(time(NULL)*world_rank);
    if (world_rank > 0)
    {
        // TODO: handle workers
	float x,y;
	for(int i = 0;i < tosses_num;i++){
		x = (float) rand() / (RAND_MAX) * 2  -1;
		y = (float) rand() / (RAND_MAX) * 2  -1;
		if(x*x + y*y <= 1.0){
			count++;
		}
	}
    	MPI_Send(&count, 1, MPI_UNSIGNED_LONG, 0, 0, MPI_COMM_WORLD);
    }
    else if (world_rank == 0)
    {
        // TODO: master
	for(int i = 1;i<world_size;i++){
		unsigned long int cnt;
		MPI_Status status;
		MPI_Recv(&cnt, 1, MPI_UNSIGNED_LONG, i, 0, MPI_COMM_WORLD, &status);
		count += cnt;
	}
	float x,y;
	for(int i = 0;i < tosses_num;i++){
		x = (float) rand() / (RAND_MAX) * 2 + -1;
		y = (float) rand() / (RAND_MAX) * 2 + -1;
		if(x*x + y*y <= 1.0){
			count++;
		}
	}
    }

    if (world_rank == 0)
    {
        // TODO: process PI result
	pi_result = 4*(float)count / tosses; 
        // --- DON'T TOUCH ---
        double end_time = MPI_Wtime();
        printf("%lf\n", pi_result);
        printf("MPI running time: %lf Seconds\n", end_time - start_time);
        // ---
    }
    MPI_Finalize();
    return 0;
}
