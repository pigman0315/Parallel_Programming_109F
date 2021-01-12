#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/types.h>
#include <unistd.h>
int fnz(unsigned long int *newS, unsigned long int *oldS, int size){
	int diff = 0;
	//
	for(int i = 0; i < size; i++){
		diff |= (newS[i] != oldS[i]);
	}
	if(diff){
		int res = 0;
		for(int i = 0;i < size;i++){
			if(newS[i] != 0)
				res += 1;
			oldS[i] = newS[i];
		}
		return(res == size-1);
	}
	return 0;
}
int main(int argc, char **argv)
{
    // --- DON'T TOUCH ---
    MPI_Init(&argc, &argv);
    double start_time = MPI_Wtime();
    double pi_result;
    long long int tosses = atoi(argv[1]);
    int world_rank, world_size;
    // ---

    MPI_Win win;

    // TODO: MPI init
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    long long int tosses_num = tosses / world_size;
    unsigned long int count = 0;
    unsigned long int sum = 0;
    srand(time(NULL)*world_rank);
    // calculate pi
    float x,y;
    for(int i = 0;i < tosses_num;i++){
	x = (float)rand()/(RAND_MAX)*2 - 1;
	y = (float)rand()/(RAND_MAX)*2 - 1;
	if(x*x + y*y <= 1.0){
		count++;
	}
    }
    //printf("count=%ld\n",count);
    if (world_rank == 0)
    {
        // Master
	unsigned long int *oldS = (unsigned long int *)malloc(world_size * sizeof(unsigned long int));
	unsigned long int *newS;
	MPI_Alloc_mem(world_size * sizeof(unsigned long int), MPI_INFO_NULL, &newS);
	for(int i = 0;i < world_size;i++){
		newS[i] = 0;
		oldS[i] = -1;
	}
	//
        MPI_Win_create(newS, world_size * sizeof(unsigned long int), sizeof(unsigned long int), MPI_INFO_NULL, MPI_COMM_WORLD, &win);

	int ready = 0;
	while(!ready){
		MPI_Win_lock(MPI_LOCK_SHARED, 0, 0, win);
		ready = fnz(newS, oldS, world_size);
		MPI_Win_unlock(0, win);
	}
	for(int i = 0;i < world_size;i++){
		count += newS[i];
	}
	MPI_Free_mem(newS);
	free(oldS);
	//printf("Master done\n");
    }
    else
    {
        // Workers
	MPI_Win_create(NULL,0,1,MPI_INFO_NULL, MPI_COMM_WORLD,&win);
	// Register with the master
	MPI_Win_lock(MPI_LOCK_EXCLUSIVE, 0, 0, win);
	MPI_Put(&count, 1, MPI_UNSIGNED_LONG, 0, world_rank, 1, MPI_UNSIGNED_LONG, win);
	MPI_Win_unlock(0,win);
	//printf("Worker %d finished\n", world_rank);
    }
    MPI_Win_free(&win);

    if (world_rank == 0)
    {
        // TODO: handle PI result
	//printf("final count = %ld\n",count);
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
