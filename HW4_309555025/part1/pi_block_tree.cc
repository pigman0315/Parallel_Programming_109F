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
    unsigned long int count = 0;
    // ---
    
    // TODO: MPI init
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    // add up count
    long long int tosses_num = tosses / world_size;
    srand(time(NULL)+world_rank*100);
    double x,y;
    for(int i = 0;i < tosses_num;i++){
	x = (double) rand() / (RAND_MAX) * 2 - 1;
	y = (double) rand() / (RAND_MAX) * 2 - 1;
	if(x*x + y*y <= 1.0){
		count++;
	}
    }
    // cout << count << endl;
    // TODO: binary tree reduction
    MPI_Status status;
    unsigned long int tmp;
    if(world_rank == 0){
	MPI_Recv(&tmp, 1, MPI_UNSIGNED_LONG, 1, 0, MPI_COMM_WORLD, &status);
	count += tmp;
    }
    else if(world_rank == 1){
	MPI_Send(&count, 1, MPI_UNSIGNED_LONG, 0,0,MPI_COMM_WORLD);
    }
    if(world_size >= 4){
	if(world_rank == 0){
		MPI_Recv(&tmp, 1, MPI_UNSIGNED_LONG, 2,0,MPI_COMM_WORLD, &status);
		count += tmp;
	}
	else if(world_rank == 2){
		MPI_Recv(&tmp, 1, MPI_UNSIGNED_LONG, 3, 0, MPI_COMM_WORLD, &status);
		count += tmp;
		MPI_Send(&count, 1, MPI_UNSIGNED_LONG, 0,0,MPI_COMM_WORLD);
	}
	else if(world_rank == 3){
		MPI_Send(&count, 1, MPI_UNSIGNED_LONG, 2,0,MPI_COMM_WORLD);
	}
    }
    if(world_size >= 8){
	if(world_rank == 0){
		MPI_Recv(&tmp, 1, MPI_UNSIGNED_LONG, 4,0,MPI_COMM_WORLD, &status);
		count += tmp;
	}
	else if(world_rank == 4){
		MPI_Recv(&tmp, 1, MPI_UNSIGNED_LONG, 5,0,MPI_COMM_WORLD, &status);
		count += tmp;
		MPI_Recv(&tmp, 1, MPI_UNSIGNED_LONG, 6,0,MPI_COMM_WORLD, &status);
		count += tmp;
		MPI_Send(&count, 1, MPI_UNSIGNED_LONG, 0,0,MPI_COMM_WORLD);
	}
	else if(world_rank == 5){
		MPI_Send(&count, 1, MPI_UNSIGNED_LONG, 4,0,MPI_COMM_WORLD);
	}
	else if(world_rank == 6){
		MPI_Recv(&tmp, 1, MPI_UNSIGNED_LONG, 7,0,MPI_COMM_WORLD, &status);
		count += tmp;
		MPI_Send(&count, 1, MPI_UNSIGNED_LONG, 4,0,MPI_COMM_WORLD);
	}
	else if(world_rank == 7){
		MPI_Send(&count, 1, MPI_UNSIGNED_LONG, 6,0,MPI_COMM_WORLD);
	}
    }
    if(world_size >= 16){
	if(world_rank == 0){
		MPI_Recv(&tmp, 1, MPI_UNSIGNED_LONG, 8,0,MPI_COMM_WORLD, &status);
		count += tmp;
	}
	else if(world_rank == 8){
		MPI_Recv(&tmp, 1, MPI_UNSIGNED_LONG, 9,0,MPI_COMM_WORLD, &status);
		count += tmp;
		MPI_Recv(&tmp, 1, MPI_UNSIGNED_LONG, 10,0,MPI_COMM_WORLD, &status);
		count += tmp;	
		MPI_Recv(&tmp, 1, MPI_UNSIGNED_LONG, 12,0,MPI_COMM_WORLD, &status);
		count += tmp;
		MPI_Send(&count, 1, MPI_UNSIGNED_LONG, 0,0,MPI_COMM_WORLD);
	}
	else if(world_rank == 9){
		MPI_Send(&count, 1, MPI_UNSIGNED_LONG, 8,0,MPI_COMM_WORLD);
	}
	else if(world_rank == 10){
		MPI_Recv(&tmp, 1, MPI_UNSIGNED_LONG, 11,0,MPI_COMM_WORLD, &status);
		count += tmp;
		MPI_Send(&count, 1, MPI_UNSIGNED_LONG, 8,0,MPI_COMM_WORLD);
	}
	else if(world_rank == 11){
		MPI_Send(&count, 1, MPI_UNSIGNED_LONG, 10,0,MPI_COMM_WORLD);
	}
	else if(world_rank == 12){
		MPI_Recv(&tmp, 1, MPI_UNSIGNED_LONG, 13,0,MPI_COMM_WORLD, &status);
		count += tmp;	
		MPI_Recv(&tmp, 1, MPI_UNSIGNED_LONG, 14,0,MPI_COMM_WORLD, &status);
		count += tmp;
		MPI_Send(&count, 1, MPI_UNSIGNED_LONG, 8,0,MPI_COMM_WORLD);
	}
	else if(world_rank == 13){
		MPI_Send(&count, 1, MPI_UNSIGNED_LONG, 12,0,MPI_COMM_WORLD);
	}
	else if(world_rank == 14){
		MPI_Recv(&tmp, 1, MPI_UNSIGNED_LONG, 15,0,MPI_COMM_WORLD, &status);
		count += tmp;
		MPI_Send(&count, 1, MPI_UNSIGNED_LONG, 12,0,MPI_COMM_WORLD);
	}
	else if(world_rank == 15){
		MPI_Send(&count, 1, MPI_UNSIGNED_LONG, 14,0,MPI_COMM_WORLD);
	}
    }
    if (world_rank == 0)
    {
        // TODO: PI result
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
