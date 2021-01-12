#include <iostream>
#include <stdlib.h>
#include <pthread.h>
#include <unistd.h>
#include <time.h>
#include <stdint.h>
#include <climits>
using namespace std;
long long int num_in_circle = 0;
pthread_mutex_t mutex;

void *sub_job(void *num){
	
	long long int num_of_tosses = *(long long int*) num;
	long long int sum = 0;
	unsigned int seed = clock();
	for(int toss = 0; toss < num_of_tosses; toss++ ){
		double x = ((double)rand_r(&seed) / (double)RAND_MAX )*2.0 - 1.0;
		double y = ((double)rand_r(&seed) / (double)RAND_MAX )*2.0 - 1.0;
		if(x * x + y * y <= 1)
			sum++;
	}
	//cout << sum << endl;
	pthread_mutex_lock(&mutex); 
    num_in_circle += sum;
    pthread_mutex_unlock(&mutex); 

	pthread_exit(NULL);
}
int main(int argc,char** argv){
	// get arguments & some setting
	int CPU_cores;
	long long int num_of_tosses;
	if(argc >= 2){
		CPU_cores = atoi(argv[1]);
	}
	else{
		CPU_cores = 1;
	}
	if(argc >= 3){
		num_of_tosses = atol(argv[2]);
	}
	else{
		num_of_tosses = 100000000;
	}
	
	// create threads
	pthread_t *thrds_ary;
	thrds_ary = (pthread_t *)malloc(CPU_cores*sizeof(pthread_t));
	// create mutex
	pthread_mutex_init(&mutex, NULL);
	// determine job number to each thread
	long long int num_chunk = num_of_tosses / CPU_cores;
	// threads do their jobs
	
	for(int i = 0;i < CPU_cores;i++){
		if(pthread_create(&thrds_ary[i], NULL, sub_job, (void*)&num_chunk) != 0){	
			cerr << "pthread error\n";
		}
	}
	// join threads
	for(int i = 0;i < CPU_cores;i++){
		pthread_join(thrds_ary[i],NULL);
	}
	double pi_estimate = 4.0 * double(num_in_circle)/((double)num_of_tosses);
	cout << pi_estimate << endl;
	pthread_mutex_destroy(&mutex);
	free(thrds_ary);
	return 0;
}