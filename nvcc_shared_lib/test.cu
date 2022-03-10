#include <stdio.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <time.h>

__global__
void test() {
	// nothing
}

int wait_time(int n){
	int *x = 0; 
	clock_t start, end;

	start = clock();
	cudaSetDevice(0);
	cudaMalloc((void**)&x, sizeof(int));
	end = clock();

	printf("Wait %d seconds\n", n);
	usleep(n*1000000);
	printf("INIT TIME = %lf\n", (double)(end-start)/CLOCKS_PER_SEC);

	return n;
}
