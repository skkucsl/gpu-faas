#include <stdio.h>
#include <time.h>

int main(){
	int *x;
	int *y;

	clock_t start, end;

	start = clock();
	cudaMalloc((void**)&x, sizeof(int));
	end = clock();
	printf("1st Malloc %lf\n", (double)(end-start)/CLOCKS_PER_SEC);
	cudaFree(x);

	start = clock();
	cudaMallocManaged((void**)&y, sizeof(int));
	end = clock();
	printf("2nd Malloc (UVM) %lf\n", (double)(end-start)/CLOCKS_PER_SEC);
	cudaFree(y);

}
