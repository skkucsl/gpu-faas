#include <stdio.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <time.h>

#define arraySize 1024
__global__
void test() {
	// nothing
}

__global__ void addKernel( int *c, const int *a, const int *b )
{
    int i = threadIdx.x;

        if( i < arraySize )
                c[i] = a[i] + b[i];
}

double init_time(){
	int *x = 0; 
	clock_t start, end;

	int a[arraySize];
	int b[arraySize];
	int c[arraySize];

	int *dev_a = 0;
	int *dev_b = 0;
	int *dev_c = 0;

	double ret;

	start = clock();
	cudaSetDevice(0);
	cudaMalloc((void**)&x, sizeof(int));
	end = clock();

	//printf("INIT TIME = %lf\n", (double)(end-start)/CLOCKS_PER_SEC);

	for (int i = 0; i < arraySize; i++){
		a[i] = i;
		b[i] = i;
	}

	cudaMalloc((void**)&dev_c, arraySize * sizeof(int));
	cudaMalloc((void**)&dev_a, arraySize * sizeof(int));
	cudaMalloc((void**)&dev_b, arraySize * sizeof(int));

	cudaMemcpy(dev_a, a, arraySize * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b, arraySize * sizeof(int), cudaMemcpyHostToDevice);

	addKernel<<<1, arraySize>>>(dev_c, dev_a, dev_b);
	cudaMemcpy(c, dev_c, arraySize * sizeof(int), cudaMemcpyDeviceToHost);

	//printf("%d + %d = %d\n", a[278], b[278], c[278]);
	//printf("{ \"Init\": %lf, \"res\": \"%d + %d = %d\" }", (double)(end-start)/CLOCKS_PER_SEC, a[278], b[278], c[278]);

	cudaFree(dev_c);
	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(x);

	ret = (double)(end-start)/CLOCKS_PER_SEC;
	printf("RETRUN %lf\n", ret);

	return ret;
}
