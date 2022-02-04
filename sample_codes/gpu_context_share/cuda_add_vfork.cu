#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <time.h>

#define arraySize 1000

__global__ void addKernel( int *c, const int *a, const int *b )
{
    int i = threadIdx.x;

	if( i < arraySize )
		c[i] = a[i] + b[i];
}

int main()
{
    int a[arraySize];
    int b[arraySize];
    int c[arraySize];
    int x;

    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;

    pid_t pid;
    clock_t start, end;

    // fill the arrays 'a' and 'b' on the CPU
    for( int i = 0 ; i < arraySize ; i++ ) {
	a[i] = i;
 	b[i] = i;
    }
    cudaSetDevice(0);
    pid = vfork();

    if (pid == 0){
    	// Add vectors in parallel.
    	// Allocate GPU buffers for three vectors (two input, one output)
	start = clock();
    	cudaMalloc((void**)&dev_c, arraySize * sizeof(int));
    	cudaMalloc((void**)&dev_a, arraySize * sizeof(int));
    	cudaMalloc((void**)&dev_b, arraySize * sizeof(int));
	end = clock();
	printf("pid: %d, child time %lf\n",(int)getpid(), (double)(end-start)/CLOCKS_PER_SEC);


    	// copy the arrays 'a' and 'b' to the GPU
    	cudaMemcpy(dev_a, a, arraySize * sizeof(int), cudaMemcpyHostToDevice);
    	cudaMemcpy(dev_b, b, arraySize * sizeof(int), cudaMemcpyHostToDevice);

	//scanf("%d",&x);

    	addKernel<<<1, arraySize>>>(dev_c, dev_a, dev_b);
    	cudaDeviceSynchronize();

	// copy the array 'c' back from the GPU to the CPU
	cudaMemcpy(c, dev_c, arraySize * sizeof(int), cudaMemcpyDeviceToHost);

        printf( "Child %d + %d = %d\n", a[126], b[126], c[126] );

	cudaFree(dev_c);
	cudaFree(dev_a);
	cudaFree(dev_b);

	return 0;
    }
    else{
	start = clock();
        cudaMalloc((void**)&dev_c, arraySize * sizeof(int));
        cudaMalloc((void**)&dev_a, arraySize * sizeof(int));
        cudaMalloc((void**)&dev_b, arraySize * sizeof(int));
	end = clock();
	printf("pid: %d, parent time %lf\n",(int)getpid(), (double)(end-start)/CLOCKS_PER_SEC);

        // copy the arrays 'a' and 'b' to the GPU
        cudaMemcpy(dev_a, a, arraySize * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_b, b, arraySize * sizeof(int), cudaMemcpyHostToDevice);

        addKernel<<<1, arraySize>>>(dev_c, dev_a, dev_b);
        cudaDeviceSynchronize();

        // copy the array 'c' back from the GPU to the CPU
        cudaMemcpy(c, dev_c, arraySize * sizeof(int), cudaMemcpyDeviceToHost);

	printf( "Parent %d + %d = %d\n", a[126], b[126], c[126] );

	// display the results
	cudaFree(dev_c);
	cudaFree(dev_a);
	cudaFree(dev_b);
    }

    // free the memory allocated on the GPU
    return 0;
}
