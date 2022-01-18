#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <time.h>
#include <pthread.h>

#define arraySize 1000

int a[arraySize];
int b[arraySize];
int c[arraySize];
int d[arraySize];

int *dev_a = 0;
int *dev_b = 0;
int *dev_c = 0;
int *dev_a1 = 0;
int *dev_b1 = 0;
int *dev_c1 = 0;
int *dmp = 0;

struct HUGE {
	int h[500000];
};

__global__ void addKernel( int *c, const int *a, const int *b )
{
    int i = threadIdx.x;

	if( i < arraySize )
		c[i] = a[i] + b[i];
}

void *add_thread(void *data){
    clock_t start, end;
    int x;
/*
	start = clock();
        cudaMalloc((void**)&dev_a, arraySize * sizeof(struct HUGE));
	end = clock();
	printf("worker first time %lf\n",(double)(end-start)/CLOCKS_PER_SEC);

        start = clock();
	cudaMalloc((void**)&dev_b, arraySize * sizeof(struct HUGE));
        end = clock();
        printf("worker second time %lf\n",(double)(end-start)/CLOCKS_PER_SEC);


        // copy the arrays 'a' and 'b' to the GPU
        cudaMemcpy(dev_a, a, arraySize * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_b, b, arraySize * sizeof(int), cudaMemcpyHostToDevice);

        addKernel<<<1, arraySize>>>(dev_c, dev_a, dev_b);
        cudaDeviceSynchronize();

        // copy the array 'c' back from the GPU to the CPU
        cudaMemcpy(c, dev_c, arraySize * sizeof(int), cudaMemcpyDeviceToHost);

        printf( "%d + %d = %d\n", a[126], b[126], c[126] );
	*/
    start = clock();
    cudaMalloc((void**)&dev_c1, arraySize * sizeof(struct HUGE));
    //end = clock();
    cudaMalloc((void**)&dev_a1, arraySize * sizeof(struct HUGE));
    cudaMalloc((void**)&dev_b1, arraySize * sizeof(struct HUGE));
    end = clock();
    printf("worker pid:%d, time %lf\n",(int)getpid(), (double)(end-start)/CLOCKS_PER_SEC);

    // copy the arrays 'a' and 'b' to the GPU
    cudaMemcpy(dev_a1, a, arraySize * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b1, b, arraySize * sizeof(int), cudaMemcpyHostToDevice);

    addKernel<<<1, arraySize>>>(dev_c1, dev_a1, dev_b1);
    cudaDeviceSynchronize();

    // copy the array 'c' back from the GPU to the CPU
    cudaMemcpy(d, dev_c1, arraySize * sizeof(int), cudaMemcpyDeviceToHost);

    printf( "Worker %d + %d = %d\n", a[126], b[126], d[126] );

    scanf("%d", &x);
}

int main()
{
    int tid, status;
    clock_t start, end;
    pthread_t thread_id;

    // fill the arrays 'a' and 'b' on the CPU
    for( int i = 0 ; i < arraySize ; i++ ) {
	a[i] = i;
 	b[i] = i;
    }
    printf("pid %d\n",(int)getpid());
    int x;
    scanf("%d",&x);

    cudaSetDevice(0);
    cudaMalloc((void**)&dmp, sizeof(int));
    
    scanf("%d",&x);
    tid = pthread_create(&thread_id, NULL, add_thread, NULL);

    usleep(5000000);
    scanf("%d",&x);
/*
    start = clock();
    cudaMalloc((void**)&dev_c, arraySize * sizeof(struct HUGE));
    end = clock();
    printf("master first time %lf\n",(double)(end-start)/CLOCKS_PER_SEC);

    start = clock();
    cudaMalloc((void**)&dev_d, arraySize * sizeof(struct HUGE));
    end = clock();
    printf("master second time %lf\n",(double)(end-start)/CLOCKS_PER_SEC);
*/ 
    start = clock();
    cudaMalloc((void**)&dev_c, arraySize * sizeof(struct HUGE));
    cudaMalloc((void**)&dev_a, arraySize * sizeof(struct HUGE));
    cudaMalloc((void**)&dev_b, arraySize * sizeof(struct HUGE));
    end = clock();
    printf("master pid:%d, time %lf\n",(int)getpid(), (double)(end-start)/CLOCKS_PER_SEC);

    // copy the arrays 'a' and 'b' to the GPU
    cudaMemcpy(dev_a, a, arraySize * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, arraySize * sizeof(int), cudaMemcpyHostToDevice);

    addKernel<<<1, arraySize>>>(dev_c, dev_a, dev_b);
    cudaDeviceSynchronize();

    // copy the array 'c' back from the GPU to the CPU
    cudaMemcpy(c, dev_c, arraySize * sizeof(int), cudaMemcpyDeviceToHost);

    printf( "Master %d + %d = %d\n", a[126], b[126], c[126] );

    //tid = pthread_create(&thread_id, NULL, add_thread, NULL);

    pthread_join(thread_id, (void **)&status);
    // display the results
    
    // free the memory allocated on the GPU
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return 0;
}
