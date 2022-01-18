#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <time.h>
#include <pthread.h>
#include <sched.h>

#define arraySize 1000

int a[arraySize];
int b[arraySize];
int c[arraySize];
int d[arraySize];

int *dmp = 0;
int *dev_a = 0;
int *dev_b = 0;
int *dev_c = 0;
int *dev_a1 = 0;
int *dev_b1 = 0;
int *dev_c1 = 0;

struct HUGE {
	int h[500000];
};

__global__ void addKernel( int *c, const int *a, const int *b )
{
    int i = threadIdx.x;

	if( i < arraySize )
		c[i] = a[i] + b[i];
}

static int add_thread(void *data){
	clock_t start, end;
	int x;
/*
	scanf("%d",&x);

	start = clock();
        cudaMalloc((void**)&dev_a, 2*arraySize * sizeof(struct HUGE));
	end = clock();
	printf("worker first time %lf\n",(double)(end-start)/CLOCKS_PER_SEC);

        start = clock();
	cudaMalloc((void**)&dev_b, 2*arraySize * sizeof(struct HUGE));
        end = clock();
        printf("worker second time %lf\n",(double)(end-start)/CLOCKS_PER_SEC);


        // copy the arrays 'a' and 'b' to the GPU
        cudaMemcpy(dev_a, a, arraySize * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_b, b, arraySize * sizeof(int), cudaMemcpyHostToDevice);

        addKernel<<<1, arraySize>>>(dev_c, dev_a, dev_b);
        cudaDeviceSynchronize();

        // copy the array 'c' back from the GPU to the CPU
        cudaMemcpy(c, dev_c, arraySize * sizeof(int), cudaMemcpyDeviceToHost);

        printf( "pid: %d, %d + %d = %d\n", (int)getpid(), a[126], b[126], c[126] );
	*/
    //usleep(1000000);
    
    start = clock();
    cudaMalloc((void**)&dev_c1, arraySize * sizeof(struct HUGE));
    cudaMalloc((void**)&dev_a1, arraySize * sizeof(struct HUGE));
    cudaMalloc((void**)&dev_b1, arraySize * sizeof(struct HUGE));
    end = clock();
    printf("child pid: %d, time %lf\n",(int)getpid(),(double)(end-start)/CLOCKS_PER_SEC);
    //usleep(10000000);
    // copy the arrays 'a' and 'b' to the GPU
    cudaMemcpy(dev_a1, a, arraySize * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b1, b, arraySize * sizeof(int), cudaMemcpyHostToDevice);

    addKernel<<<1, arraySize>>>(dev_c1, dev_a1, dev_b1);
    cudaDeviceSynchronize();

    // copy the array 'c' back from the GPU to the CPU
    cudaMemcpy(d, dev_c1, arraySize * sizeof(int), cudaMemcpyDeviceToHost);

    // display the results
    printf("Child %d + %d = %d\n", a[126], b[126], d[126]);

    // free the memory allocated on the GPU
    cudaFree(dev_c1);
    cudaFree(dev_a1);
    cudaFree(dev_b1);

	return 0;
}

int main()
{
    int tid, status;
    clock_t start, end;
    pthread_t thread_id;

    int x;

    const int STACK_SIZE = 65536;
    char* stack = (char*)malloc(STACK_SIZE);
    if (!stack) {
      perror("malloc");
      exit(1);
    }

    // fill the arrays 'a' and 'b' on the CPU
    for( int i = 0 ; i < arraySize ; i++ ) {
	a[i] = i;
 	b[i] = i;
    }
    cudaSetDevice(0);
    cudaMalloc((void**)&dmp, sizeof(int));

    char buf[100];
    strcpy(buf, "hello from parent");
    //CLONE_VM|CLONE_FS|CLONE_FILES|CLONE_SIGHAND|CLONE_THREAD|CLONE_SYSVSEM|SIGCHLD
    if (clone(add_thread, stack + STACK_SIZE, CLONE_VM|CLONE_SIGHAND|CLONE_THREAD, buf) == -1) {
      exit(1);
    }
    
    usleep(1000000);
/*    
    start = clock();
    cudaMalloc((void**)&dev_c, arraySize * sizeof(struct HUGE));
    end = clock();
    printf("pid: %d, master first time %lf\n",(int)getpid(), (double)(end-start)/CLOCKS_PER_SEC);

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
    printf("parent pid: %d, time %lf\n", (int)getpid(), (double)(end-start)/CLOCKS_PER_SEC);

    // usleep(10000000);
    // copy the arrays 'a' and 'b' to the GPU
    cudaMemcpy(dev_a, a, arraySize * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, arraySize * sizeof(int), cudaMemcpyHostToDevice);

    addKernel<<<1, arraySize>>>(dev_c, dev_a, dev_b);
    cudaDeviceSynchronize();

    // copy the array 'c' back from the GPU to the CPU
    cudaMemcpy(c, dev_c, arraySize * sizeof(int), cudaMemcpyDeviceToHost);

    // display the results
    printf("Parent %d + %d = %d\n", a[126], b[126], c[126]);

    usleep(22000000);
    
    // free the memory allocated on the GPU
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return 0;
}
