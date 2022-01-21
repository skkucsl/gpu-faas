/**
 * gramschmidt.cu: This file is part of the PolyBench/GPU 1.0 test suite.
 *
 *
 * Contact: Scott Grauer-Gray <sgrauerg@gmail.com>
 * Will Killian <killian@udel.edu>
 * Louis-Noel Pouchet <pouchet@cse.ohio-state.edu>
 * Web address: http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU
 */

#include <unistd.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <cuda.h>
#include <pthread.h>

#define POLYBENCH_TIME 1

#include "gramschmidt.cuh"
#include <polybench.h>
#include <polybenchUtilFuncts.h>

//define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.05

#define GPU_DEVICE 0

//#define RUN_ON_CPU

// hdi=host_data_init, kdi=kernel-related_data_init gset=gpu_set, 
clock_t start, end;
double t_hdi, t_gset, t_malloc, t_write, t_kdi, t_kernel, t_read, t_clear;


void gramschmidt(int ni, int nj, DATA_TYPE POLYBENCH_2D(A,NI,NJ,ni,nj), DATA_TYPE POLYBENCH_2D(R,NJ,NJ,nj,nj), DATA_TYPE POLYBENCH_2D(Q,NI,NJ,ni,nj))
{
	int i,j,k;
	DATA_TYPE nrm;
	for (k = 0; k < _PB_NJ; k++)
	{
		nrm = 0;
		for (i = 0; i < _PB_NI; i++)
		{
			nrm += A[i][k] * A[i][k];
		}
		
		R[k][k] = sqrt(nrm);
		for (i = 0; i < _PB_NI; i++)
		{
			Q[i][k] = A[i][k] / R[k][k];
		}
		
		for (j = k + 1; j < _PB_NJ; j++)
		{
			R[k][j] = 0;
			for (i = 0; i < _PB_NI; i++)
			{
				R[k][j] += Q[i][k] * A[i][j];
			}
			for (i = 0; i < _PB_NI; i++)
			{
				A[i][j] = A[i][j] - Q[i][k] * R[k][j];
			}
		}
	}
}

/* Array initialization. */
void init_array(int ni, int nj,
		DATA_TYPE POLYBENCH_2D(A,NI,NJ,ni,nj),
		DATA_TYPE POLYBENCH_2D(R,NJ,NJ,nj,nj),
		DATA_TYPE POLYBENCH_2D(Q,NI,NJ,ni,nj))
{
	int i, j;

	for (i = 0; i < ni; i++)
	{
		for (j = 0; j < nj; j++) 
		{
			A[i][j] = ((DATA_TYPE) i*j) / ni;
			Q[i][j] = ((DATA_TYPE) i*(j+1)) / nj;
		}
	}

	for (i = 0; i < nj; i++)
	{
		for (j = 0; j < nj; j++)
		{
			R[i][j] = ((DATA_TYPE) i*(j+2)) / nj;
		}
	}
}

void compareResults(int ni, int nj, DATA_TYPE POLYBENCH_2D(A,NI,NJ,ni,nj), DATA_TYPE POLYBENCH_2D(A_outputFromGpu,NI,NJ,ni,nj))
{
	int i, j, fail;
	fail = 0;

	for (i=0; i < ni; i++) 
	{
		for (j=0; j < nj; j++) 
		{
			if (percentDiff(A[i][j], A_outputFromGpu[i][j]) > PERCENT_DIFF_ERROR_THRESHOLD) 
			{				
				fail++;
			}
		}
	}
	
	// Print results
	printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", PERCENT_DIFF_ERROR_THRESHOLD, fail);
}


void GPU_argv_init()
{
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, GPU_DEVICE);
	//printf("setting device %d with name %s\n",GPU_DEVICE,deviceProp.name);
	cudaSetDevice( GPU_DEVICE );	
	return;
}


__global__ void gramschmidt_kernel1(int ni, int nj, DATA_TYPE *a, DATA_TYPE *r, DATA_TYPE *q, int k)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	if(tid==0)
	{
		DATA_TYPE nrm = 0.0;
		int i;
		for (i = 0; i < _PB_NI; i++)
		{
			nrm += a[i * NJ + k] * a[i * NJ + k];
		}
      		r[k * NJ + k] = sqrt(nrm);
	}
}


__global__ void gramschmidt_kernel2(int ni, int nj, DATA_TYPE *a, DATA_TYPE *r, DATA_TYPE *q, int k)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (i < _PB_NI)
	{	
		q[i * NJ + k] = a[i * NJ + k] / r[k * NJ + k];
	}
}


__global__ void gramschmidt_kernel3(int ni, int nj, DATA_TYPE *a, DATA_TYPE *r, DATA_TYPE *q, int k)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;

	if ((j > k) && (j < _PB_NJ))
	{
		r[k*NJ + j] = 0.0;

		int i;
		for (i = 0; i < _PB_NI; i++)
		{
			r[k*NJ + j] += q[i*NJ + k] * a[i*NJ + j];
		}
		
		for (i = 0; i < _PB_NI; i++)
		{
			a[i*NJ + j] -= q[i*NJ + k] * r[k*NJ + j];
		}
	}
}


void gramschmidtCuda(int ni, int nj, DATA_TYPE POLYBENCH_2D(A,NI,NJ,ni,nj), DATA_TYPE POLYBENCH_2D(R,NJ,NJ,nj,nj), DATA_TYPE POLYBENCH_2D(Q,NI,NJ,ni,nj), DATA_TYPE POLYBENCH_2D(A_outputFromGpu,NI,NJ,ni,nj))
{
	start = clock();
	dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
	dim3 gridKernel1(1, 1);
	dim3 gridKernel2((size_t)ceil(((float)NJ) / ((float)DIM_THREAD_BLOCK_X)), 1);
	dim3 gridKernel3((size_t)ceil(((float)NJ) / ((float)DIM_THREAD_BLOCK_X)), 1);
	end = clock();
	t_kdi = (double)(end-start) / CLOCKS_PER_SEC;
	
	DATA_TYPE *A_gpu;
	DATA_TYPE *R_gpu;
	DATA_TYPE *Q_gpu;

	start = clock();
	cudaMalloc((void **)&A_gpu, sizeof(DATA_TYPE) * NI * NJ);
	cudaMalloc((void **)&R_gpu, sizeof(DATA_TYPE) * NJ * NJ);
	cudaMalloc((void **)&Q_gpu, sizeof(DATA_TYPE) * NI * NJ);
	end = clock();
	t_malloc = (double)(end-start) / CLOCKS_PER_SEC;

	start = clock();
	cudaMemcpy(A_gpu, A, sizeof(DATA_TYPE) * NI * NJ, cudaMemcpyHostToDevice);
	end = clock();
	t_write = (double)(end-start) / CLOCKS_PER_SEC;
		
	start = clock();
	
	/* Start timer. */
  	//polybench_start_instruments;
	int k;
	for (k = 0; k < _PB_NJ; k++)
	{
		gramschmidt_kernel1<<<gridKernel1,block>>>(ni, nj, A_gpu, R_gpu, Q_gpu, k);
		cudaThreadSynchronize();
		gramschmidt_kernel2<<<gridKernel2,block>>>(ni, nj, A_gpu, R_gpu, Q_gpu, k);
		cudaThreadSynchronize();
		gramschmidt_kernel3<<<gridKernel3,block>>>(ni, nj, A_gpu, R_gpu, Q_gpu, k);
		cudaThreadSynchronize();
	}
	/* Stop and print timer. */
	//printf("GPU Time in seconds:\n");
  	//polybench_stop_instruments;
 	//polybench_print_instruments;
	end = clock();
	t_kernel = (double)(end-start) / CLOCKS_PER_SEC;
	
	start = clock();
	cudaMemcpy(A_outputFromGpu, A_gpu, sizeof(DATA_TYPE) * NI * NJ, cudaMemcpyDeviceToHost);   
	end = clock();
	t_read = (double)(end-start) / CLOCKS_PER_SEC;
	
	start = clock(); 

	cudaFree(A_gpu);
	cudaFree(R_gpu);
	cudaFree(Q_gpu);

	end = clock();
	t_clear = (double)(end-start) / CLOCKS_PER_SEC;
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int ni, int nj, DATA_TYPE POLYBENCH_2D(A,NI,NJ,ni,nj))
{
  int i, j;

  for (i = 0; i < ni; i++)
    for (j = 0; j < nj; j++) {
	fprintf (stderr, DATA_PRINTF_MODIFIER, A[i][j]);
	if (i % 20 == 0) fprintf (stderr, "\n");
    }

  fprintf (stderr, "\n");
}


void *poly_main(void *x)
{
	start = clock();

	/* Retrieve problem size. */
	int ni = NI;
	int nj = NJ;

	POLYBENCH_2D_ARRAY_DECL(A,DATA_TYPE,NI,NJ,ni,nj);
  	POLYBENCH_2D_ARRAY_DECL(A_outputFromGpu,DATA_TYPE,NI,NJ,ni,nj);
	POLYBENCH_2D_ARRAY_DECL(R,DATA_TYPE,NJ,NJ,nj,nj);
	POLYBENCH_2D_ARRAY_DECL(Q,DATA_TYPE,NI,NJ,ni,nj);
	
	init_array(ni, nj, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(R), POLYBENCH_ARRAY(Q));

	end = clock();
	t_hdi += (double)(end-start) / CLOCKS_PER_SEC;

	start = clock();
	
	GPU_argv_init();

	end = clock();
	t_gset += (double)(end-start) / CLOCKS_PER_SEC;

	gramschmidtCuda(ni, nj, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(R), POLYBENCH_ARRAY(Q), POLYBENCH_ARRAY(A_outputFromGpu));

	#ifdef RUN_ON_CPU
	
		/* Start timer. */
	  	polybench_start_instruments;

		gramschmidt(ni, nj, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(R), POLYBENCH_ARRAY(Q));

		printf("CPU Time in seconds:\n");
	  	polybench_stop_instruments;
	 	polybench_print_instruments;
	
		compareResults(ni, nj, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(A_outputFromGpu));
	
	#else //prevent dead code elimination

		//polybench_prevent_dce(print_array(ni, nj, POLYBENCH_ARRAY(A_outputFromGpu)));

	#endif //RUN_ON_CPU

	start = clock();

	POLYBENCH_FREE_ARRAY(A);
	POLYBENCH_FREE_ARRAY(A_outputFromGpu);
	POLYBENCH_FREE_ARRAY(R);
	POLYBENCH_FREE_ARRAY(Q);  

	end = clock();
	t_clear += (double)(end-start) / CLOCKS_PER_SEC;

	printf("{\"hdi\": %lf, \"gset\": %lf, \"malloc\": %lf, \"write\": %lf, \"kdi\": %lf, \"kernel\": %lf, \"read\": %lf, \"clear\": %lf}",
		t_hdi, t_gset, t_malloc, t_write, t_kdi, t_kernel, t_read, t_clear);

    return 0;
}

int main()
{
	int tid, status;
	int *init_mem = 0;
	pthread_t thread_id;

	//start = clock();
	cudaSetDevice(0);
	cudaMalloc((void**)&init_mem, 4096 * sizeof(int));
	//end = clock();
	//printf("Master overhead: %lf\n",(double)(end-start) / CLOCKS_PER_SEC);

	tid = pthread_create(&thread_id, NULL, poly_main, NULL);
	pthread_join(thread_id, (void**)&status);

	cudaFree(init_mem);

	return 0;
}
#include <polybench.c>