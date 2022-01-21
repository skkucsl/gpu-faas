/**
 * covariance.cu: This file is part of the PolyBench/GPU 1.0 test suite.
 *
 *
 * Contact: Scott Grauer-Gray <sgrauerg@gmail.com>
 * Will Killian <killian@udel.edu>
 * Louis-Noel Pouchet <pouchet@cse.ohio-state.edu>
 * Web address: http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <unistd.h>
#include <sys/time.h>
#include <cuda.h>
#include <pthread.h>

#define POLYBENCH_TIME 1

#include "covariance.cuh"
#include <polybench.h>
#include <polybenchUtilFuncts.h>

//define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 1.05

#define GPU_DEVICE 0

#define sqrt_of_array_cell(x,j) sqrt(x[j])

#define FLOAT_N 3214212.01
#define EPS 0.005

//#define RUN_ON_CPU

// hdi=host_data_init, kdi=kernel-related_data_init gset=gpu_set, 
clock_t start, end;
double t_hdi, t_gset, t_malloc, t_write, t_kdi, t_kernel, t_read, t_clear;


void init_arrays(int m, int n, DATA_TYPE POLYBENCH_2D(data,M,N,m,n))
{
	int i, j;

	for (i = 0; i < m; i++)
	{
		for (j = 0; j < n; j++)
		{
			data[i][j] = ((DATA_TYPE) i*j) / M;
		}
	}
}


void covariance(int m, int n, DATA_TYPE POLYBENCH_2D(data,M,N,m,n), DATA_TYPE POLYBENCH_2D(symmat,M,M,m,m), DATA_TYPE POLYBENCH_1D(mean,M,m))
{
	int i, j, j1,j2;

  	/* Determine mean of column vectors of input data matrix */
	for (j = 0; j < _PB_M; j++)
	{
		mean[j] = 0.0;
		for (i = 0; i < _PB_N; i++)
		{
        		mean[j] += data[i][j];
		}
		mean[j] /= FLOAT_N;
	}

  	/* Center the column vectors. */
	for (i = 0; i < _PB_N; i++)
	{
		for (j = 0; j < _PB_M; j++)
		{
			data[i][j] -= mean[j];
		}
	}

  	/* Calculate the m * m covariance matrix. */
	for (j1 = 0; j1 < _PB_M; j1++)
	{
		for (j2 = j1; j2 < _PB_M; j2++)
     		{
       		symmat[j1][j2] = 0.0;
			for (i = 0; i < _PB_N; i++)
			{
				symmat[j1][j2] += data[i][j1] * data[i][j2];
			}
        		symmat[j2][j1] = symmat[j1][j2];
      		}
	}
}


void compareResults(int m, int n, DATA_TYPE POLYBENCH_2D(symmat,M,M,m,m), DATA_TYPE POLYBENCH_2D(symmat_outputFromGpu,M,M,m,m))
{
	int i,j,fail;
	fail = 0;

	for (i=0; i < m; i++)
	{
		for (j=0; j < n; j++)
		{
			if (percentDiff(symmat[i][j], symmat_outputFromGpu[i][j]) > PERCENT_DIFF_ERROR_THRESHOLD)
			{
				fail++;
			}			
		}
	}
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


__global__ void mean_kernel(int m, int n, DATA_TYPE *mean, DATA_TYPE *data)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;

	if (j < _PB_M)
	{
		mean[j] = 0.0;

		int i;
		for(i = 0; i < _PB_N; i++)
		{
			mean[j] += data[i * M + j];
		}
		mean[j] /= (DATA_TYPE)FLOAT_N;
	}
}


__global__ void reduce_kernel(int m, int n, DATA_TYPE *mean, DATA_TYPE *data)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;
		
	if ((i < _PB_N) && (j < _PB_M))
	{
		data[i * M + j] -= mean[j];	
	}
}


__global__ void covar_kernel(int m, int n, DATA_TYPE *symmat, DATA_TYPE *data)
{
	int j1 = blockIdx.x * blockDim.x + threadIdx.x;
	int i, j2;

	if (j1 < _PB_M)
	{
		for (j2 = j1; j2 < _PB_M; j2++)
		{		
			symmat[j1*M + j2] = 0.0;
			for(i = 0; i < _PB_N; i++)
			{
				symmat[j1 * M + j2] += data[i * M + j1] * data[i * M + j2];
			}
			symmat[j2 * M + j1] = symmat[j1 * M + j2];
		}
	}
}


void covarianceCuda(int m, int n, DATA_TYPE POLYBENCH_2D(data,M,N,m,n), DATA_TYPE POLYBENCH_2D(symmat,M,M,m,m), DATA_TYPE POLYBENCH_1D(mean,M,m), 
		DATA_TYPE POLYBENCH_2D(symmat_outputFromGpu,M,M,m,m))
{
	DATA_TYPE *data_gpu;
	DATA_TYPE *mean_gpu;
	DATA_TYPE *symmat_gpu;

	start = clock();
	cudaMalloc((void **)&data_gpu, sizeof(DATA_TYPE) * M * N);
	cudaMalloc((void **)&symmat_gpu, sizeof(DATA_TYPE) * M * M);
	cudaMalloc((void **)&mean_gpu, sizeof(DATA_TYPE) * M);
	end = clock();
	t_malloc = (double)(end-start) / CLOCKS_PER_SEC;

	start = clock();
	cudaMemcpy(data_gpu, data, sizeof(DATA_TYPE) * M * N, cudaMemcpyHostToDevice);
	cudaMemcpy(symmat_gpu, symmat, sizeof(DATA_TYPE) * M * M, cudaMemcpyHostToDevice);
	cudaMemcpy(mean_gpu, mean, sizeof(DATA_TYPE) * M, cudaMemcpyHostToDevice);
	end = clock();
	t_write = (double)(end-start) / CLOCKS_PER_SEC;
		
	start = clock();
	
	dim3 block1(DIM_THREAD_BLOCK_KERNEL_1_X, DIM_THREAD_BLOCK_KERNEL_1_Y);
	dim3 grid1((size_t)(ceil((float)M) / ((float)DIM_THREAD_BLOCK_KERNEL_1_X)), 1);
	
	dim3 block2(DIM_THREAD_BLOCK_KERNEL_2_X, DIM_THREAD_BLOCK_KERNEL_2_Y);
	dim3 grid2((size_t)(ceil((float)M) / ((float)DIM_THREAD_BLOCK_KERNEL_2_X)), (size_t)(ceil((float)N) / ((float)DIM_THREAD_BLOCK_KERNEL_2_X)));
	
	dim3 block3(DIM_THREAD_BLOCK_KERNEL_3_X, DIM_THREAD_BLOCK_KERNEL_3_Y);
	dim3 grid3((size_t)(ceil((float)M) / ((float)DIM_THREAD_BLOCK_KERNEL_3_X)), 1);
	end = clock();
	t_kdi = (double)(end-start) / CLOCKS_PER_SEC;

	/* Start timer. */
  	//polybench_start_instruments;
	start = clock();

	mean_kernel<<<grid1, block1>>>(m,n,mean_gpu,data_gpu);
	cudaThreadSynchronize();
	reduce_kernel<<<grid2, block2>>>(m,n,mean_gpu,data_gpu);
	cudaThreadSynchronize();
	covar_kernel<<<grid3, block3>>>(m,n,symmat_gpu,data_gpu);
	cudaThreadSynchronize();
	
	/* Stop and print timer. */
	//printf("GPU Time in seconds:\n");
  	//polybench_stop_instruments;
 	//polybench_print_instruments;
	end = clock();
	t_kernel = (double)(end-start) / CLOCKS_PER_SEC;

	start = clock();

	cudaMemcpy(symmat_outputFromGpu, symmat_gpu, sizeof(DATA_TYPE) * M * N, cudaMemcpyDeviceToHost);
	end = clock();
	t_read = (double)(end-start) / CLOCKS_PER_SEC;
	
	start = clock();
	cudaFree(data_gpu);
	cudaFree(symmat_gpu);
	cudaFree(mean_gpu);
	end = clock();
	t_clear = (double)(end-start) / CLOCKS_PER_SEC;
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int m, DATA_TYPE POLYBENCH_2D(symmat,M,M,m,m))
{
  int i, j;

  for (i = 0; i < m; i++)
    for (j = 0; j < m; j++) {
      fprintf (stderr, DATA_PRINTF_MODIFIER, symmat[i][j]);
      if ((i * m + j) % 20 == 0) fprintf (stderr, "\n");
    }
  fprintf (stderr, "\n");
}


void *poly_main(void *x)
{
	start = clock();
	
	int m = M;
	int n = N;

	POLYBENCH_2D_ARRAY_DECL(data,DATA_TYPE,M,N,m,n);
	POLYBENCH_2D_ARRAY_DECL(symmat,DATA_TYPE,M,M,m,m);
	POLYBENCH_1D_ARRAY_DECL(mean,DATA_TYPE,M,m);
	POLYBENCH_2D_ARRAY_DECL(symmat_outputFromGpu,DATA_TYPE,M,M,m,m);	

	init_arrays(m, n, POLYBENCH_ARRAY(data));

	end = clock();
	t_hdi += (double)(end-start) / CLOCKS_PER_SEC;

	start = clock();
    
	GPU_argv_init();

	end = clock();
	t_gset += (double)(end-start) / CLOCKS_PER_SEC;

	covarianceCuda(m, n, POLYBENCH_ARRAY(data), POLYBENCH_ARRAY(symmat), POLYBENCH_ARRAY(mean), POLYBENCH_ARRAY(symmat_outputFromGpu));
	

	#ifdef RUN_ON_CPU

		/* Start timer. */
	  	polybench_start_instruments;

		covariance(m, n, POLYBENCH_ARRAY(data), POLYBENCH_ARRAY(symmat), POLYBENCH_ARRAY(mean));

		/* Stop and print timer. */
		printf("CPU Time in seconds:\n");
	  	polybench_stop_instruments;
	 	polybench_print_instruments;

		compareResults(m, n, POLYBENCH_ARRAY(symmat), POLYBENCH_ARRAY(symmat_outputFromGpu));

	#else //prevent dead code elimination

		//polybench_prevent_dce(print_array(m, POLYBENCH_ARRAY(symmat_outputFromGpu)));

	#endif //RUN_ON_CPU

	start = clock();

	POLYBENCH_FREE_ARRAY(data);
	POLYBENCH_FREE_ARRAY(symmat);
	POLYBENCH_FREE_ARRAY(mean);
	POLYBENCH_FREE_ARRAY(symmat_outputFromGpu);	

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