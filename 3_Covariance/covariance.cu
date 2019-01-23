#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <unistd.h>
#include <sys/time.h>
#include <cuda.h>

#include "header.h"

#define BLOCK_SIZE 32

//define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 1.05

#define GPU_DEVICE 0

/* Problem size */
#define M 1000
#define N 500

/* Thread block dimensions for kernel 1*/
#define DIM_THREAD_BLOCK_KERNEL_1_X 128
#define DIM_THREAD_BLOCK_KERNEL_1_Y 1

/* Thread block dimensions for kernel 2*/
#define DIM_THREAD_BLOCK_KERNEL_2_X 32
#define DIM_THREAD_BLOCK_KERNEL_2_Y 8

/* Thread block dimensions for kernel 3*/
#define DIM_THREAD_BLOCK_KERNEL_3_X 256
#define DIM_THREAD_BLOCK_KERNEL_3_Y 1

#define sqrt_of_array_cell(x,j) sqrt(x[j])

#define FLOAT_N 100000000000.0//3214212.01




void init_arrays(double* data)
{
	int i, j;

	for (i = 1; i < (M+1); i++)
	{
		for (j = 1; j < (N+1); j++)
		{
			data[i*(N+1) + j] = ((double) i*j) / M;
		}
	}
}


void covariance(double* data, double* symmat, double* mean)
{
	int i, j, j1,j2;

  	/* Determine mean of column vectors of input data matrix */
	for (j = 1; j < (M+1); j++)
	{
		mean[j] = 0.0;
		for (i = 1; i < (N+1); i++)
		{
        		mean[j] += data[i*(M+1) + j];
		}
		mean[j] /= FLOAT_N;
	}

  	/* Center the column vectors. */
	for (i = 1; i < (N+1); i++)
	{
		for (j = 1; j < (M+1); j++)
		{
			data[i*(M+1) + j] -= mean[j];
		}
	}

  	/* Calculate the m * m covariance matrix. */
	for (j1 = 1; j1 < (M+1); j1++)
	{
		for (j2 = j1; j2 < (M+1); j2++)
     		{
       		symmat[j1*(M+1) + j2] = 0.0;
			for (i = 1; i < N+1; i++)
			{
				symmat[j1*(M+1) + j2] += data[i*(M+1) + j1] * data[i*(M+1) + j2];
			}
        		symmat[j2*(M+1) + j1] = symmat[j1*(M+1) + j2];
      		}
	}
}


void compareResults(double* symmat, double* symmat_outputFromGpu)
{
	int i,j,fail;
	fail = 0;

	for (i=1; i < (M+1); i++)
	{
		for (j=1; j < (N+1); j++)
		{
			if (percentDiff(symmat[i*(N+1) + j], symmat_outputFromGpu[i*(N+1) + j]) > PERCENT_DIFF_ERROR_THRESHOLD)
			{
				fail++;
			}
		}
	}
	printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", PERCENT_DIFF_ERROR_THRESHOLD, fail);
}


__global__ void mean_kernel(double *mean, double *data)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x + 1;

	double mean_local = 0.0;

	if ((j >= 1) && (j < (M+1)))
	{
		// mean[j] = 0.0;

		int i;
		for(i = 1; i < (N+1); i++)
		{
			mean_local += data[i * (M+1) + j];
		}
		mean[j] = mean_local / (double)FLOAT_N;

	}
}


__global__ void reduce_kernel(double *mean, double *data)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x + 1;
	int i = blockIdx.y * blockDim.y + threadIdx.y + 1;

	//__shared__ double mean_shared;

	// if (j == 0 && j == 0)
	// {
	// 	mean_shared = mean[j];
	// }


	if ((i >= 1) && (i < (N+1)) && (j >= 1) && (j < (M+1)))
	{
		data[i * (M+1) + j] -= mean[j];
		//printf("%d\n", j);
	}
}


__global__ void covar_kernel(double *symmat, double *data)
{
	int j1 = blockIdx.x * blockDim.x + threadIdx.x + 1;
	int i, j2;

	double local_res = 0.0;

	if ((j1 >= 1) && (j1 < (M+1)))
	{
		for (j2 = j1; j2 < (M+1); j2++)
		{
			//symmat[j1*(M+1) + j2] = 0.0;
			local_res = 0.0;
			for(i = 1; i < (N+1); i++)
			{
				// symmat[j1 * (M+1) + j2] += data[i *(M+1) + j1] * data[i *(M+1) + j2];
				local_res += data[i *(M+1) + j1] * data[i *(M+1) + j2];
			}
			// symmat[j2 * (M+1) + j1] = symmat[j1 * (M+1) + j2];
			symmat[j1 * (M+1) + j2] = local_res;
			symmat[j2 * (M+1) + j1] = local_res;
		}
	}
}


void calculate_on_GPU(double* data, double* symmat, double* mean, double* symmat_outputFromGpu)
{
	double *data_gpu;
	double *mean_gpu;
	double *symmat_gpu;

	cudaMalloc((void **)&data_gpu, sizeof(double) * (M+1) * (N+1));
	cudaMalloc((void **)&symmat_gpu, sizeof(double) * (M+1) * (M+1));
	cudaMalloc((void **)&mean_gpu, sizeof(double) * (M+1));

	cudaMemcpy(data_gpu, data, sizeof(double) * (M+1) * (N+1), cudaMemcpyHostToDevice);
	cudaMemcpy(symmat_gpu, symmat, sizeof(double) * (M+1) * (M+1), cudaMemcpyHostToDevice);
	cudaMemcpy(mean_gpu, mean, sizeof(double) * (M+1), cudaMemcpyHostToDevice);

	dim3 block1(DIM_THREAD_BLOCK_KERNEL_1_X, DIM_THREAD_BLOCK_KERNEL_1_Y);
	dim3 grid1((size_t)(ceil((float)M) / ((float)DIM_THREAD_BLOCK_KERNEL_1_X)), 1);

	dim3 block2(DIM_THREAD_BLOCK_KERNEL_2_X, DIM_THREAD_BLOCK_KERNEL_2_Y);
	dim3 grid2((size_t)(ceil((float)M) / ((float)DIM_THREAD_BLOCK_KERNEL_2_X)), (size_t)(ceil((float)N) / ((float)DIM_THREAD_BLOCK_KERNEL_2_X)));

	dim3 block3(DIM_THREAD_BLOCK_KERNEL_3_X, DIM_THREAD_BLOCK_KERNEL_3_Y);
	dim3 grid3((size_t)(ceil((float)M) / ((float)DIM_THREAD_BLOCK_KERNEL_3_X)), 1);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);

	mean_kernel<<<grid1, block1>>>(mean_gpu,data_gpu);
	cudaThreadSynchronize();
	reduce_kernel<<<grid2, block2, sizeof(double)>>>(mean_gpu,data_gpu);
	cudaThreadSynchronize();
	covar_kernel<<<grid3, block3>>>(symmat_gpu,data_gpu);
	cudaThreadSynchronize();

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	cudaEventDestroy(start);
    cudaEventDestroy(stop);

	printf("GPU Runtime :%0.6lf sec\n", ((float) (milliseconds) / 1000.0));

	cudaMemcpy(symmat_outputFromGpu, symmat_gpu, sizeof(double) * (M+1) * (N+1), cudaMemcpyDeviceToHost);

	cudaFree(data_gpu);
	cudaFree(symmat_gpu);
	cudaFree(mean_gpu);
}


int main()
{
	// open file
	FILE *output1;
	FILE *output2;
	output1 = fopen("gpu.out", "w");
	output2 = fopen("cpu.out", "w");
	if (output1 == NULL || output2 == NULL) {
		printf("Could not open output file!");
		exit(1);
	}


	double t_start, t_end;

	double* data_h;
	double* symmat_h;
	double* mean_h;
	double* symmat_outputFromGpu_h;

	data_h = (double*)malloc((M+1)*(N+1)*sizeof(double));
	symmat_h = (double*)malloc((M+1)*(M+1)*sizeof(double));
	mean_h = (double*)malloc((M+1)*sizeof(double));
	symmat_outputFromGpu_h = (double*)malloc((M+1)*(M+1)*sizeof(double));

	init_arrays(data_h);

	calculate_on_GPU(data_h, symmat_h, mean_h, symmat_outputFromGpu_h);

	t_start = rtclock();
	covariance(data_h, symmat_h, mean_h);
	t_end = rtclock();
	fprintf(stdout, "CPU Runtime: %0.6lfs\n", t_end - t_start);

	compareResults(symmat_h, symmat_outputFromGpu_h);

	// for (int i = 1; i < M+1; ++i) {
	// 	for (int j = 1; j < M+1; ++j) {
	// 		if(i%10==0)
	// 		{
	// 			fprintf(output1, "%19.15f\n", symmat_outputFromGpu_h[i*(M+1) + j]);
	// 			fprintf(output2, "%19.15f\n",               symmat_h[i*(M+1) + j]);
	// 		}
	// 	}
	// }

	free(data_h);
	free(symmat_h);
	free(mean_h);
	free(symmat_outputFromGpu_h);
	fclose(output1);
	fclose(output2);

  	return 0;
}
