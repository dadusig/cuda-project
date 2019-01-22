#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <unistd.h>
#include <sys/time.h>
#include <curand.h>
#include "cublas_v2.h"

#define BLOCK_HEIGHT 1024
#define BLOCK_WIDTH 64

/* Problem size. */
#define NX 16000
#define NY 16000

#define TILE_DIM 16

#define 	BLOCK_SIZE		128 // !!! 128?256
#define		RESTRICT __restrict__

//same M_PI on both GPU and CPU
#ifdef M_PI
#undef M_PI
#endif
//#ifndef M_PI
#define M_PI 3.14159
//#endif

void print_matrix(double* C, int width, int height);

__global__ void matvec_kernel_row_major(const double* RESTRICT A_d, const double* RESTRICT x_d, double* RESTRICT y_d, const unsigned int rows, const unsigned int cols)
{
    const unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;

	__shared__ double x_shared[BLOCK_SIZE];

	double y_val = 0.0;

	unsigned int numBlocks = (cols + BLOCK_SIZE - 1) / BLOCK_SIZE;

	#pragma unroll
	for (unsigned int m = 0; m < numBlocks; m++)
	{
		unsigned int blockID = m*BLOCK_SIZE + threadIdx.x;

		if (blockID < cols)
		{
			x_shared[threadIdx.x] = x_d[m * BLOCK_SIZE + threadIdx.x];
		}
		else
		{
			 x_shared[threadIdx.x] = 0.0;
		}

		__syncthreads();

		for (unsigned int e = 0; e < BLOCK_SIZE; e++)
		{
			y_val += A_d[tid * cols + (e + BLOCK_SIZE * m)] * x_shared[e];
		}

		__syncthreads();

		if (tid < rows)
		{
			y_d[tid] = y_val;
		}

	}

}

__host__ float matvec_ROW_MAJOR(const double* RESTRICT A_d, const double* RESTRICT x_d, double* RESTRICT y_d, const unsigned int nRows, const unsigned int nCols)
{

	dim3 dim_grid(((nRows + BLOCK_SIZE - 1) / BLOCK_SIZE));
	dim3 dim_block(BLOCK_SIZE);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);
	matvec_kernel_row_major<<<dim_grid, dim_block>>>(A_d, x_d, y_d, nRows, nCols);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	cudaEventDestroy(start);
    cudaEventDestroy(stop);
	return milliseconds;

}

__host__ float calculateUsingBLAS(double *A_d, double *x_d, double *Ax_d)
{
	double alf = 1.0;
	double beta = 0.0;
	cublasHandle_t handle;
	cublasCreate(&handle);
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);
	cublasDgemv(handle, CUBLAS_OP_T, NY, NX, &alf, A_d, NY, x_d, 1, &beta, Ax_d, 1);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	cudaEventDestroy(start);
    cudaEventDestroy(stop);
	return milliseconds;
}




__global__ void matvec_kernel_column_major(const double* RESTRICT A_d, const double* RESTRICT  x_d, double* RESTRICT y_d, const unsigned int nRows, const unsigned int nCols)
{

	const unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;

	__shared__ double x_shared[BLOCK_SIZE];

	double y_val = 0.0;

	#pragma unroll
	for (unsigned int m = 0; m < ((nCols + BLOCK_SIZE - 1) / BLOCK_SIZE); ++m)
	{

		if ((m * BLOCK_SIZE + threadIdx.x) < nCols)
		{
			x_shared[threadIdx.x] = x_d[threadIdx.x + m * BLOCK_SIZE];
		}
		else
		{
			x_shared[threadIdx.x] = 0.f;
		}

		__syncthreads();

		#pragma unroll
		for (unsigned int e = 0; e < BLOCK_SIZE; ++e)
		{
			y_val += A_d[tid + (e + BLOCK_SIZE * m) * nRows] * x_shared[e];
		}

		__syncthreads();
	}

	if (tid < nRows)
  	{
    	y_d[tid] = y_val;
	}

}

__host__ float matvec_COL_MAJOR(const double* RESTRICT A_d, const double* RESTRICT x_d, double* RESTRICT y_d, const unsigned int nRows, const unsigned int nCols)
{

	dim3 dim_grid((nRows + BLOCK_SIZE - 1) / BLOCK_SIZE);
	dim3 dim_block(BLOCK_SIZE);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);
	matvec_kernel_column_major<<<dim_grid, dim_block>>>(A_d, x_d, y_d, nRows, nCols);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	cudaEventDestroy(start);
    cudaEventDestroy(stop);
	return milliseconds;

}


void init_array(double *x, double *A)
{
	int i, j;

	for (i = 0; i < NX; i++)
	{
		for (j = 0; j < NY; j++)
		{
			A[i*NY + j] = ((double) i*(j)) / NX;
		}
	}

	for (j = 0; j < NY; j++)
	{
		x[j] = j * M_PI;
	}
}



int main(int argc, char *argv[])
{
	// cudaDeviceReset();
	// open file
	FILE *output;
	output = fopen("gpu.out", "w");
	if (output == NULL) {
		printf("Could not open file \"gpu.out\"");
		exit(1);
	}

	/*- allocate memory on host -*/
	double *A_h; 	// NX X NY
	double *x_h; 	// NY X 1
	double *Ax_h;	// NX X 1
	double *y_h;	// NY X 1


	A_h  =  (double*) calloc(NX*NY, sizeof(double));
	x_h  =  (double*) calloc(NY, sizeof(double));
	Ax_h =  (double*) calloc(NX, sizeof(double));
	y_h  =  (double*) calloc(NY, sizeof(double));
	init_array(x_h, A_h);

	/*- allocate memory on device -*/
	double *A_d; 	// NX X NY
	double *x_d; 	// NY X 1
	double *Ax_d;	// NX X 1
	double *y_d;	// NY X 1

	cudaMalloc((void**) &A_d, NX*NY*sizeof(double));
	cudaMalloc((void**) &x_d, NY*sizeof(double));
	cudaMalloc((void**) &Ax_d,NX*sizeof(double));
	cudaMalloc((void**) &y_d, NY*sizeof(double));

	/*- copy to device A and x -*/
	cudaMemcpy(A_d, A_h, 	NX*NY*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(x_d, x_h, 	NY*sizeof(double), cudaMemcpyHostToDevice);
	//cudaMemcpy(Ax_d, Ax_h, 	NX*sizeof(double), cudaMemcpyHostToDevice);
	//cudaMemcpy(y_d, y_h, 	NY*sizeof(double), cudaMemcpyHostToDevice);

	float milliseconds1 = matvec_ROW_MAJOR(A_d, x_d, Ax_d, NX, NY);
	float milliseconds2 = matvec_COL_MAJOR(A_d, Ax_d, y_d, NY, NX);

	/*- copy back to host desired result -*/
	cudaMemcpy(y_h, y_d, NY*sizeof(double), cudaMemcpyDeviceToHost);

	printf("GPU Runtime :%0.6lf sec\n", ((float) (milliseconds1 + milliseconds2) / 1000.0));

	printf("%s\n", "Writing results to gpu.out...");
	for (int i = 0; i < NY; ++i)
	{
			fprintf(output, "%19.15f\n", y_h[i]);
	}

	float milliseconds_BLAS = calculateUsingBLAS(A_d, x_d, Ax_d);
	printf("GPU Runtime :%0.6lf sec (on cuBLAS)\n", ((float) (milliseconds_BLAS) / 1000.0));

	/*- free space on both device and host -*/
	cudaFree(A_d);
	cudaFree(x_d);
	cudaFree(Ax_d);
	cudaFree(y_d);
	free(A_h);
	free(x_h);
	free(Ax_h);
	free(y_h);
	fclose(output);
  	return 0;
}

void print_matrix(double* C, int width, int height)
{
	printf("\n");
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			//printf("(%d, %d) -> %d\t", i, j, C[i*width+j]);
			printf("%- 6.2f   ", C[i*width+j]);
		}
		printf("%s\n", " ");
	}
}
