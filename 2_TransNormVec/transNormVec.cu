#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <unistd.h>
#include <sys/time.h>
#include <curand.h>

#define BLOCK_HEIGHT 1024
#define BLOCK_WIDTH 64

/* Problem size. */
#define NX 4096
#define NY 4096

#define TILE_DIM 16

#define 	BLOCK_SIZE		256 // !!! 128
#define		RESTRICT __restrict__

#ifndef M_PI
#define M_PI 3.14159
#endif



void print_matrix(double* C, int width, int height);


int my_ceil(double num)
{
    int inum = (int)num;
    if (num == (double)inum) {
        return inum;
    }
    return inum + 1;
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

void trans_norm_vector(double* A, double* x, double* y, double* tmp)
{
	int i,j;

	for (i= 0; i < NY; i++) {
    	y[i] = 0;
	}

	for (i = 0; i < NX; i++) {
      		tmp[i] = 0;

	      	for (j = 0; j < NY; j++) {
			tmp[i] = tmp[i] + A[i*NY + j] * x[j];
		}

	      	for (j = 0; j < NY; j++) {
			y[j] = y[j] + A[i*NY + j] * tmp[i];
		}
	}
}

__global__ void MatMul(double* A, double* B, double* C, int ARows, int ACols, int BRows, int BCols, int CRows, int CCols)
{
    double CValue = 0;

    int Row = blockIdx.y*TILE_DIM + threadIdx.y;
    int Col = blockIdx.x*TILE_DIM + threadIdx.x;

    __shared__ double As[TILE_DIM][TILE_DIM];
    __shared__ double Bs[TILE_DIM][TILE_DIM];

    for (int k = 0; k < (TILE_DIM + ACols - 1)/TILE_DIM; k++) {

         if (k*TILE_DIM + threadIdx.x < ACols && Row < ARows)
             As[threadIdx.y][threadIdx.x] = A[Row*ACols + k*TILE_DIM + threadIdx.x];
         else
             As[threadIdx.y][threadIdx.x] = 0.0;

         if (k*TILE_DIM + threadIdx.y < BRows && Col < BCols)
             Bs[threadIdx.y][threadIdx.x] = B[(k*TILE_DIM + threadIdx.y)*BCols + Col];
         else
             Bs[threadIdx.y][threadIdx.x] = 0.0;

         __syncthreads();

         for (int n = 0; n < TILE_DIM; ++n)
             CValue += As[threadIdx.y][n] * Bs[n][threadIdx.x];

         __syncthreads();
    }

    if (Row < CRows && Col < CCols)
        C[((blockIdx.y * blockDim.y + threadIdx.y)*CCols) +
           (blockIdx.x * blockDim.x)+ threadIdx.x] = CValue;
}

__global__ void matvec_kernel(const double* RESTRICT  dA, const double* RESTRICT  dx, double* RESTRICT dy,
const unsigned int nRows, const unsigned int nx)
{
  const unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;

  __shared__ double x_shared[BLOCK_SIZE];

  double y_val = 0.0;

  #pragma unroll
  for (unsigned int m = 0; m < ((nx + BLOCK_SIZE - 1) / BLOCK_SIZE); ++m) {

    if ((m * BLOCK_SIZE + threadIdx.x) < nx)
      x_shared[threadIdx.x] = dx[threadIdx.x + m * BLOCK_SIZE];
    else
      x_shared[threadIdx.x] = 0.f;

    __syncthreads();

    #pragma unroll
    for (unsigned int e = 0; e < BLOCK_SIZE; ++e) {
      y_val += dA[tid + (e + BLOCK_SIZE * m) * nRows] * x_shared[e];
    }

    __syncthreads();
  }

  if (tid < nRows)
    dy[tid] = y_val;

} /* End function matvec_kernel */


//column major
__host__ void matvec(const double* RESTRICT dA, const double* RESTRICT dx, double* RESTRICT dy, const unsigned int nRows, const unsigned int nx)
{
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	dim3 dim_grid((nRows + BLOCK_SIZE - 1) / BLOCK_SIZE);
	dim3 dim_block(BLOCK_SIZE);
	matvec_kernel<<<dim_grid, dim_block>>>(dA, dx, dy, nRows, nx);

}

//row major
// __global__ void gen_matvec(double *A, double *x, double *y, const int m, const int n)
// {
//   unsigned int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
//   if ( xIndex < m ){
//     double c = 0.0f;
//     for(int i=0; i<n; i++)
//       c = c + x[i] * A[xIndex + m * i];
//     y[xIndex] = c;
//   }
// }




__device__ double atomicAdd(double* address, double val)
{
    unsigned long long int* address_as_ull =
                             (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                               __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}




__global__ void MatMulKernel(double *out, double *in, double *a, const int matrixHeight, const int matrixWidth)
{
  // get variables for loop
  // copy section of b into shared mem
  // go through the threads vertically and sum them into a variable
  // atomic add these variables to the corresponding c index

  // looping is happening horizontally on the matrix
  // BLOCK_WIDTH is again horizontal
  // BLOCK_HEIGHT is going vertical
  // n / BLOCK_WIDTH blocks horizontally
  // m / BLOCK_HEIGHT block vertically

  // get variables for loop
  // variable for loop length: blockEltHeight
  __shared__ int blockElt;
  __shared__ int blockxInd;
  __shared__ int blockyInd;
  if (threadIdx.x == 0) {
    if ((blockIdx.x + 1) * BLOCK_WIDTH <= matrixWidth)
      blockElt = BLOCK_WIDTH;
    else blockElt = matrixWidth % BLOCK_WIDTH;
    blockxInd = blockIdx.x * BLOCK_WIDTH;
    blockyInd = blockIdx.y * BLOCK_HEIGHT;
  }

  __syncthreads();

  // copy section of b into shared mem
  // use the first BLOCK_WIDTH of thread
  __shared__ double b[BLOCK_WIDTH];

  if (threadIdx.x < blockElt)
    b[threadIdx.x] = in[blockxInd + threadIdx.x];

  __syncthreads();

  // summing variable
  double cSum = (double) 0;
  int threadyInd = blockyInd + threadIdx.x;

  // make sure we are inside the matrix verticallly
  if (threadyInd < matrixHeight) {

    // go through the threads vertically and sum them into a variable
    for (int i=0; i<blockElt; i++)
      // A col index   : blockIdx.x * BLOCK_WIDTH + i : blockxInd + i
      // A row index  : blockIdx.y * BLOCK_HEIGHT + threadIdx.x : blockyInd + threadIdx.x : threadyInd
      // B index : b[i]

      // cSum = B index * ( A col index * matrixHeight + A row index)
      cSum += b[i] * a[(blockxInd + i) * (matrixHeight) + (threadyInd)];
      //printf("csum = %f\n", cSum);

    // atomic add these variables to the corresponding c index
    atomicAdd(out + threadyInd, cSum);
  }

}



__global__ void zero_vector_double(double *vec, const int n)
{
  unsigned int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
  if ( xIndex < n )
    vec[xIndex]=0.0f;
}






double matVecMul (double* out, double* in, double * A, const int m, const int n)
{
  // set up threading and blocking variables
  cudaDeviceProp dp;
  cudaGetDeviceProperties(&dp,0);
  unsigned int max_threads_per_block = dp.maxThreadsPerBlock;

  int threads_perblockm = min(m, max_threads_per_block);
  dim3 threadsPerBlockm(threads_perblockm);
  int num_blocksm = (int)ceil((double)m/(double)threads_perblockm);
  dim3 numBlocksm(num_blocksm);

  int blockCols = (int) ceil(n / (double) BLOCK_WIDTH);
  int blockRows = (int) ceil(m / (double) BLOCK_HEIGHT);
  dim3 dimBlock(BLOCK_HEIGHT);
  dim3 dimGrid(blockCols, blockRows);

  int sharedMem = 3 * sizeof (int) + BLOCK_WIDTH * sizeof (double);

  // set up timing
  cudaEvent_t start, stop;
  float time;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start,0);

  // execute kernels
  zero_vector_double<<<numBlocksm, threadsPerBlockm>>>(out, m); //ας τον μεταφερω με μηδενικα γεμισμενο
  MatMulKernel<<<dimGrid, dimBlock, sharedMem>>>(out, in, A, m, n);

  cudaThreadSynchronize();
  cudaEventRecord(stop,0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  return time;
}








int main(int argc, char *argv[])
{
	// open file
	FILE *output;
	output = fopen("gpu.out", "w");
	if (output == NULL) {
		printf("Could not open file \"gpu.out\"");
		exit(1);
	}

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	double		*A_h;
	double		*x_h;
	double		*y_h;
	// double		*tmp;
	// struct timeval	cpu_start, cpu_end;

	A_h = (double*)malloc(NX*NY*sizeof(double));
	x_h = (double*)malloc(NY*sizeof(double));
	y_h = (double*)malloc(NY*sizeof(double));
	// tmp = (double*)malloc(NX*sizeof(double));

	init_array(x_h, A_h);

	// allocate matrices, vectors to device

	// create matrices in device
	double *A_d;
	double *x_d;
	double *y_d;

	// allocate memory in device
	cudaMalloc((void**) &A_d, NX*NY*sizeof(double));
	cudaMalloc((void**) &x_d, NY*sizeof(double));
	cudaMalloc((void**) &y_d, NY*sizeof(double));

	// tranfer data to device
	cudaMemcpy(A_d, A_h, NX*NY*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(x_d, x_h, NY*sizeof(double), cudaMemcpyHostToDevice);

	// set grid, block dimensions
	// given block size
	int block_dim_x = 16;
	int block_dim_y = 16;

	//calculate grid dimensions
	int grid_dim_x = my_ceil( (double) NX / block_dim_x);
	int grid_dim_y = my_ceil( (double) NY / block_dim_y);

	dim3 dimGrid(grid_dim_x, grid_dim_x);
	dim3 dimBlock(block_dim_x, block_dim_y);


	// launch first kernel
	double first = matVecMul (y_d, x_d, A_d, NX, NY);
	//matVecMul (double * out, double * in, double * A, const int m, const int n);
	//gen_matvec<<<dimGrid, dimBlock>>>(A_d, x_d, y_d, NX, NY); // ??? NX,NY?
	//gen_matvec(double *A, double *x, double *y, const int m, const int n)
	//MatMul<<<dimGrid,dimBlock>>>(A_d, x_d, y_d, NY, NX, NY, 1, NY, 1);
	//MatMul(double* A, double* B, double* C, int ARows, int ACols, int BRows, int BCols, int CRows, int CCols)

	// launch second kernel
	// Atrasposed X (Ax) which is located in y_d and save result to x_d


	cudaEventRecord(start);
	matvec(A_d, y_d, x_d, NY, NX);
	cudaEventRecord(stop);

	// so result must be in x_d, copy it back to host
	cudaMemcpy(y_h, x_d, NY*sizeof(double), cudaMemcpyDeviceToHost);
	cudaEventSynchronize(stop);

	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	printf("GPU Runtime: %0.6lf sec\n", (milliseconds+first)/1000.0);

	//print_matrix(y_h, 1, NY);
	for (int i = 0; i < NY; ++i)
	{
			fprintf(output, "%19.15f\n", y_h[i]);
	}

	// free device and host allocated space
	cudaFree(A_d);
	cudaFree(x_d);
	cudaFree(y_d);
	free(A_h);
	free(x_h);
	free(y_h);
	fclose(output);

	//free(tmp);

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


// nvcc -o test my_attempt.cu mult_kernels.cu transpose_kernel.cu gen_gpu.cu zero_kernels.cu -I./lib/ -I. -arch=sm_20 -lcurand -lm
