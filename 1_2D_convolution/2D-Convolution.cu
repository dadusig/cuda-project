#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/time.h>

/* Problem size */
#define NI 8192 // height
#define NJ 8192 // width

__global__ void convolutionKernel(double *A_d, double *B_d, int width, int height)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	int i = y, j = x;
	double c11, c12, c13, c21, c22, c23, c31, c32, c33;

	c11 = +0.2;  c21 = +0.5;  c31 = -0.8;
	c12 = -0.3;  c22 = +0.6;  c32 = -0.9;
	c13 = +0.4;  c23 = +0.7;  c33 = +0.10;

	if ( (x < width) && (y < height) )
	{
		if ( i >= 1 && i < (height-1))
		{
			if ( j >= 1 && j < (width-1))
			{
				B_d[y*width+x] = c11 * A_d[(i - 1)*NJ + (j - 1)]  +  c12 * A_d[(i + 0)*NJ + (j - 1)]  +  c13 * A_d[(i + 1)*NJ + (j - 1)]
						+ c21 * A_d[(i - 1)*NJ + (j + 0)]  +  c22 * A_d[(i + 0)*NJ + (j + 0)]  +  c23 * A_d[(i + 1)*NJ + (j + 0)]
						+ c31 * A_d[(i - 1)*NJ + (j + 1)]  +  c32 * A_d[(i + 0)*NJ + (j + 1)]  +  c33 * A_d[(i + 1)*NJ + (j + 1)];
			}
		}
	}
}

void print_matrix(double* C, int width, int height)
{
	printf("%s\n", " ");
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			//printf("(%d, %d) -> %d\t", i, j, C[i*width+j]);
			printf("%- 3.2f   ", C[i*width+j]);
		}
		printf("%s\n", " ");
	}
}

void Convolution(double* A, double* B)
{
	int i, j;
	double c11, c12, c13, c21, c22, c23, c31, c32, c33;

	c11 = +0.2;  c21 = +0.5;  c31 = -0.8;
	c12 = -0.3;  c22 = +0.6;  c32 = -0.9;
	c13 = +0.4;  c23 = +0.7;  c33 = +0.10;


	for (i = 1; i < NI - 1; ++i) {
		for (j = 1; j < NJ - 1; ++j) {
			B[i*NJ + j] = c11 * A[(i - 1)*NJ + (j - 1)]  +  c12 * A[(i + 0)*NJ + (j - 1)]  +  c13 * A[(i + 1)*NJ + (j - 1)]
				    + c21 * A[(i - 1)*NJ + (j + 0)]  +  c22 * A[(i + 0)*NJ + (j + 0)]  +  c23 * A[(i + 1)*NJ + (j + 0)]
				    + c31 * A[(i - 1)*NJ + (j + 1)]  +  c32 * A[(i + 0)*NJ + (j + 1)]  +  c33 * A[(i + 1)*NJ + (j + 1)];
		}
	}
}

void init(double* A)
{
	int i, j;

	for (i = 0; i < NI; ++i) {
		for (j = 0; j < NJ; ++j) {
			A[i*NJ + j] = (double)rand()/RAND_MAX;
        	}
    	}
}

int main(int argc, char *argv[])
{
	// open file
	FILE *output;
	output = fopen("gpu.out", "w");
	if (output == NULL) {
		printf("Could not open file");
		exit(1);
	}

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	double		*A_h;
	double		*B_h;

	A_h = (double*)calloc(NI*NJ, sizeof(double)); // better use calloc than malloc, initializes elements to 0
	B_h = (double*)calloc(NI*NJ, sizeof(double));


	// create matrices in device
	double *A_d;
	double *B_d;
	// allocate memory in device
	long size = NI*NJ*sizeof(double);
	cudaMalloc((void**) &A_d, size);
	cudaMalloc((void**) &B_d, size);

	/* zero-out B  ---- Is it really necessary? */
	// for (int i = 0; i < NI; ++i) {
	// 	for (int j = 0; j < NJ; ++j) {
	// 		B_h[i*NJ + j] = 0;
    //     	}
    // 	}

	//initialize the arrays
	init(A_h);
	// printf("\ninitialized A:");
	// print_matrix(A_h, NJ, NI); //print initialized A

	/* Print B initial state on HOST */
	// printf("\nB before kernel call:");
	// print_matrix(B_h, NJ, NI);

	// transfer matrix A to device
	cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice);
	cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice);

	// !!! set grid and block dimensions
	dim3 dimGrid(512,512);
	dim3 dimBlock(16,16);

	// call GPU kernel
	cudaEventRecord(start);
	convolutionKernel<<<dimGrid, dimBlock>>>(A_d, B_d, NJ, NI);
	cudaEventRecord(stop);

	// transer matrix B from DEVICE to HOST
	cudaMemcpy(B_h, B_d, size, cudaMemcpyDeviceToHost);
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	printf("GPU Runtime: %0.6lfs\n", milliseconds/1000.0);

	/* print kernel result */
	// printf("\nB kernel result:");
	// print_matrix(B_h, NJ, NI);

	// write results to file
	// for (int i = 0; i < NI*NJ; i++)
	// {
	// 	if(i%NJ==0)
	// 	fprintf(output, "%19.15f\n", B_h[i]);
	// }
	for (int i = 1; i < NI - 1; ++i) {
		for (int j = 1; j < NJ - 1; ++j) {
			//B[i*NJ + j]
			if(i%NJ/2==0)
			fprintf(output, "%19.15f\n", B_h[i*NJ + j]);
		}
	}

	/* 	for validation purposes
	 	lets calculate b using CPU convolution function and print it */
	// Convolution(A_h, B_h);
	// printf("\nB CPU result:");
	// print_matrix(B_h, NJ, NI);

	free(A_h);
	free(B_h);
	fclose(output);

	// free memory from device
	cudaFree(A_d);
	cudaFree(B_d);

	return 0;
}
