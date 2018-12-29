#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/time.h>

/* Problem size */
#define NI 8 //4096 // height
#define NJ 8 //4096 // width

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
				//printf("block(%d, %d) tid:(%d, %d) in original(%d, %d): proccessing element %d\n", blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.x, x, y, y*width+x);
				B_d[y*width+x] = c11 * A_d[(i - 1)*NJ + (j - 1)]  +  c12 * A_d[(i + 0)*NJ + (j - 1)]  +  c13 * A_d[(i + 1)*NJ + (j - 1)]
						+ c21 * A_d[(i - 1)*NJ + (j + 0)]  +  c22 * A_d[(i + 0)*NJ + (j + 0)]  +  c23 * A_d[(i + 1)*NJ + (j + 0)]
						+ c31 * A_d[(i - 1)*NJ + (j + 1)]  +  c32 * A_d[(i + 0)*NJ + (j + 1)]  +  c33 * A_d[(i + 1)*NJ + (j + 1)];
			}
		}

		//B_d[y*width+x] = 6.666;
		//printf("%d\n", y*width+x);

		//B_d[y*width+x] = 6.6;
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

__global__ void myKernelNew(double* A, double* B, double* C, int width, int height)
{
		int x = blockIdx.x * blockDim.x + threadIdx.x;
		int y = blockIdx.y * blockDim.y + threadIdx.y;
		if ( (x < width) && (y < height) )
		{
			printf("block(%d, %d) tid:(%d, %d) in original(%d, %d): proccessing element %d\n", blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.x, x, y, y*width+x);
			C[y*width+x] = 9.0;
		}
}

int main(int argc, char *argv[])
{
	// open file
	FILE *output;
	output = fopen("results.out", "w");
	if (output == NULL) {
		printf("Could not open file");
		exit(1);
	}

	double		*A_h;
	double		*B_h;
	// struct timeval	cpu_start, cpu_end;

	// create matrices in device
	double *A_d, *B_d;
	// allocate memory in device
	int size = NI*NJ*sizeof(double);
	cudaMalloc((void**) &A_d, size);
	cudaMalloc((void**) &B_d, size);

	A_h = (double*)malloc(NI*NJ*sizeof(double));
	B_h = (double*)malloc(NI*NJ*sizeof(double));

	// zero-out B  ---- Is it really necessary?
	// for (int i = 0; i < NI; ++i) {
	// 	for (int j = 0; j < NJ; ++j) {
	// 		B_h[i*NJ + j] = 0;
    //     	}
    // 	}

	//initialize the arrays
	init(A_h);
	printf("\ninitialized A:");
	print_matrix(A_h, NJ, NI); //print initialized A

	// Calc
	//Convolution(A_h, B_h);
	printf("\nB before kernel call:");
	print_matrix(B_h, NJ, NI);


	// transfer matrix A to device
	cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice);
	//cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice);

	// gettimeofday(&cpu_start, NULL);
	// Convolution(A, B);
	// gettimeofday(&cpu_end, NULL);
	// fprintf(stdout, "CPU Runtime: %0.6lfs\n", ((cpu_end.tv_sec - cpu_start.tv_sec) * 1000000.0 + (cpu_end.tv_usec - cpu_start.tv_usec)) / 1000000.0);

	// !!! set grid and block dimensions
	dim3 dimGrid(1,1);
	dim3 dimBlock(8,8);

	// !!! call GPU kernel
	convolutionKernel<<<dimGrid, dimBlock>>>(A_d, B_d, NJ, NI);

	// // call the kernel
	// dim3 dimGrid(12,12);
	// dim3 dimBlock(2,2);
	// myKernelNew<<<dimGrid, dimBlock>>>(A_d, B_d, B_d, 4, 4);

	// transer matrix B from device to Host
	cudaMemcpy(B_h, B_d, size, cudaMemcpyDeviceToHost);

	printf("\nB kernel result:");
	print_matrix(B_h, NJ, NI);

	// write results to file
	for (int i = 0; i < NI*NJ; i++)
	{
		fprintf(output, "%19.15f\n", B_h[i]);
	}

	// for validation purposes
	// lets calculate b using CPU convolution function
	Convolution(A_h, B_h);
	printf("\nB CPU result:");
	print_matrix(B_h, NJ, NI);

	free(A_h);
	free(B_h);
	fclose(output);

	// free memory from device
	cudaFree(A_d);
	cudaFree(B_d);

	return 0;
}
