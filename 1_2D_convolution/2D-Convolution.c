#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/time.h>

/* Problem size */
#define NI 8192
#define NJ 8192

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
	double		*A;
	double		*B;
	struct timeval	cpu_start, cpu_end;

	// open file
	FILE *output;
	output = fopen("cpu.out", "w");
	if (output == NULL) {
		printf("Could not open file \"spacetime.out\"");
		exit(1);
	}


	A = (double*)malloc(NI*NJ*sizeof(double));
	B = (double*)malloc(NI*NJ*sizeof(double));

	//initialize the arrays
	init(A);

	gettimeofday(&cpu_start, NULL);
	Convolution(A, B);
	gettimeofday(&cpu_end, NULL);
	fprintf(stdout, "CPU Runtime: %0.6lf sec\n", ((cpu_end.tv_sec - cpu_start.tv_sec) * 1000000.0 + (cpu_end.tv_usec - cpu_start.tv_usec)) / 1000000.0);

	// write results to file
	// for (int i = 0; i < NI*NJ; i++)
	// {
	// 	if(i%NJ==0)
	// 	fprintf(output, "%19.15f\n", B[i]);
	// }

	for (int i = 1; i < NI - 1; ++i) {
		for (int j = 1; j < NJ - 1; ++j) {
			//B[i*NJ + j]
			if(i%NJ/2==0)
			fprintf(output, "%19.15f\n", B[i*NJ + j]);
		}
	}

	free(A);
	free(B);
	fclose(output);

	return 0;
}
