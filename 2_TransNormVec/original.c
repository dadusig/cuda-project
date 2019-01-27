#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <unistd.h>
#include <sys/time.h>

/* Problem size. */
#define NX 4096
#define NY 4096

#ifdef M_PI
#undef M_PI
#endif
//#ifndef M_PI
#define M_PI 3.14159
//#endif

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

	// int counter = 0;
	// for (i = 0; i < NX; i++)
	// {
	// for (j = 0; j < NY; j++)
	// {
	// counter++;
	// A[i*NY + j] = (float) counter;//((double) i*(j)) / NX;
	// printf("%3.2f\t", A[i*NY + j]);
	// }
	// printf("\n");
	// }

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

int main(int argc, char *argv[])
{
	//open file
	FILE *output;
	output = fopen("cpu.out", "w");
	if (output == NULL) {
		printf("Could not open file \"cpu.out\"");
		exit(1);
	}

	double		*A;
	double		*x;
	double		*y;
	double		*tmp;
	struct timeval	cpu_start, cpu_end;

	A = (double*)malloc(NX*NY*sizeof(double));
	x = (double*)malloc(NY*sizeof(double));
	y = (double*)malloc(NY*sizeof(double));
	tmp = (double*)malloc(NX*sizeof(double));

	init_array(x, A);

	gettimeofday(&cpu_start, NULL);
	trans_norm_vector(A, x, y, tmp);
	gettimeofday(&cpu_end, NULL);
	fprintf(stdout, "CPU Runtime :%0.6lfs\n", ((cpu_end.tv_sec - cpu_start.tv_sec) * 1000000.0 + (cpu_end.tv_usec - cpu_start.tv_usec)) / 1000000.0);

	//print_matrix(y, 1, NY);

	for (int i = 0; i < NY; ++i)
	{
			fprintf(output, "%19.15f\n", y[i]);
	}


	free(A);
	free(x);
	free(y);
	free(tmp);
	fclose(output);

  	return 0;
}
