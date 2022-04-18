#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include "utils.h"

#define THREADS 1000
__global__ void inner(double *a, double *b, double *c){

    __shared__ double temp[THREADS];
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    temp[threadIdx.x] = a[idx] * b[idx];
    __syncthreads();

    if( 0 == threadIdx.x ) {
        double sum = 0;
        for( int i = 0; i < THREADS; i++ )
            sum += temp[i];
        atomicAdd(c ,sum);
    }

} 

int main(int argc, char *argv[])
{

    long N = atoi(argv[1]);

    double * a  = (double *) malloc(sizeof(double)*N);
    double * b  = (double *) malloc(sizeof(double)*N);
    double * c  = (double *) malloc(sizeof(double));
    double sum = 0;
    double *ad, *bd, *sumd;

    for (long i = 0; i < N; i++) a[i] = drand48();
    for (long i = 0; i < N; i++) b[i] = drand48();

    Timer t;
    t.tic();
    for (long i = 0; i < N; i++){
        sum += a[i] * b[i];
    }
    double time = t.toc();

    
    printf("Serial Inner Product: %f\n", sum);
    printf("Time taken for serial portion: %f\n", time);


    int numblocks;
    int threadsperblock;
 
    if( (N % THREADS) == 0 )
	    numblocks = N / THREADS;
    else 
      	numblocks = (N/THREADS)>0? (N/THREADS)+1:1 ;
    
    threadsperblock = THREADS;


    dim3 grid(numblocks, 1, 1);
    dim3 block(threadsperblock, 1, 1);

    t.tic();

    cudaMalloc((void **)&ad, N*sizeof(double));
    cudaMalloc((void **)&bd, N*sizeof(double));
    cudaMalloc((void **)&sumd, sizeof(double));

    cudaMemcpy(ad, a, N*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(bd, b, N*sizeof(double), cudaMemcpyHostToDevice);

    inner<<<numblocks , threadsperblock>>>(ad, bd, sumd);

    cudaMemcpy(c, sumd, sizeof(double), cudaMemcpyDeviceToHost);
    time = t.toc();

    printf("CUDA Inner Product: %f\n", c[0]);
    printf("Time taken for CUDA portion: %f\n", time);

    double err = fabs(c[0] - sum);
    printf("Error: %10e\n", err);

    free( a ); 
    free( b );
    cudaFree( ad );
    cudaFree( bd );





}