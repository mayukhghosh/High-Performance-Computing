#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include "utils.h"

#define THREADS 1000
__global__ void matvec(double *a, double *b, double *c, int N){

    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if( idx<N ) {
        double sum = 0;
        for( int j = 0; j < N; j++ )
            sum += a[idx*N + j] * b[j];
        c[idx] = sum;
    }

} 

int main(int argc, char *argv[])
{

    long N = atoi(argv[1]);

    double * a  = (double *) malloc(sizeof(double) * N*N);
    double * b  = (double *) malloc(sizeof(double)* N);
    double * sum_ref  = (double *) malloc(sizeof(double)* N);
    double * sum  = (double *) malloc(sizeof(double)*N);
    double *ad, *bd, *sumd;

    for (long i = 0; i < N*N; i++) a[i] = drand48();        
    for (long i = 0; i < N; i++) b[i] = drand48();

    Timer t;
    t.tic();
    for (long i = 0; i < N; i++){
        for (long j = 0; j < N; j++){
            sum_ref[i] += a[i*N + j] * b[j];
        }
    }
    double time = t.toc();

    
    //printf("Serial Product: %f\n", sum);
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

    cudaMalloc((void **)&ad, N*N*sizeof(double));
    cudaMalloc((void **)&bd, N*sizeof(double));
    cudaMalloc((void **)&sumd, N*sizeof(double));

    cudaMemcpy(ad, a, N*N*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(bd, b, N*sizeof(double), cudaMemcpyHostToDevice);

    matvec<<<numblocks , threadsperblock>>>(ad, bd, sumd, N);

    cudaMemcpy(sum, sumd, N*sizeof(double), cudaMemcpyDeviceToHost);
    time = t.toc();

    //printf("CUDA Inner Product: %f\n", sumd);
    printf("Time taken for CUDA portion: %f\n", time);

    double max_err = 0;
    for (long i = 0; i < N; i++) max_err = std::max(max_err, fabs(sum[i] - sum_ref[i]));
    printf("Error: %10e\n", max_err);

    free( a ); 
    free( b );
    free( sum );
    free( sum_ref );
    cudaFree( ad );
    cudaFree( bd );
    cudaFree( sumd );





}