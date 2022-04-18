#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include "utils.h"


__global__ void jacobi_2D(double* ud, double* unewd, int Ncom, double hsq){

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx>=Ncom+1 && idx<=(Ncom*Ncom)-Ncom-1 && (idx % Ncom) != 0 && (idx % Ncom) != Ncom-1){
        unewd[idx] = 0.25 * (hsq + ud[idx-1] + ud[idx+1] + ud[idx+Ncom] + ud[idx-Ncom]);
    }

}

int main(int argc, char *argv[])
{
    int N, iter, max_iters;
    double *ud, *unewd;

    N = atoi(argv[1]);
    max_iters = atoi(argv[2]);
    int THREADS = 1000;
    int cpu_threads = 8;

    int Ncom = N+2;
    int Ncomsq = Ncom * Ncom;

    double * u    = (double *) malloc(sizeof(double)* Ncomsq);
	//double * unew = (double *) calloc(sizeof(double), Ncomsq);	
	double h = 1.0 / (N + 1); 
	double hsq = h * h;
	double invhsq = 1./hsq;
	double res, res0, tol = 1e-5;
    res0 = N;
    res = res0;

    int numblocks;
    int threadsperblock;
 
    if( (Ncomsq % THREADS) == 0 )
	    numblocks = Ncomsq / THREADS;
    else 
      	numblocks = (Ncomsq/THREADS)>0? (Ncomsq/THREADS)+1:1 ;
    
    threadsperblock = THREADS;

    printf("GPU: %d blocks of %d threads each\n", numblocks, threadsperblock);     

    Timer t;
    t.tic();

    dim3 grid(numblocks, 1, 1);
    dim3 block(threadsperblock, 1, 1);

    cudaMalloc((void **)&ud, Ncomsq*sizeof(double));
    cudaMalloc((void **)&unewd, Ncomsq*sizeof(double));

    cudaMemset(ud, 0, Ncomsq*sizeof(double));
    cudaMemset(unewd, 0, Ncomsq*sizeof(double));


    for (iter = 0; iter < max_iters && res/res0 > tol; iter++) {

        jacobi_2D<<<numblocks , threadsperblock>>>(ud, unewd, Ncom, hsq);
        cudaMemcpy(ud, unewd, Ncomsq*sizeof(double), cudaMemcpyDeviceToDevice);

        if (iter % 10 == 0) {
            cudaMemcpy(u, unewd, Ncomsq*sizeof(double), cudaMemcpyDeviceToHost);
            double temp;
            res=0;
            #pragma omp parallel for num_threads(cpu_threads) reduction(+:res)
            for (int i = Ncom+1; i <= Ncomsq-Ncom-1; i++) {
                if ((i % Ncom) != 0 && (i % Ncom) != Ncom-1) {
				    temp = 1.0 + (u[i-1] + u[i+1] + u[i+Ncom] + u[i-Ncom] - 4.0*u[i]) * invhsq;
                    res += temp*temp;			
			}	
		}
      	res = sqrt(res);
        printf("Iteration: %d. Residual: %f\n", iter, res/res0);  	
		}


    }
    double time = t.toc();

    free(u);
	cudaFree(unewd);
    cudaFree(ud);

    printf("Time elapsed: %f\n", time);


}
