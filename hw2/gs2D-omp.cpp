#include <stdio.h>
#include <math.h>
#include "utils.h"
#include <string.h>


int main(int argc, char* argv[]){
    // #ifdef _OPENMP
    //     omp_set_num_threads(4);
    // #endif
    int i, N, iter, max_iters;

    N = atoi(argv[1]);
    max_iters = atoi(argv[2]);
    int num_threads = atoi(argv[3]);

    int Ncom = N+2;
    int Ncomsq = Ncom * Ncom;

    double * u    = (double *) calloc(sizeof(double), Ncomsq);
	double * unew = (double *) calloc(sizeof(double), Ncomsq);	
	double h = 1.0 / (N + 1); 
	double hsq = h * h;
	double invhsq = 1./hsq;
	double res, res0, tol = 1e-5;

    Timer t;
    t.tic();
    res0 = N;
    res = res0;
    

    for (iter = 0; iter < max_iters && res/res0 > tol; iter++) {

        #pragma omp parallel for num_threads(num_threads) //Red Update
    	for (i = Ncom+1; i <= Ncomsq-Ncom-1; i=i+2) {
			if ((i % Ncom) != 0 && (i % Ncom) != Ncom-1) {
				unew[i] = 0.25 * (hsq + u[i-1] + u[i+1] + u[i+Ncom] + u[i-Ncom]);			
			}	
		}

        #pragma omp parallel for num_threads(num_threads) //Black Update
    	for (i = Ncom+2; i <= Ncomsq-Ncom-1; i=i+2) {
			if ((i % Ncom) != 0 && (i % Ncom) != Ncom-1) {
				unew[i] = 0.25 * (hsq + unew[i-1] + unew[i+1] + unew[i+Ncom] + unew[i-Ncom]);			
			}	
		}

		double *utemp;
		utemp = u;	
		u = unew;
		unew = utemp;
		
		if (0 == (iter % 10)) {
            double temp;
            res=0;
            #pragma omp parallel for num_threads(num_threads) reduction(+:res)
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

    double time= t.toc();
	free(u);
	free(unew);
    printf("Time elapsed: %f\n", time);
}