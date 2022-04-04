#include <algorithm>
#include <stdio.h>
#include <math.h>
#include <omp.h>

// Scan A array and write result into prefix_sum array;
// use long data type to avoid overflow
void scan_seq(long* prefix_sum, const long* A, long n) {
  if (n == 0) return;
  prefix_sum[0] = 0;
  for (long i = 1; i < n; i++) {
    prefix_sum[i] = prefix_sum[i-1] + A[i-1];
  }
  // for(long i=0;i<n; i++){
  //   printf("prefix_sum[%d] = %d\n",i, prefix_sum[i]);
  // }
}

void scan_omp(long* prefix_sum, const long* A, long n) {
  // TODO: implement multi-threaded OpenMP scan
  int numthreads = 4;
  // #pragma omp parallel for num_threads(numthreads)
  // for (long i = 1; i < n; i++){
  //   prefix_sum[i] = 0;
  // }

  long parts = n/numthreads;
  #pragma omp parallel for num_threads(numthreads)
  for (long i = 0; i < n; i++){
    if (i%parts == 0){
      prefix_sum[i] = 0;
      continue;
    }
    prefix_sum[i] = prefix_sum[i-1] + A[i-1];   
  }

  // for(long i=0;i<n; i++){
  //   printf("prefix_sum[%d] = %d\n",i, prefix_sum[i]);
  // }
  
  long tempsum = prefix_sum[(n/numthreads)-1];
  for (long i=parts; i<n ; i++){
    if (i%parts == 0){
      tempsum = prefix_sum[i-1] + A[i-1];
      prefix_sum[i] += prefix_sum[i-1] + A[i-1];
      continue;
    }
    prefix_sum[i] += tempsum;
    
  }
  // for(long i=0;i<n; i++){
  //     printf("prefix_sum[%d] = %d\n",i, prefix_sum[i]);
  //   }

 

}

int main() {
  long N = 100000000;
  long* A = (long*) malloc(N * sizeof(long));
  long* B0 = (long*) malloc(N * sizeof(long));
  long* B1 = (long*) malloc(N * sizeof(long));
  for (long i = 0; i < N; i++) A[i] = rand();

  double tt = omp_get_wtime();
  scan_seq(B0, A, N);
  printf("sequential-scan = %fs\n", omp_get_wtime() - tt);

  tt = omp_get_wtime();
  scan_omp(B1, A, N);
  printf("parallel-scan   = %fs\n", omp_get_wtime() - tt);

  long err = 0;
  for (long i = 0; i < N; i++) err = std::max(err, std::abs(B0[i] - B1[i]));
  printf("error = %ld\n", err);

  free(A);
  free(B0);
  free(B1);
  return 0;
}
