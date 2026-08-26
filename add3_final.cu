#include <stdio.h>
#include <stdlib.h>

__global__
void add3_kernel(double *a, double *b, double *c, int n) {
  
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int str = blockDim.x * gridDim.x;
  for (int i = idx; i < n; i += str)
    c[i] = a[i] + b[i];

}

int main(int argc, char **argv) {

  const int n = 1000000000;
  double *a = (double *) malloc(n * sizeof(double));
  double *b = (double *) malloc(n * sizeof(double));
  double *c = (double *) malloc(n * sizeof(double));

  for (int i = 0; i < n; i++) {
    a[i] = 1 + i;
    b[i] = 1 - i;
  }

  double *a_d, *b_d, *c_d;
  cudaMalloc(&a_d, n * sizeof(double));
  cudaMalloc(&b_d, n * sizeof(double));
  cudaMalloc(&c_d, n * sizeof(double));

  cudaMemcpy(a_d, a, n * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(b_d, b, n * sizeof(double), cudaMemcpyHostToDevice);

  const dim3 nthrds(1024, 1, 1);
  const dim3 nblcks((n + 1024 - 1) / 1024, 1, 1);
  
  add3_kernel<<<nblcks, nthrds>>>(a_d, b_d, c_d, n);
 
  cudaMemcpy(c, c_d, n * sizeof(double), cudaMemcpyDeviceToHost);

  for (int i = 0; i < 5; i++) {
    printf("c[%d] = %g\n", i, c[i]);
  }
  printf("c[%d] = %g\n", n-1, c[n-1]);


  cudaFree(a_d);
  cudaFree(b_d);
  cudaFree(c_d);

  free(a);
  free(b);
  free(c);
}
