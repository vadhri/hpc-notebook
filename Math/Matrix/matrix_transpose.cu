%%writefile matrix_transpose_row_wise.cu

#include<stdio.h>
#include<stdlib.h>

void print_2d_matrix(int *a, int r, int c) {
  for (int i = 0; i < r; i++) {
    for (int j = 0; j < c; j++) {
      printf("%d ", a[i*c+j]);
    }
    printf("\n");
  }
}

__global__ void transpose_row_wise_per_thread(int *a, int *out, int r, int c) {
  // Extract threadid
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  for (int i=0; i < c; i++) {
    out[i*c+idx] = a[idx*c+i];
  }
}

int main() {
  int R = 100;
  int C = 100;
  int *a, *d_a, *d_out, *out;

  a = (int *)malloc(sizeof(int) * R * C);
  out = (int *)malloc(sizeof(int) * R * C);

  for (int i = 0; i < R; i++) {
    for (int j = 0; j < C; j++) {
      a[(i*C) + j] = (100^i)*(10^j);
    }
  }

  // Move the memory to GPU

  cudaMalloc((void **)&d_a, R * C * sizeof(int));
  cudaMalloc((void **)&d_out, R * C * sizeof(int));

  cudaMemcpy(d_a, a, R * C * sizeof(int), cudaMemcpyHostToDevice);

  printf("Assign 2d memory col memory for the GPU pointers.\n");

  transpose_row_wise_per_thread<<<1,R>>>(d_a, d_out, R, C);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
      printf("CUDA Error: %s\n", cudaGetErrorString(err));
  }

  cudaMemcpy(out, d_out, R * C * sizeof(int), cudaMemcpyDeviceToHost);

  printf("Out from GPU .. \n");


  for (int i=0; i<R; i++){
    for (int j =0; j <C; j++) {
      if (a[i*C+j] != out[j*C+i]) {
        printf("Incorrect transpose !!!\n");
        print_2d_matrix(out, R, C);
        print_2d_matrix(a, R, C);
      }
    }
  }

  cudaFree(d_a);
  cudaFree(d_out);

  free(out);
  free(a);

  return 0;
}