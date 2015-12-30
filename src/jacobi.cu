#include "jacobi.h"

#include <iostream>
#include <cstdio>
#include <cuda.h>

#include "config.h"

using namespace std;
using namespace Eigen;

__global__ void jacobi_on_device(const int *outptr, const int *inptr, const double *valptr,
                                 const size_t rows, const size_t cols, const size_t nnz,
                                 const double *b, const double *x_curr, double *x_next) {
  int i = blockIdx.x*blockDim.x+threadIdx.x;
  if ( i < rows ) {
    double diag = 0.0;
    double temp = b[i];
    for (long int cnt = outptr[i]; cnt < outptr[i+1]; ++cnt) {
      if ( inptr[cnt] == i )
        diag = valptr[cnt];
      else
        temp -= x_curr[inptr[cnt]]*valptr[cnt];
    }
    x_next[i] = temp/diag;
  }
}

static void select_gpu(int *gpu_num, int *num_devs) {
  // gpu_num: (I/O): I: Default choice,
  //                 O: best device, changed only if more than one device
  // num_devs: (O)   Number of found devices.
  int best = *gpu_num;
  cudaGetDeviceCount(num_devs);

  if ( *num_devs > 1 ) {
    int dev_num;
    int max_cores = 0;
    for (dev_num = 0; dev_num < *num_devs; dev_num++) {
      cudaDeviceProp dev_properties;
      cudaGetDeviceProperties(&dev_properties, dev_num);
      if (max_cores < dev_properties.multiProcessorCount) {
        max_cores = dev_properties.multiProcessorCount;
        best = dev_num;
      }
    }
    *gpu_num = best;
  }
}

static void test_device(int devID) {
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, devID);
  if (deviceProp.major == 9999 && deviceProp.minor == 9999) {
    printf("There is no device supporting CUDA.\n");
    cudaThreadExit();
  }
  else
    printf("Using GPU device number %d.\n", devID);
}

cuda_jacobi_solver::cuda_jacobi_solver(const SparseMatrix<double, RowMajor> &A)
  : rows_(A.rows()), cols_(A.cols()), nnz_(A.nonZeros()) {
  // allocate memory
  cudaMalloc((void **)&outptr_d_, (rows_+1)*sizeof(int));
  cudaMalloc((void **)&inptr_d_,  nnz_*sizeof(int));
  cudaMalloc((void **)&valptr_d_, nnz_*sizeof(double));
  cudaMalloc((void **)&b_d_,      rows_*sizeof(double));
  cudaMalloc((void **)&x_curr_d_, cols_*sizeof(double));
  cudaMalloc((void **)&x_next_d_, cols_*sizeof(double));
  // copy the system matrix
  ASSERT(cudaMemcpy(outptr_d_, A.outerIndexPtr(), (rows_+1)*sizeof(int), cudaMemcpyHostToDevice) == cudaSuccess);
  ASSERT(cudaMemcpy(inptr_d_,  A.innerIndexPtr(), nnz_*sizeof(int), cudaMemcpyHostToDevice) == cudaSuccess);
  ASSERT(cudaMemcpy(valptr_d_, A.valuePtr(),      nnz_*sizeof(double), cudaMemcpyHostToDevice) == cudaSuccess);
}

cuda_jacobi_solver::~cuda_jacobi_solver() {
  cudaFree(outptr_d_);
  cudaFree(inptr_d_);
  cudaFree(valptr_d_);
  cudaFree(b_d_);
  cudaFree(x_curr_d_);
  cudaFree(x_next_d_);
}

int cuda_jacobi_solver::apply(const VectorXd &b, VectorXd &x) {
  ASSERT(cudaMemcpy(b_d_,      b.data(), rows_*sizeof(double), cudaMemcpyHostToDevice) == cudaSuccess);
  ASSERT(cudaMemcpy(x_curr_d_, x.data(), cols_*sizeof(double), cudaMemcpyHostToDevice) == cudaSuccess);
  x /= 0.0;
  ASSERT(cudaMemcpy(x_next_d_, x.data(), cols_*sizeof(double), cudaMemcpyHostToDevice) == cudaSuccess);

  int blockSize = 1024;
  int nBlocks = 2;
  jacobi_on_device<<<nBlocks, blockSize>>>(outptr_d_, inptr_d_, valptr_d_,
                                           rows_, cols_, nnz_,
                                           b_d_, x_curr_d_, x_next_d_);
  cudaDeviceSynchronize();
  cudaMemcpy(x.data(), x_next_d_, cols_*sizeof(double), cudaMemcpyDeviceToHost);
  return 0;
}
