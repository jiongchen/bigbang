#include "jacobi.h"

#include <iostream>
#include <cstdio>
#include <cuda.h>

#include "config.h"
#include "helper_cuda.h"

__global__ void jacobi_csr_kernel(const int *outptr, const int *inptr, const double *valptr,
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

namespace bigbang {

cuda_jacobi_solver::cuda_jacobi_solver(const Eigen::SparseMatrix<double, Eigen::RowMajor> &A)
  : rows_(A.rows()), cols_(A.cols()), nnz_(A.nonZeros()) {
  // allocate memory
  checkCudaErrors(cudaMalloc((void **)&d_outptr_, (rows_+1)*sizeof(int)));
  checkCudaErrors(cudaMalloc((void **)&d_inptr_,  nnz_*sizeof(int)));
  checkCudaErrors(cudaMalloc((void **)&d_valptr_, nnz_*sizeof(double)));
  checkCudaErrors(cudaMalloc((void **)&d_b_,      rows_*sizeof(double)));
  checkCudaErrors(cudaMalloc((void **)&d_xcurr_,  cols_*sizeof(double)));
  checkCudaErrors(cudaMalloc((void **)&d_xnext_,  cols_*sizeof(double)));
  // copy the system matrix
  checkCudaErrors(cudaMemcpy(d_outptr_, A.outerIndexPtr(), (rows_+1)*sizeof(int), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_inptr_,  A.innerIndexPtr(), nnz_*sizeof(int), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_valptr_, A.valuePtr(),      nnz_*sizeof(double), cudaMemcpyHostToDevice));
}

cuda_jacobi_solver::~cuda_jacobi_solver() {
  checkCudaErrors(cudaFree(d_outptr_));
  checkCudaErrors(cudaFree(d_inptr_));
  checkCudaErrors(cudaFree(d_valptr_));
  checkCudaErrors(cudaFree(d_b_));
  checkCudaErrors(cudaFree(d_xcurr_));
  checkCudaErrors(cudaFree(d_xnext_));
}

int cuda_jacobi_solver::apply(const Eigen::VectorXd &b, Eigen::VectorXd &x) {
  checkCudaErrors(cudaMemcpy(d_b_,     b.data(), rows_*sizeof(double), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_xcurr_, x.data(), cols_*sizeof(double), cudaMemcpyHostToDevice));

  int blockSize = 1024;
  int nBlocks = (int)ceil(rows_/1024.0);
  jacobi_csr_kernel<<<nBlocks, blockSize>>>(d_outptr_, d_inptr_, d_valptr_, rows_, cols_, nnz_, d_b_, d_xcurr_, d_xnext_);

  checkCudaErrors(cudaMemcpy(x.data(), d_xnext_, cols_*sizeof(double), cudaMemcpyDeviceToHost));
  return 0;
}

}
