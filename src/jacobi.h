#ifndef CUDA_JACOBI_H
#define CUDA_JACOBI_H

#include <Eigen/Sparse>

#ifdef USE_CUDA
/** Note that Eigen 3.3 will use ptrdiff_t
 *  instead of int for 'Index' type **/
class cuda_jacobi_solver
{
public:
  cuda_jacobi_solver(const Eigen::SparseMatrix<double, Eigen::RowMajor> &A);
  ~cuda_jacobi_solver();
  int apply(const Eigen::VectorXd &b, Eigen::VectorXd &x);
private:
  size_t rows_, cols_, nnz_;
  int *outptr_d_, *inptr_d_;
  double *valptr_d_, *b_d_, *x_curr_d_, *x_next_d_;
};
#endif

#endif
