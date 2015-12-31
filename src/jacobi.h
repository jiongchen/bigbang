#ifndef CUDA_JACOBI_H
#define CUDA_JACOBI_H

#include <Eigen/Sparse>

namespace bigbang {
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
  int *d_outptr_, *d_inptr_;
  double *d_valptr_, *d_b_, *d_xcurr_, *d_xnext_;
};
#endif
}
#endif
