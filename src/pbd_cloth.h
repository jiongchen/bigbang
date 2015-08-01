#ifndef PBD_CLOTH_H
#define PBD_CLOTH_H

#include <zjucad/matrix/matrix.h>
#include <Eigen/Sparse>

namespace bigbang {

class pbd_cloth_solver
{
public:
  using mati_t = zjucad::matrix::matrix<size_t>;
  using matd_t = zjucad::matrix::matrix<double>;
  using vec_t = Eigen::Matrix<double, -1, 1>;
  using spmat_t = Eigen::SparseMatrix<double>;
  int advance();
private:
  mati_t tris_;
  matd_t nods_;
  spmat_t M_;
  vec_t vel_, fext_;
  double h_;
};

}

#endif
