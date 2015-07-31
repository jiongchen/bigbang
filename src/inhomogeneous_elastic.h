#ifndef INHOMOGENEOUS_ELASTIC_H
#define INHOMOGENEOUS_ELASTIC_H

#include <zjucad/matrix/matrix.h>
#include <Eigen/Sparse>

namespace bigbang {

class inhomogeneous_elastic_solver
{
public:
  typedef zjucad::matrix::matrix<size_t> mati_t;
  typedef zjucad::matrix::matrix<double> matd_t;
  typedef Eigen::SparseMatrix<double> spmat_t;
  typedef Eigen::Matrix<double, -1, 1> vec_t;
private:
  mati_t fTets_;
  matd_t fNods_;
  matd_t fC_;
  vec_t  fMass_;
  matd_t fLame_;  // for isotropic material

  mati_t cTets_;
  matd_t cNods_;
  matd_t cC_;
  vec_t  cMass_;
};

}

#endif
