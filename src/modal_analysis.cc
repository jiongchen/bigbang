#include "modal_analysis.h"

#include <iostream>
#include <Eigen/UmfPackSupport>

#include "src/arpaca.h"
#include "src/config.h"
#include "src/util.h"

using namespace std;
using namespace Eigen;
using namespace arpaca;

namespace bigbang {

//int gevp_solver::solve(const spmatd_t &A, const spmatd_t &B, const size_t num,
//                       const EigenvalueType mode, MatrixXd &U, VectorXd &lambda) {
//  const size_t dim = A.cols();
//  U.resize(dim, num);
//  lambda.resize(num);

//  SimplicialLLT<spmatd_t> llt_solver;
//  UmfPackLU<spmatd_t> lu_solver;
//  spmatd_t L, Linv, C, Id(dim, dim);
//  Id.setIdentity();

//  llt_solver.compute(B);
//  ASSERT(llt_solver.info() == Success);
//  L = llt_solver.matrixL();

//  lu_solver.compute(L);
//  ASSERT(lu_solver.info() == Success);
//  Linv = lu_solver.solve(Id);

//  C = Linv*A*Linv.transpose();
//  SymmetricEigenSolver<double> eig_solver;
//  eig_solver.SetEigenvalueType(static_cast<arpaca::EigenvalueType>(mode));
//  eig_solver.SetMaxIterations(10000);
//  eig_solver.SetNumLanczosVectors(0);
//  eig_solver.SetTolerance(-1);
//  eig_solver.Solve(C.cols(), num, MakeDefaultOperator(C));
//  printf("[INFO] arpack %d iter, %d converged, %s\n",
//         eig_solver.num_actual_iterations(), eig_solver.num_converged_eigenvalues(), eig_solver.GetInfo());

//  U = eig_solver.eigenvectors();
//  lambda = eig_solver.eigenvalues();
//  U = (Linv.transpose()*U).eval();

//  return 0;
//}

basis_builder::basis_builder(const spmatd_t &M, const spmatd_t &K, const unordered_set<size_t> &fixDOF)
  : dim_(M.cols()) {
  cout << "[info] init basis builder\n";
  SimplicialLLT<spmatd_t> llt_solver;
  UmfPackLU<spmatd_t> lu_solver;
  spmatd_t L, Id(dim_, dim_);
  Id.setIdentity();

  llt_solver.compute(M);
  ASSERT(llt_solver.info() == Success);
  L = llt_solver.matrixL();

  lu_solver.compute(L);
  ASSERT(lu_solver.info() == Success);
  Linv_ = lu_solver.solve(Id);

  C_ = Linv_*K*Linv_.transpose();

  g2l_.resize(dim_);
  size_t ptr = 0;
  for (size_t i = 0; i < dim_; ++i) {
    if ( fixDOF.find(i) != fixDOF.end() )
      g2l_[i] = -1;
    else
      g2l_[i] = ptr++;
  }
  rm_spmat_col_row(C_, g2l_);
}

int basis_builder::compute(const size_t num, MatrixXd &U, VectorXd &lambda) const {
  cout << "[info] compute basis\n";
  SymmetricEigenSolver<double> solver;
  solver.SetEigenvalueType(arpaca::ALGEBRAIC_SMALLEST);
  solver.SetMaxIterations(20000);
  solver.SetNumLanczosVectors(0);
  solver.SetTolerance(-1);
  solver.Solve(C_.cols(), num, MakeDefaultOperator(C_));
  printf("[INFO] arpack %d iter, %d converged, %s\n",
         solver.num_actual_iterations(), solver.num_converged_eigenvalues(), solver.GetInfo());

  MatrixXd Ut = solver.eigenvectors();
  lambda = solver.eigenvalues();

  U.resize(dim_, num);
#pragma omp parallel for
  for (size_t i = 0; i < U.rows(); ++i) {
    if ( g2l_[i] == -1 )
      U.row(i).setZero();
    else
      U.row(i) = Ut.row(g2l_[i]);
  }
  U = Linv_*U.eval();
  return 0;
}

}
