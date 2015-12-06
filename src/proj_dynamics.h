#ifndef PROJECTIVE_DYNAMICS_H
#define PROJECTIVE_DYNAMICS_H

#include <zjucad/matrix/matrix.h>
#include <Eigen/Sparse>

namespace bigbang {

template <typename>
class Functional;

using mati_t=zjucad::matrix::matrix<size_t>;
using matd_t=zjucad::matrix::matrix<double>;
using pfunc_t=std::shared_ptr<Functional<double>>;

struct proj_dyn_args {
  double rho, h;
  size_t maxiter;
  double eps;
  double ws;  // for stretch
  double wb;  // for bending
  double wg;  // for gravity
  double wp;  // for position
};

class proj_dyn_solver
{
public:
  proj_dyn_solver(const mati_t &tris, const matd_t &nods);
  int initialize(const proj_dyn_args &args);
  int pin_down_vert(const size_t id, const double *pos);
  int release_vert(const size_t id);
  int apply_force(const size_t id, const double *f);
  int remove_force(const size_t id);
  int precompute();
  int advance(double *x) const;
  int advance_beta(double *x) const;
private:
  const mati_t &tris_;
  const matd_t &nods_;
  mati_t edges_, diams_;

  proj_dyn_args args_;
  std::vector<pfunc_t> impebf_, expebf_;
  pfunc_t impE_, expE_;

  const size_t dim_;
  Eigen::SparseMatrix<double> LHS_;
  Eigen::SimplicialCholesky<Eigen::SparseMatrix<double>> solver_;
};

}

#endif
