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

/**
 * @param method option:
 * 0-Liu13;
 * 1-kovalsky15;
 * 2-modified FMS;
 * 3-todo;
 * 4-direct+chebyshev
 * 5-Jacobi+chebyshev
*/
struct proj_dyn_args {
  double rho, h;
  size_t maxiter;
  int method;
  double eps;
  double ws;  // for stretch
  double wb;  // for bending
  double wg;  // for gravity
  double wp;  // for position
};

class proj_dyn_spring_solver
{
public:
  proj_dyn_spring_solver(const mati_t &tris, const matd_t &nods);
  int initialize(const proj_dyn_args &args);
  int pin_down_vert(const size_t id, const double *pos);
  int release_vert(const size_t id);
  int apply_force(const size_t id, const double *f);
  int remove_force(const size_t id);
  int precompute();
  int advance(double *x) const;
private:
  int advance_alpha(double *x) const;
  int advance_beta(double *x) const;
  int advance_gamma(double *x) const;
  int advance_delta(double *x) const;
  int advance_epsilon(double *x) const;
private:
  const mati_t &tris_;
  const matd_t &nods_;
  mati_t edges_, diams_;

  proj_dyn_args args_;
  std::vector<pfunc_t> impebf_, expebf_;
  pfunc_t impE_, expE_;

  const size_t dim_;
  Eigen::SparseMatrix<double> LHS_;
};

class proj_dyn_tet_solver
{
public:
  proj_dyn_tet_solver(const mati_t &tets, const matd_t &nods);
  int initialize(const proj_dyn_args &args);
  int pin_down_vert(const size_t id, const double *pos);
  int release_vert(const size_t id);
  int apply_force(const size_t id, const double *f);
  int remove_force(const size_t id);
  int precompute();
  int advance(double *x) const;
private:
  const mati_t &tets_;
  const matd_t &nods_;

  proj_dyn_args args_;
  std::vector<pfunc_t> ebf_;
  pfunc_t energy_;

  const size_t dim_;
  Eigen::SparseMatrix<double> LHS_;
};

}

#endif
