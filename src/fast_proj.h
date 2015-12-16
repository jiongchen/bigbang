#ifndef FAST_PROJ_H
#define FAST_PROJ_H

#include <zjucad/matrix/matrix.h>
#include <Eigen/Sparse>

namespace bigbang {

template <typename T>
class Functional;

template <typename T>
class Constraint;

template <typename T>
class constraint_piece;

using mati_t=zjucad::matrix::matrix<size_t>;
using matd_t=zjucad::matrix::matrix<double>;
using pfunc_t=std::shared_ptr<Functional<double>>;
using pcons_t=std::shared_ptr<Constraint<double>>;

struct inext_cloth_args {
  double rho, h;
  int option; // 0-fast proj; 1-GS; 2-colored GS;
  size_t maxiter;
  double eps;
  double wb, wg, wp;
};

struct para_unit {
  int color_tag;
  std::shared_ptr<std::vector<std::shared_ptr<constraint_piece<double>>>> cluster;
};

class inext_cloth_solver
{
public:
  enum Color {
    RED,
    YELLOW,
    BLUE,
    BLACK
  };
  inext_cloth_solver(const mati_t &tris, const matd_t &nods);
  int initialize(const inext_cloth_args &args);
  void pin_down_vert(const size_t id, const double *pos);
  void release_vert(const size_t id);
  void apply_force(const size_t id, const double *f);
  void remove_force(const size_t id);
  int precompute();
  int advance(double *x);
private:
  int fast_project(double *x);
  int gs_solve(double *x, const std::vector<para_unit> &partition);
  int color_gs_solve(double *x, const std::vector<para_unit> &partition);
  int apply(double *x, Color color, const std::vector<para_unit> &partition);
  int symplectic_integrate(double *x);
private:
  const size_t dim_;
  const mati_t &tris_;
  const matd_t &nods_;
  mati_t edges_, diams_;

  inext_cloth_args args_;
  Eigen::SparseMatrix<double> M_, Minv_;
  Eigen::VectorXd vel_;

  std::vector<pfunc_t> ebf_;
  pfunc_t energy_;
  std::vector<para_unit> cbf_;
  pcons_t constraint_;
};

}

#endif
