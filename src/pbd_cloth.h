#ifndef PBD_CLOTH_H
#define PBD_CLOTH_H

#include <zjucad/matrix/matrix.h>
#include <Eigen/Sparse>

namespace bigbang {

class constraint_collector;

class pbd_cloth_solver
{
public:
  using mati_t = zjucad::matrix::matrix<size_t>;
  using matd_t = zjucad::matrix::matrix<double>;
  using vec_t = Eigen::Matrix<double, -1, 1>;
  using spmat_t = Eigen::SparseMatrix<double>;
  pbd_cloth_solver();
  // io
  int load_model(const char *filename);
  int save_model(const char *filename) const;
  // config
  int init();
  int set_density(const double rho);
  int set_time_step(const double h);
  int apply_ext_force(const size_t id, const double *force);
  int apply_gravity();
  // solve
  int precompute();
  int set_init_state();
  int advance();
  // debug
private:
  int project_constraints(vec_t &x);

  mati_t tris_;
  matd_t nods_;
  spmat_t M_, Minv_;
  vec_t vel_, fext_;
  double h_;
//  std::vector<std::shared_ptr<Constraint<double>>> buff_;
//  std::shared_ptr<Constraint<double>> cons_;
  std::shared_ptr<constraint_collector> collect_;
  const size_t MAX_ITER = 10000;
};

}

#endif
