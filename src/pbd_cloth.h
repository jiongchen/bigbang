#ifndef PBD_CLOTH_H
#define PBD_CLOTH_H

#include <zjucad/matrix/matrix.h>
#include <Eigen/Sparse>

namespace bigbang {

template <typename T>
class constraint_piece;

struct pbd_args {
  size_t maxiter;
};

class pbd_cloth_solver
{
public:
  using mati_t = zjucad::matrix::matrix<size_t>;
  using matd_t = zjucad::matrix::matrix<double>;
  using vec_t = Eigen::Matrix<double, -1, 1>;
  using spmat_t = Eigen::SparseMatrix<double>;
  pbd_cloth_solver();
  // io
  int load_model_from_obj(const char *filename);
  int load_model_from_vtk(const char *filename);
  int save_model_to_obj(const char *filename) const;
  int save_model_to_vtk(const char *filename) const;
  // config
  int init();
  void set_mass_matrix(const double rho);
  void set_time_step(const double h) { h_ = h; }
  void apply_ext_force(const size_t id, const double *force);
  void apply_gravity();
  void erase_gravity();
  void attach_vert(const size_t id, const double *pos=nullptr);
  // solve
  int precompute();
  int advance();
  // debug
  int test_calc_length();
  int test_calc_dihedral_angle();
  int test_edge_extraction();
  int test_diamond_extraction();
private:
  int project_constraints(vec_t &x, const size_t iter_num);
  int add_one_coll_constraint(); /// @todo
  int add_strecth_constraints(const mati_t &tris, const matd_t &nods);
  int add_bend_constraints(const mati_t &tris, const matd_t &nods);
  double query_constraint_squared_norm(const double *x) const;

  mati_t tris_;
  matd_t nods_;
  spmat_t M_;
  vec_t Minv_;
  vec_t vel_, fext_, grav_;
  double h_, rho_;
  std::vector<std::shared_ptr<constraint_piece<double>>> buff_, coll_;

  const size_t MAX_ITER = 10;
};

}

#endif
