#include "fast_proj.h"

#include <iostream>

#include "geom_util.h"
#include "energy.h"
#include "mass_matrix.h"

using namespace std;
using namespace zjucad::matrix;
using namespace Eigen;

namespace bigbang {

inext_cloth_solver::inext_cloth_solver(const mati_t &tris, const matd_t &nods)
  : dim_(nods.size()), tris_(tris), nods_(nods) {
  get_edge_elem(tris_, edges_);
  get_diam_elem(tris_, diams_);
}

int inext_cloth_solver::initialize(const inext_cloth_args &args) {
  args_ = args;

  ebf_.resize(3);
  ebf_[0] = make_shared<surf_bending_potential>(diams_, nods_, args_.wb);
  ebf_[1] = make_shared<gravitational_potential>(tris_, nods_, args_.density, args_.wg);
  ebf_[2] = make_shared<ext_force_energy>(nods_, 1e0);

  cbf_.resize(2);
  cbf_[0] = make_shared<position_constraint>(nods_);
  cbf_[1] = make_shared<inextensible_constraint>(edges_, nods_);

  try {
    energy_ = make_shared<energy_t<double>>(ebf_);
    constraint_ = make_shared<constraint_t<double>>(cbf_);
  } catch ( exception &e ) {
    cerr << "[exception] " << e.what() << endl;
    exit(EXIT_FAILURE);
  }
  return 0;
}

void inext_cloth_solver::pin_down_vert(const size_t id, const double *pos) {
  dynamic_pointer_cast<position_constraint>(cbf_[0])->Pin(id, pos);
}

void inext_cloth_solver::release_vert(const size_t id) {
  dynamic_pointer_cast<position_constraint>(cbf_[0])->Release(id);
}

void inext_cloth_solver::apply_force(const size_t id, const double *f) {
  dynamic_pointer_cast<ext_force_energy>(ebf_[2])->ApplyForce(id, f);
}

void inext_cloth_solver::remove_force(const size_t id) {
  dynamic_pointer_cast<ext_force_energy>(ebf_[2])->RemoveForce(id);
}

int inext_cloth_solver::advance(double *x) const {
  // explicit integrate the system
  // project the constraint
  return 0;
}

}
