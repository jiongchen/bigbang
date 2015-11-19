#include "proj_dynamics.h"

#include <iostream>

#include "def.h"
#include "energy.h"
#include "geom_util.h"
#include "config.h"

using namespace std;
using namespace Eigen;

namespace bigbang {

proj_dyn_solver::proj_dyn_solver(const mati_t &tris, const matd_t &nods)
  : tris_(tris), nods_(nods), dim_(nods.size()) {
  get_edge_elem(tris_, edges_);
  get_diam_elem(tris_, diams_);
}

int proj_dyn_solver::initialize(const proj_dyn_args &args) {
  cout << "[info] init the projective dynamics solver";
  args_ = args;

  impebf_.resize(5);
  impebf_[0] = make_shared<momentum_potential_imp_euler>(tris_, nods_, args_.rho, args_.h);
  impebf_[1] = make_shared<fast_mass_spring>(tris_, nods_, args_.ws);
  impebf_[2] = make_shared<positional_potential>(nods_, args_.wp);
  impebf_[3] = make_shared<gravitational_potential>(tris_, nods_, args_.rho, args_.wg);
  impebf_[4] = make_shared<ext_force_energy>(nods_, 1e0);

  expebf_.resize(1);
  expebf_[0] = make_shared<surf_bending_potential>(diams_, nods_, args_.wb);

  try {
    impE_ = make_shared<energy_t<double>>(impebf_);
    expE_ = make_shared<energy_t<double>>(expebf_);
  } catch ( exception &e ) {
    cerr << "[exception] " << e.what() << endl;
    exit(EXIT_FAILURE);
  }
  cout << "......done\n";
  return 0;
}

int proj_dyn_solver::pin_down_vert(const size_t id, const double *pos) {
  return dynamic_pointer_cast<positional_potential>(impebf_[2])->Pin(id, pos);
}

int proj_dyn_solver::release_vert(const size_t id) {
  return dynamic_pointer_cast<positional_potential>(impebf_[2])->Release(id);
}

int proj_dyn_solver::apply_force(const size_t id, const double *f) {
  return dynamic_pointer_cast<ext_force_energy>(impebf_[4])->ApplyForce(id, f);
}

int proj_dyn_solver::remove_force(const size_t id) {
  return dynamic_pointer_cast<ext_force_energy>(impebf_[4])->RemoveForce(id);
}

int proj_dyn_solver::precompute() {
  cout << "[info] solver is doing precomputation";
  ASSERT(dim_ == impE_->Nx());
  const SparseMatrix<double> &M = dynamic_pointer_cast<momentum_potential>(impebf_[0])
      ->MassMatrix();
  SparseMatrix<double> Id(dim_, dim_);
  Id.setIdentity();
  solver_.compute(M);
  ASSERT(solver_.info() == Success);
  Minv_ = solver_.solve(Id);
  ASSERT(solver_.info() == Success);

  vector<Triplet<double>> trips;
  impE_->Hes(nullptr, &trips);
  LHS_.resize(dim_, dim_);
  LHS_.reserve(trips.size());
  LHS_.setFromTriplets(trips.begin(), trips.end());
  solver_.compute(LHS_);
  ASSERT(solver_.info() == Success);
  cout << "......done\n";
  return 0;
}

int proj_dyn_solver::advance(double *x) const {
  Map<VectorXd> X(x, dim_);
  VectorXd fexp = VectorXd::Zero(dim_); {
    expE_->Gra(x, &fexp[0]);
    fexp *= -1;
  }
  const VectorXd &vel = dynamic_pointer_cast<momentum_potential>(impebf_[0])->CurrVelocity();
  VectorXd xstar = X;//+args_.h*vel+args_.h*args_.h*Minv_*fexp;

  for (size_t iter = 0; iter < args_.maxiter; ++iter) {
    if ( iter % 10 == 0 ) {
      double value = 0;
      impE_->Val(&xstar[0], &value);
      cout << "\t@energy value: " << value << endl;
    }
    // local solve
    dynamic_pointer_cast<fast_mass_spring>(impebf_[1])->LocalSolve(&xstar[0]);
    // global solve
    VectorXd jac = VectorXd::Zero(dim_); {
      impE_->Gra(&xstar[0], &jac[0]);
    }
    VectorXd dx = -solver_.solve(jac);
    double xstar_norm = xstar.norm();
    xstar += dx;
    if ( dx.norm() <= args_.eps*xstar_norm ) {
      cerr << "[info] converged after " << iter+1 << " iterations\n";
      break;
    }
  }
  // update configuration
  dynamic_pointer_cast<momentum_potential>(impebf_[0])->Update(&xstar[0]);
  X = xstar;
  return 0;
}

}
