#include "proj_dynamics.h"

#include <iostream>

#include "def.h"
#include "energy.h"
#include "geom_util.h"
#include "config.h"

using namespace std;
using namespace Eigen;

namespace bigbang {

static Eigen::SimplicialCholesky<Eigen::SparseMatrix<double>> solver_;

proj_dyn_spring_solver::proj_dyn_spring_solver(const mati_t &tris, const matd_t &nods)
  : tris_(tris), nods_(nods), dim_(nods.size()) {
  get_edge_elem(tris_, edges_);
  get_diam_elem(tris_, diams_);
}

int proj_dyn_spring_solver::initialize(const proj_dyn_args &args) {
  cout << "[info] init the projective dynamics solver";
  args_ = args;

  impebf_.resize(6);
  impebf_[0] = make_shared<momentum_potential_imp_euler>(tris_, nods_, args_.rho, args_.h);
  switch ( args_.method ) {
    case 0: impebf_[1] = make_shared<fast_mass_spring>(edges_, nods_, args_.ws); break;
    case 1: impebf_[1] = make_shared<fast_mass_spring>(edges_, nods_, args_.ws); break;
    case 2: impebf_[1] = make_shared<modified_fms_energy>(edges_, nods_, args_.ws); break;
    default: break;
  }
  impebf_[2] = make_shared<positional_potential>(nods_, args_.wp);
  impebf_[3] = make_shared<gravitational_potential>(tris_, nods_, args_.rho, args_.wg);
  impebf_[4] = make_shared<ext_force_energy>(nods_, 1e0);
  impebf_[5] = make_shared<isometric_bending>(diams_, nods_, args_.wb);

  try {
    impE_ = make_shared<energy_t<double>>(impebf_);
  } catch ( exception &e ) {
    cerr << "[exception] " << e.what() << endl;
    exit(EXIT_FAILURE);
  }
  cout << "......done\n";
  return 0;
}

int proj_dyn_spring_solver::pin_down_vert(const size_t id, const double *pos) {
  return dynamic_pointer_cast<positional_potential>(impebf_[2])->Pin(id, pos);
}

int proj_dyn_spring_solver::release_vert(const size_t id) {
  return dynamic_pointer_cast<positional_potential>(impebf_[2])->Release(id);
}

int proj_dyn_spring_solver::apply_force(const size_t id, const double *f) {
  return dynamic_pointer_cast<ext_force_energy>(impebf_[4])->ApplyForce(id, f);
}

int proj_dyn_spring_solver::remove_force(const size_t id) {
  return dynamic_pointer_cast<ext_force_energy>(impebf_[4])->RemoveForce(id);
}

int proj_dyn_spring_solver::precompute() {
  cout << "[info] solver is doing precomputation";
  ASSERT(dim_ == impE_->Nx());
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

int proj_dyn_spring_solver::advance(double *x) const {
  int rtn = 0;
  switch ( args_.method ) {
    case 0: rtn = advance_alpha(x); break;
    case 1: rtn = advance_beta(x); break;
    case 2: rtn = advance_gamma(x); break;
    case 3: rtn = advance_delta(x); break;
    case 4: rtn = advance_epsilon(x); break;
    default: return __LINE__;
  }
  return rtn;
}

int proj_dyn_spring_solver::advance_alpha(double *x) const {
  ASSERT(args_.method == 0);
  Map<VectorXd> X(x, dim_);
  VectorXd xstar = X;
  const auto fms = dynamic_pointer_cast<fast_mass_spring>(impebf_[1]);
  VectorXd prev_step, next_step;
  const static SparseMatrix<double>& S = fms->get_df_mat();
  // iterate solve
  for (size_t iter = 0; iter < args_.maxiter; ++iter) {
    if ( iter % 100 == 0 ) {
      double value = 0;
      impE_->Val(&xstar[0], &value);
      cout << "\t@iter " << iter << " energy value: " << value << endl;
    }
    // local step for constraint projection
    fms->LocalSolve(&xstar[0]);
    prev_step = S*xstar-Map<const VectorXd>(fms->get_aux_var(), fms->aux_dim());
    // global step for compatible position
    VectorXd jac = VectorXd::Zero(dim_); {
      impE_->Gra(&xstar[0], &jac[0]);
    }
    VectorXd dx = -solver_.solve(jac);
    double xstar_norm = xstar.norm();
    xstar += dx;
    next_step = S*xstar-Map<const VectorXd>(fms->get_aux_var(), fms->aux_dim());
    if ( iter < 5 ) {
      cout << "\t@prev step size: " << prev_step.norm() << endl;
      cout << "\t@post step size: " << next_step.norm() << endl;
      cout << "\t@dx norm: " << dx.norm() << endl;
      cout << "\t@turning angle: " << acos(prev_step.dot(next_step)/(prev_step.norm()*next_step.norm()))/M_PI*180 << endl << endl;
    }
    if ( dx.norm() <= args_.eps*xstar_norm ) {
      cout << "[info] converged after " << iter+1 << " iterations\n";
      break;
    }
  }
  // update configuration
  dynamic_pointer_cast<momentum_potential>(impebf_[0])->Update(&xstar[0]);
  X = xstar;
  return 0;
}

int proj_dyn_spring_solver::advance_beta(double *x) const {
  ASSERT(args_.method == 1);
  Map<VectorXd> X(x, dim_);
  auto fms = dynamic_pointer_cast<fast_mass_spring>(impebf_[1]);
  const size_t fdim = fms->aux_dim();

  VectorXd xstar = X, jac(dim_), eta(dim_);
  VectorXd z(fdim), n(fdim);
  Map<const VectorXd> Pz(fms->get_aux_var(), fdim);
  double d0 = 0;
  const SparseMatrix<double>& S = fms->get_df_mat();
  VectorXd next_step(fdim);

  // iterative solve
  VectorXd unkown(dim_+1), inve(dim_), invb(dim_);
  for (size_t iter = 0; iter < args_.maxiter; ++iter) {
    if ( iter % 100 == 0 ) {
      double value = 0;
      impE_->Val(&xstar[0], &value);
      cout << "\t@iter " << iter << " energy value: " << value << endl;
    }
    // local solve
    z = S*xstar;
    fms->LocalSolve(&xstar[0]);
    n = z-Pz;
    if ( iter % 100 == 0 ) {
      cout << "\t@iter " << iter << " normal: " << n.norm() << endl << endl;
    }
    // global solve
    jac.setZero(); {
      impE_->Gra(&xstar[0], &jac[0]);
      jac *= -1;
    }
    eta = S.transpose()*n;
    d0 = n.dot(Pz-z);

    invb = solver_.solve(jac);
    ASSERT(solver_.info() == Success);
    inve = solver_.solve(eta);
    ASSERT(solver_.info() == Success);

    if ( n.norm() < args_.eps ) {
      unkown.head(dim_) = invb;
    } else {
      unkown[unkown.size()-1] = (-d0+eta.dot(invb))/eta.dot(inve);
      unkown.head(dim_) = invb-unkown[unkown.size()-1]*inve;
    }
    double xstar_norm = xstar.norm();
    xstar += unkown.head(dim_);
    next_step = S*xstar-Pz;
    if ( iter < 5 ) {
      cout << "\t@prev step size: " << n.norm() << endl;
      cout << "\t@post step size: " << next_step.norm() << endl;
      cout << "\t@dx norm: " << unkown.head(dim_).norm() << endl;
      cout << "\t@turning angle: " << acos(n.dot(next_step)/(n.norm()*next_step.norm()))/M_PI*180 << "\n\n";
    }
    if ( unkown.head(dim_).norm() <= args_.eps*xstar_norm ) {
      cout << "[info] converged after " << iter+1 << " iterations\n";
      break;
    }
  }
  dynamic_pointer_cast<momentum_potential>(impebf_[0])->Update(&xstar[0]);
  X = xstar;
  return 0;
}

int proj_dyn_spring_solver::advance_gamma(double *x) const {
  ASSERT(args_.method == 2);
  Map<VectorXd> X(x, dim_);
  VectorXd xstar = X;
  const auto fms = dynamic_pointer_cast<modified_fms_energy>(impebf_[1]);
  VectorXd prev_step, next_step;
  const static SparseMatrix<double>& S = fms->get_df_mat();
  // iterate solve
  for (size_t iter = 0; iter < args_.maxiter; ++iter) {
    if ( iter % 100 == 0 ) {
      double value = 0;
      impE_->Val(&xstar[0], &value);
      cout << "\t@iter " << iter << " energy value: " << value << endl;
    }
    // local step for constraint projection
    fms->LocalSolve(&xstar[0]);
    prev_step = S*xstar-Map<const VectorXd>(fms->get_aux_var(), fms->aux_dim());
    // global step for compatible position
    VectorXd jac = VectorXd::Zero(dim_); {
      impE_->Gra(&xstar[0], &jac[0]);
    }
    VectorXd dx = -solver_.solve(jac);
    double xstar_norm = xstar.norm();
    xstar += dx;
    next_step = S*xstar-Map<const VectorXd>(fms->get_aux_var(), fms->aux_dim());
    if ( iter < 5 ) {
      cout << "\t@prev step size: " << prev_step.norm() << endl;
      cout << "\t@post step size: " << next_step.norm() << endl;
      cout << "\t@dx norm: " << dx.norm() << endl;
      cout << "\t@turning angle: " << acos(prev_step.dot(next_step)/(prev_step.norm()*next_step.norm()))/M_PI*180 << endl << endl;
    }
    if ( dx.norm() <= args_.eps*xstar_norm ) {
      cout << "[info] converged after " << iter+1 << " iterations\n";
      break;
    }
  }
  // update configuration
  dynamic_pointer_cast<momentum_potential>(impebf_[0])->Update(&xstar[0]);
  X = xstar;
  return 0;
}

static std::vector<Eigen::Vector3d> record;

int proj_dyn_spring_solver::advance_delta(double *x) const {
  ASSERT(args_.method == 3);
  Map<VectorXd> X(x, dim_);
  VectorXd xstar = X;
  const auto fms = dynamic_pointer_cast<fast_mass_spring>(impebf_[1]);
  Map<const MatrixXd> d(fms->get_aux_var(), 3, fms->aux_dim()/3);
  // iterate solve
  record.clear();
  for (size_t iter = 0; iter < args_.maxiter; ++iter) {
    if ( iter % 100 == 0 ) {
      double value = 0;
      impE_->Val(&xstar[0], &value);
      cout << "\t@iter " << iter << " energy value: " << value << endl;
    }
    /*+++++++++++++++++++++++++++*/
    record.push_back(d.col(0));
    /*+++++++++++++++++++++++++++*/

    // local step for constraint projection
    fms->LocalSolve(&xstar[0]);
    // global step for compatible position
    VectorXd jac = VectorXd::Zero(dim_); {
      impE_->Gra(&xstar[0], &jac[0]);
    }
    VectorXd dx = -solver_.solve(jac);
    double xstar_norm = xstar.norm();
    xstar += dx;
    if ( dx.norm() <= args_.eps*xstar_norm ) {
      cout << "[info] converged after " << iter+1 << " iterations\n";
      break;
    }
  }
  // update configuration
  dynamic_pointer_cast<momentum_potential>(impebf_[0])->Update(&xstar[0]);
  X = xstar;
  return 0;
}

int proj_dyn_spring_solver::advance_epsilon(double *x) const {
  ASSERT(args_.method == 4);
  Map<VectorXd> X(x, dim_);
  VectorXd xstar = X;
  // chebyshev iteration
  for (size_t iter = 0; iter < args_.maxiter; ++iter) {

  }
  return 0;
}

//====================== tetrahedral projective solver =========================
proj_dyn_tet_solver::proj_dyn_tet_solver(const mati_t &tets, const matd_t &nods)
  : tets_(tets), nods_(nods), dim_(nods.size()) {}

int proj_dyn_tet_solver::initialize(const proj_dyn_args &args) {
  cout << "[info] init the projective dynamics solver...";
  args_ = args;

  ebf_.resize(5);
  ebf_[0] = make_shared<momentum_potential_imp_euler>(tets_, nods_, args_.rho, args_.h);
  ebf_[1] = make_shared<tet_arap_energy>(tets_, nods_, args_.ws);
  ebf_[2] = make_shared<positional_potential>(nods_, args_.wp);
  ebf_[3] = make_shared<gravitational_potential>(tets_, nods_, args_.rho, args_.wg);
  ebf_[4] = make_shared<ext_force_energy>(nods_, 1e0);

  try {
    energy_ = make_shared<energy_t<double>>(ebf_);
  } catch ( exception &e ) {
    cerr << "[exception] " << e.what() << endl;
    exit(EXIT_FAILURE);
  }
  cout << "done\n";
  return 0;
}

int proj_dyn_tet_solver::pin_down_vert(const size_t id, const double *pos) {
  return dynamic_pointer_cast<positional_potential>(ebf_[2])->Pin(id, pos);
}

int proj_dyn_tet_solver::release_vert(const size_t id) {
  return dynamic_pointer_cast<positional_potential>(ebf_[2])->Release(id);
}

int proj_dyn_tet_solver::apply_force(const size_t id, const double *f) {
  return dynamic_pointer_cast<ext_force_energy>(ebf_[4])->ApplyForce(id, f);
}

int proj_dyn_tet_solver::remove_force(const size_t id) {
  return dynamic_pointer_cast<ext_force_energy>(ebf_[4])->RemoveForce(id);
}

int proj_dyn_tet_solver::precompute() {
  cout << "[info] solver is doing precomputation";
  ASSERT(dim_ == energy_->Nx());
  vector<Triplet<double>> trips;
  energy_->Hes(nullptr, &trips);
  LHS_.resize(dim_, dim_);
  LHS_.reserve(trips.size());
  LHS_.setFromTriplets(trips.begin(), trips.end());
  solver_.compute(LHS_);
  ASSERT(solver_.info() == Success);
  cout << "......done\n";
  return 0;
}

int proj_dyn_tet_solver::advance(double *x) const {
  Map<VectorXd> X(x, dim_);
  VectorXd xstar = X;
  const auto arap = dynamic_pointer_cast<tet_arap_energy>(ebf_[1]);
  // iterate solve
  for (size_t iter = 0; iter < args_.maxiter; ++iter) {
    if ( iter % 10 == 0 ) {
      double value = 0;
      energy_->Val(&xstar[0], &value);
      cout << "\t@iter " << iter << " energy value: " << value << endl;
    }
    // local step for constraint projection
    arap->LocalSolve(&xstar[0]);
    // global step for compatible position
    VectorXd jac = VectorXd::Zero(dim_); {
      energy_->Gra(&xstar[0], &jac[0]);
    }
    VectorXd dx = -solver_.solve(jac);
    double xstar_norm = xstar.norm();
    xstar += dx;
    if ( dx.norm() <= args_.eps*xstar_norm ) {
      cout << "\t@CONVERGED after " << iter+1 << " iterations\n";
      break;
    }
  }
  // update configuration
  dynamic_pointer_cast<momentum_potential>(ebf_[0])->Update(&xstar[0]);
  X = xstar;
  return 0;
}

}
