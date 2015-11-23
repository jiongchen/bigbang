#include "fast_proj.h"

#include <iostream>
#include <Eigen/UmfPackSupport>

#include "constraint.h"
#include "geom_util.h"
#include "energy.h"
#include "mass_matrix.h"
#include "config.h"

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

  // calc mass matrix and its inverse
  vel_ = VectorXd::Zero(dim_);
  calc_mass_matrix(tris_, nods_, args_.rho, 3, &M_, true); {
    SparseMatrix<double> Id(dim_, dim_);
    Id.setIdentity();
    solver_.compute(M_);
    ASSERT(solver_.info() == Success);
    Minv_ = solver_.solve(Id);
    ASSERT(solver_.info() == Success);
  }

  // add inextensible constraints
  for (size_t i = 0; i < edges_.size(2); ++i)
    cbf_.push_back(make_shared<inext_constraint>(edges_(colon(), i), nods_));

  ebf_.resize(3);
  ebf_[0] = make_shared<surf_bending_potential>(diams_, nods_, args_.wb);
  ebf_[1] = make_shared<gravitational_potential>(tris_, nods_, args_.rho, args_.wg);
  ebf_[2] = make_shared<ext_force_energy>(nods_, 1e0);
  try {
    energy_ = make_shared<energy_t<double>>(ebf_);
  } catch ( exception &e ) {
    cerr << "[exception] " << e.what() << endl;
    exit(EXIT_FAILURE);
  }
  return 0;
}

void inext_cloth_solver::pin_down_vert(const size_t id, const double *pos) {
  mati_t pid = id*ones<size_t>(1, 1);
  cbf_.push_back(make_shared<position_constraint>(pid, dim_, pos));
}

void inext_cloth_solver::release_vert(const size_t id) {
  /// @todo
}

void inext_cloth_solver::apply_force(const size_t id, const double *f) {
  dynamic_pointer_cast<ext_force_energy>(ebf_[2])->ApplyForce(id, f);
}

void inext_cloth_solver::remove_force(const size_t id) {
  dynamic_pointer_cast<ext_force_energy>(ebf_[2])->RemoveForce(id);
}

int inext_cloth_solver::assemble_constraints() {
  constraint_ = make_shared<asm_constraint>(cbf_);
  return 0;
}

int inext_cloth_solver::advance(double *x) {
  // explicit integrate the system
  Map<VectorXd> X(x, dim_);
  VectorXd force = VectorXd::Zero(dim_); {
    energy_->Gra(x, &force[0]);
  }
  VectorXd xstar = X+args_.h*vel_-args_.h*args_.h*Minv_*force;
  // project the constraint
  fast_project(&xstar[0]);

  vel_ = (xstar-X)/args_.h;
  X = xstar;
  return 0;
}

int inext_cloth_solver::fast_project(double *x) {
  UmfPackLU<SparseMatrix<double>> solver;
  Map<VectorXd> X(x, dim_);
  for (size_t iter = 0; iter < args_.maxiter; ++iter) {
    VectorXd cv(constraint_->Nf()); {
      cv.setZero();
      constraint_->Val(x, &cv[0]);
      if ( iter % 1 == 0 )
        cout << "\t@max entry: " << cv.maxCoeff() << endl;
    }
    if ( cv.maxCoeff() < args_.eps ) {
      cout << "\t@converged after " << iter << " iteration\n";
      break;
    }
    SparseMatrix<double> J(constraint_->Nf(), constraint_->Nx()); {
      vector<Triplet<double>> trips;
      constraint_->Jac(x, 0, &trips);
      J.reserve(trips.size());
      J.setFromTriplets(trips.begin(), trips.end());
    }
    SparseMatrix<double> LHS = args_.h*args_.h*(J*Minv_*J.transpose());
    solver.compute(LHS);
    ASSERT(solver.info() == Success);
    VectorXd dl = solver.solve(cv);
    ASSERT(solver.info() == Success);
    X += -args_.h*args_.h*Minv_*J.transpose()*dl;
  }
  return 0;
}

}
