#include "fast_proj.h"

#include <iostream>
#include <Eigen/UmfPackSupport>

#include "constraint.h"
#include "geom_util.h"
#include "energy.h"
#include "mass_matrix.h"
#include "config.h"
#include "util.h"

using namespace std;
using namespace zjucad::matrix;
using namespace Eigen;

namespace bigbang {

static SparseMatrix<double> LHS;
static SimplicialCholesky<SparseMatrix<double>> solver;

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
    solver.compute(M_);
    ASSERT(solver.info() == Success);
    Minv_ = solver.solve(Id);
    ASSERT(solver.info() == Success);
  }

  // add inextensible constraints
  for (size_t i = 0; i < edges_.size(2); ++i)
    cbf_.push_back(make_shared<inext_constraint>(edges_(colon(), i), nods_));

  // assemble energy
  ebf_.resize(4);
  ebf_[0] = make_shared<isometric_bending>(diams_, nods_, args_.wb);
  ebf_[1] = make_shared<gravitational_potential>(tris_, nods_, args_.rho, args_.wg);
  ebf_[2] = make_shared<ext_force_energy>(nods_, 1e0);
  ebf_[3] = make_shared<positional_potential>(nods_, args_.wp);
  try {
    energy_ = make_shared<energy_t<double>>(ebf_);
  } catch ( exception &e ) {
    cerr << "[exception] " << e.what() << endl;
    exit(EXIT_FAILURE);
  }
  return 0;
}

void inext_cloth_solver::pin_down_vert(const size_t id, const double *pos) {
  dynamic_pointer_cast<positional_potential>(ebf_[3])->Pin(id, pos);
}

void inext_cloth_solver::release_vert(const size_t id) {
  dynamic_pointer_cast<positional_potential>(ebf_[3])->Release(id);
}

void inext_cloth_solver::apply_force(const size_t id, const double *f) {
  dynamic_pointer_cast<ext_force_energy>(ebf_[2])->ApplyForce(id, f);
}

void inext_cloth_solver::remove_force(const size_t id) {
  dynamic_pointer_cast<ext_force_energy>(ebf_[2])->RemoveForce(id);
}

int inext_cloth_solver::precompute() {
  // assemble constraints
  constraint_ = make_shared<asm_constraint>(cbf_);

  cout << "here\n";
  // prefactorize
  LHS.resize(dim_, dim_); {
    vector<Triplet<double>> trips;
    cout << "fuck\n";
    energy_->Hes(nullptr, &trips);
    cout << "fuck1\n";
    LHS.reserve(trips.size());
    LHS.setFromTriplets(trips.begin(), trips.end());
    LHS = M_+args_.h*args_.h*LHS;
  }
  solver.compute(LHS);
  ASSERT(solver.info() == Success);
  return 0;
}

int inext_cloth_solver::advance(double *x) {
  Map<VectorXd> X(x, dim_);
  // implicit integration
  VectorXd xstar = X;
  VectorXd fi = VectorXd::Zero(dim_); {
    energy_->Gra(&xstar[0], &fi[0]);
  }
  VectorXd rhs = args_.h*M_*vel_-args_.h*args_.h*fi;
  VectorXd dx = solver.solve(rhs);
  xstar += dx;

  // project the constraint
  fast_project(&xstar[0]);

  vel_ = (xstar-X)/args_.h;
  X = xstar;
  return 0;
}

int inext_cloth_solver::symplectic_integrate(double *x) {
  Map<VectorXd> Xstar(x, dim_);
  VectorXd force = VectorXd::Zero(dim_); {
    energy_->Gra(x, &force[0]);
  }
  Xstar += args_.h*vel_-args_.h*args_.h*Minv_*force;
}

int inext_cloth_solver::fast_project(double *x) {
  UmfPackLU<SparseMatrix<double>> lu_solver;
  Map<VectorXd> Xstar(x, dim_);
  for (size_t iter = 0; iter < args_.maxiter; ++iter) {
    VectorXd cv(constraint_->Nf()); {
      cv.setZero();
      constraint_->Val(&Xstar[0], &cv[0]);
      if ( iter % 10 == 0 )
        cout << "\t@max entry: " << cv.lpNorm<Infinity>() << endl;
    }
    if ( cv.lpNorm<Infinity>() < args_.eps ) {
      cout << "\t@converged after " << iter << " iteration\n";
      break;
    }
    SparseMatrix<double> J(constraint_->Nf(), constraint_->Nx()); {
      vector<Triplet<double>> trips;
      constraint_->Jac(&Xstar[0], 0, &trips);
      J.reserve(trips.size());
      J.setFromTriplets(trips.begin(), trips.end());
    }
    SparseMatrix<double> lhs = args_.h*args_.h*J*Minv_*J.transpose();
    lu_solver.compute(lhs);
    ASSERT(lu_solver.info() == Success);
    VectorXd dl = lu_solver.solve(cv);
    ASSERT(lu_solver.info() == Success);
    Xstar += -args_.h*args_.h*Minv_*J.transpose()*dl;
  }
  return 0;
}

int inext_cloth_solver::gs_solve(double *x) {
//  for (size_t iter = 0; iter < MAX_ITER; ++iter) {
//    double cons_sqr = query_constraint_squared_norm(Xstar.data());
//    if ( i % 100 == 0 )
//      cout << "\t@constraint norm: " << cons_sqr << endl;
//    if ( cons_sqr < 1e-8 ) {
//      cout << "\t@converged\n";
//      break;
//    }
//    for (auto &co : buff_) {
//      double val = 0.0;
//      co->eval_val(x.data(), &val);
//      if ( val == 0.0 )
//        continue;
//      if ( co->type_ == constraint_piece<double>::EQUAL
//           || (co->type_ == constraint_piece<double>::GREATER && val < 0.0) ) {
//        matd_t jac = zeros<double>(3, co->pn_.size());
//        co->eval_jac(&X[0], &jac[0]);
//        double s = 0;
//        for (size_t i = 0; i < co->pn_.size(); ++i)
//          s += Minv_[3*co->pn_[i]]*dot(jac(colon(), i), jac(colon(), i));
//        if ( s == 0.0 )
//          continue;
//        s = val/s;
//        for (size_t i = 0; i < co->pn_.size(); ++i)
//          X(colon(), co->pn_[i]) += -s*Minv_[3*co->pn_[i]]*jac(colon(), i);
//      }
//    }
//  }
//  return 0;
}

}
