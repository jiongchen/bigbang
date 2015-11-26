#include "fast_proj.h"

#include <iostream>
#include <Eigen/UmfPackSupport>
#include <Eigen/SparseQR>

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

static SimplicialCholesky<SparseMatrix<double>> LLTsolverA, LLTsolverB;

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
    vector<Triplet<double>> trips(M_.cols());
#pragma omp parallel for
    for (size_t i = 0; i < M_.cols(); ++i)
      trips[i] = Triplet<double>(i, i, 1.0/M_.coeff(i, i));
    Minv_.resize(M_.cols(), M_.cols());
    Minv_.reserve(trips.size());
    Minv_.setFromTriplets(trips.begin(), trips.end());
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

  // prefactorize
  SparseMatrix<double> LHS(dim_, dim_); {
    vector<Triplet<double>> trips;
    energy_->Hes(nullptr, &trips);
    LHS.reserve(trips.size());
    LHS.setFromTriplets(trips.begin(), trips.end());
    LHS = M_+args_.h*args_.h*LHS;
  }
  LLTsolverA.compute(LHS);
  ASSERT(LLTsolverA.info() == Success);
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
  xstar += LLTsolverA.solve(rhs);
  ASSERT(LLTsolverA.info() == Success);

  // project the constraint
//    fast_project(&xstar[0]);
  gs_solve(&xstar[0], cbf_);

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
//  UmfPackLU<SparseMatrix<double>> lu_solver;
  Map<VectorXd> Xstar(x, dim_);
  for (size_t iter = 0; iter < args_.maxiter; ++iter) {
    VectorXd cv(constraint_->Nf()); {
      cv.setZero();
      constraint_->Val(&Xstar[0], &cv[0]);
      if ( iter % 1 == 0 )
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
      SparseQR<SparseMatrix<double>, COLAMDOrdering<int>> qr;
      qr.compute(J);
      cout << "J: " << J.rows() << " " << J.cols() << " rank: " << qr.rank() << endl;
    }
    SparseMatrix<double> lhs = args_.h*args_.h*J*Minv_*J.transpose();
    SparseQR<SparseMatrix<double>, COLAMDOrdering<int>> qr;
    qr.compute(lhs);
    cout << "LHS: " << qr.rows() << " " << qr.cols() << " rank: " << qr.rank() << endl;

//    lu_solver.compute(lhs);
//    ASSERT(lu_solver.info() == Success);
//    VectorXd dl = lu_solver.solve(cv);
//    ASSERT(lu_solver.info() == Success);
    LLTsolverB.compute(lhs);
    ASSERT(LLTsolverB.info() == Success);
    VectorXd dl = LLTsolverB.solve(cv);
    ASSERT(LLTsolverB.info() == Success);

    Xstar += -args_.h*args_.h*Minv_*J.transpose()*dl;
  }
  return 0;
}

int inext_cloth_solver::gs_solve(double *x, const vector<shared_ptr<constraint_piece<double>>> &bf) {
  itr_matrix<double *> X(3, dim_/3, x);
  for (size_t iter = 0; iter < args_.maxiter; ++iter) {
    VectorXd cv(constraint_->Nf()); {
      cv.setZero();
      constraint_->Val(&X[0], &cv[0]);
      if ( iter % 100 == 0 )
        cout << "\t@max entry: " << cv.lpNorm<Infinity>() << endl;
    }
    if ( cv.lpNorm<Infinity>() < args_.eps ) {
      cout << "\t@converged after " << iter << " iteration\n";
      break;
    }
    for (auto &pc : bf) {
      double value = 0;
      pc->eval_val(&X[0], &value);
      if ( value == 0.0 )
        continue;
      matd_t grad = zeros<double>(3, pc->pn_.size());
      pc->eval_jac(&X[0], &grad[0]);
      double fnorm = 0;
      for (size_t j = 0; j < pc->pn_.size(); ++j) { // calc $\|M^{-1}\nabla C(x)\|_F^2$
        size_t idx = 3*pc->pn_[j];
        fnorm += Minv_.coeff(idx, idx)*dot(grad(colon(), j), grad(colon(), j));
      }
      for (size_t j = 0; j < pc->pn_.size(); ++j) {
        size_t idx = 3*pc->pn_[j];
        X(colon(), pc->pn_[j]) += -value/fnorm*Minv_.coeff(idx, idx)*grad(colon(), j);
      }
    }
  }
  return 0;
}

int inext_cloth_solver::red_black_gs_solve(double *x) {
//  gs_solve(red);
//#pragma omp parallel for
//  for () {
//    if ( tag != red )
//      continue;
//    apply() {
//        for() {
//          one gs
//        }
//  }

//  gs_solve(black);
  return 0;
}

}
