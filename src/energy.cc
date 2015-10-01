#include "energy.h"

#include <zjucad/matrix/itr_matrix.h>

#include "mass_matrix.h"
#include "config.h"

using namespace std;
using namespace zjucad::matrix;
using namespace Eigen;
using mati_t=zjucad::matrix::matrix<size_t>;
using matd_t=zjucad::matrix::matrix<double>;

namespace bigbang {

extern "C" {

void tet_linear_(double *val, const double *x, const double *Dm, const double *vol, const double *lam, const double *miu);
void tet_linear_jac_(double *jac, const double *x, const double *Dm, const double *vol, const double *lam, const double *miu);
void tet_linear_hes_(double *hes, const double *x, const double *Dm, const double *vol, const double *lam, const double *miu);

void tet_stvk_(double *val, const double *x, const double *Dm, const double *vol, const double *lam, const double *miu);
void tet_stvk_jac_(double *jac, const double *x, const double *Dm, const double *vol, const double *lam, const double *miu);
void tet_stvk_hes_(double *hes, const double *x, const double *Dm, const double *vol, const double *lam, const double *miu);

void tet_neohookean_(double *val, const double *x, const double *Dm, const double *vol, const double *lam, const double *miu);
void tet_neohookean_jac_(double *jac, const double *x, const double *Dm, const double *vol, const double *lam, const double *miu);
void tet_neohookean_hes_(double *hes, const double *x, const double *Dm, const double *vol, const double *lam, const double *miu);

}
//==============================================================================
momentum_potential_imp_euler::momentum_potential_imp_euler(const mati_t &cell, const matd_t &nods, const double rho, const double h, const double w)
  : rho_(rho), h_(h), w_(w), dim_(nods.size()) {
  calc_mass_matrix(cell, nods, rho, nods.size(1), &M_, false);
  xn_ = Map<const VectorXd>(&nods[0], dim_);
  vn_.setZero(dim_);
}

size_t momentum_potential_imp_euler::Nx() const {
  return dim_;
}

int momentum_potential_imp_euler::Val(const double *x, double *val) const {
  RETURN_WITH_COND_TRUE(w_ == 0.0);
  Map<const VectorXd> X(x, dim_);
  VectorXd dv = (X-xn_)/h_-vn_;
  *val += w_*0.5*dv.dot(M_*dv);
  return 0;
}

int momentum_potential_imp_euler::Gra(const double *x, double *gra) const {
  RETURN_WITH_COND_TRUE(w_ == 0.0);
  Map<const VectorXd> X(x, dim_);
  Map<VectorXd> g(gra, dim_);
  g += w_*M_*((X-xn_)/h_-vn_)/h_;
  return 0;
}

int momentum_potential_imp_euler::Hes(const double *x, vector<Triplet<double>> *hes) const {
  RETURN_WITH_COND_TRUE(w_ == 0.0);
  const double coeff = w_/(h_*h_);
  for (size_t j = 0; j < M_.outerSize(); ++j) {
    for (SparseMatrix<double>::InnerIterator it(M_, j); it; ++it)
      hes->push_back(Triplet<double>(it.row(), it.col(), coeff*it.value()));
  }
  return 0;
}

void momentum_potential_imp_euler::Init(const double *x0, const double *v0) {
  if ( x0 != nullptr )
    xn_ = Map<const VectorXd>(x0, dim_);
  if ( v0 != nullptr )
    vn_ = Map<const VectorXd>(v0, dim_);
}

void momentum_potential_imp_euler::Update(const double *x) {
  Map<const VectorXd> X(x, dim_);
  vn_ = (X-xn_)/h_;
  xn_ = X;
}

double momentum_potential_imp_euler::QueryKineticEnergy() const {
  return 0.5*vn_.dot(M_*vn_);
}
//==============================================================================
momentum_potential_bdf2::momentum_potential_bdf2(const mati_t &cell, const matd_t &nods, const double rho, const double h, const double w)
  : rho_(rho), h_(h), w_(w), dim_(nods.size()) {
  calc_mass_matrix(cell, nods, rho, cell.size(1), &M_, false);
  xn_ = Map<const VectorXd>(&nods[0], dim_);
  xnn_ = xn_;
  vn_.setZero(dim_);
  vnn_.setZero(dim_);
}

size_t momentum_potential_bdf2::Nx() const {
  return dim_;
}

int momentum_potential_bdf2::Val(const double *x, double *val) const {
  RETURN_WITH_COND_TRUE(w_ == 0.0);
  Map<const VectorXd> X(x, dim_);
  VectorXd dv = 1.5/h_*X-2.0/h_*xn_+0.5/h_*xnn_-4.0/3*vn_+1.0/3*vnn_;
  *val += 0.5*w_*dv.dot(M_*dv);
  return 0;
}

int momentum_potential_bdf2::Gra(const double *x, double *gra) const {
  RETURN_WITH_COND_TRUE(w_ == 0.0);
  Map<const VectorXd> X(x, dim_);
  Map<VectorXd> g(gra, dim_);
  g += w_*1.5/h_*M_*(1.5/h_*X-2.0/h_*xn_+0.5/h_*xnn_-4.0/3*vn_+1.0/3*vnn_);
  return 0;
}

int momentum_potential_bdf2::Hes(const double *x, vector<Triplet<double> > *hes) const {
  RETURN_WITH_COND_TRUE(w_ == 0.0);
  const double coeff = w_*2.25/(h_*h_);
  for (size_t j = 0; j < M_.outerSize(); ++j) {
    for (SparseMatrix<double>::InnerIterator it(M_, j); it; ++it)
      hes->push_back(Triplet<double>(it.row(), it.col(), coeff*it.value()));
  }
  return 0;
}

void momentum_potential_bdf2::Init(const double *x0, const double *v0) {
  if ( x0 != nullptr ) {

  }
  if ( v0 != nullptr ) {

  }
}

void momentum_potential_bdf2::Update(const double *x) {
  Map<const VectorXd> X(x, dim_);
  vnn_ = vn_;
  vn_ = (3*X-4*xn_+xnn_)/(2*h_);
  xnn_ = xn_;
  xn_ = X;
}

double momentum_potential_bdf2::QueryKineticEnergy() {
  return 0.5*vn_.dot(M_*vn_);
}
//==============================================================================
gravitational_potential::gravitational_potential(const mati_t &cell, const matd_t &nods, const double rho, const double w)
  : dim_(nods.size()), w_(w) {
  calc_mass_matrix(cell, nods, rho, 1, &M_, true);
}

size_t gravitational_potential::Nx() const {
  return dim_;
}

int gravitational_potential::Val(const double *x, double *val) const {
  RETURN_WITH_COND_TRUE(w_ == 0.0);
  Map<const MatrixXd> X(x, 3, dim_/3);
  for (size_t i = 0; i < X.cols(); ++i)
    *val += w_*9.8*M_.coeff(i, i)*X(1, i);
  return 0;
}

int gravitational_potential::Gra(const double *x, double *gra) const {
  RETURN_WITH_COND_TRUE(w_ == 0.0);
  Map<MatrixXd> g(gra, 3, dim_/3);
#pragma omp parallel for
  for (size_t i = 0; i < g.cols(); ++i)
    g(1, i) += w_*9.8*M_.coeff(i, i);
  return 0;
}
//==============================================================================
elastic_potential::elastic_potential(const mati_t &tets, const matd_t &nods, const double Ym, const double Pr, const double w)
  : dim_(nods.size()), w_(w) {

}

size_t elastic_potential::Nx() const {
  return dim_;
}

int elastic_potential::Val(const double *x, double *val) const {
  RETURN_WITH_COND_TRUE(w_ == 0.0);
  itr_matrix<const double*> X(3, dim_/3, x);
  for (size_t i = 0; i < tets_.size(2); ++i) {

  }
  return 0;
}

int elastic_potential::Gra(const double *x, double *gra) const {
  RETURN_WITH_COND_TRUE(w_ == 0.0);
  return 0;
}

int elastic_potential::Hes(const double *x, vector<Triplet<double>> *hes) const {
  RETURN_WITH_COND_TRUE(w_ == 0.0);
  return 0;
}
//==============================================================================
}
