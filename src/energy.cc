#include "energy.h"

#include <iostream>
#include <hjlib/math/blas_lapack.h>
#include <zjucad/matrix/lapack.h>
#include <zjucad/matrix/itr_matrix.h>
#include <zjucad/matrix/io.h>
#include <unsupported/Eigen/KroneckerProduct>

#include "mass_matrix.h"
#include "config.h"
#include "geom_util.h"
#include "util.h"

using namespace std;
using namespace zjucad::matrix;
using namespace Eigen;

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

void hex_linear_(double *val, const double *x, const double *h, const double *lam, const double *miu);
void hex_linear_jac_(double *jac, const double *x, const double *h, const double *lam, const double *miu);
void hex_linear_hes_(double *hes, const double *x, const double *h, const double *lam, const double *miu);

void hex_stvk_(double *val, const double *x, const double *h, const double *lam, const double *miu);
void hex_stvk_jac_(double *jac, const double *x, const double *h, const double *lam, const double *miu);
void hex_stvk_hes_(double *hes, const double *x, const double *h, const double *lam, const double *miu);

void calc_edge_length_(double *val, const double *x);
void calc_edge_length_jac_(double *jac, const double *x);
void calc_edge_length_hes_(double *hes, const double *x);

void mass_spring_(double *val, const double *x, const double *d);
void mass_spring_jac_(double *jac, const double *x, const double *d);
void mass_spring_hes_(double *hes, const double *x, const double *d);

void calc_dih_angle_(double *val, const double *x);
void calc_dih_angle_jac_(double *jac, const double *x);
void calc_dih_angle_hes_(double *hes, const double *x);

void surf_bending_(double *val, const double *x, const double *d, const double *l, const double *area);
void surf_bending_jac_(double *jac, const double *x, const double *d, const double *l, const double *area);
void surf_bending_hes_(double *hes, const double *x, const double *d, const double *l, const double *area);

void line_bending_(double *val, const double *x, const double *d1, const double *d2);
void line_bending_jac_(double *jac, const double *x, const double *d1, const double *d2);
void line_bending_hes_(double *hes, const double *x, const double *d1, const double *d2);

}

void tet_corotational_jac_(double *jac, const double *x, const double *Dm, const double *vol, const double *lam, const double *miu) {
  Map<VectorXd> grad(jac, 3, 4);
  Map<const MatrixXd> X(x, 3, 4);
  Map<const Matrix3d> DM(Dm);
}

void tet_corotational_hes_(double *hes, const double *x, const double *Dm, const double *vol, const double *lam, const double *miu) {
  Map<MatrixXd> H(hes, 12, 12);
  Map<const MatrixXd> X(x, 3, 4);
  Map<const Matrix3d> DM(Dm);
  Matrix3d defoGrad = (X.block<3, 3>(1, 3)-X.col(0)*Vector3d::Ones().transpose())*DM;
  Matrix3d R;
  MatrixXd Rll = kroneckerProduct(Matrix3d::Identity(), R);
}

//==============================================================================
momentum_potential_imp_euler::momentum_potential_imp_euler(const mati_t &cell, const matd_t &nods,
                                                           const double rho, const double h, const double w)
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
momentum_potential_bdf2::momentum_potential_bdf2(const mati_t &cell, const matd_t &nods,
                                                 const double rho, const double h, const double w)
  : rho_(rho), h_(h), w_(w), dim_(nods.size()) {
  calc_mass_matrix(cell, nods, rho, nods.size(1), &M_, false);
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

int momentum_potential_bdf2::Hes(const double *x, vector<Triplet<double>> *hes) const {
  RETURN_WITH_COND_TRUE(w_ == 0.0);
  const double coeff = w_*2.25/(h_*h_);
  for (size_t j = 0; j < M_.outerSize(); ++j) {
    for (SparseMatrix<double>::InnerIterator it(M_, j); it; ++it) {
      hes->push_back(Triplet<double>(it.row(), it.col(), coeff*it.value()));
    }
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

double momentum_potential_bdf2::QueryKineticEnergy() const {
  return 0.5*vn_.dot(M_*vn_);
}
//==============================================================================
gravitational_potential::gravitational_potential(const mati_t &cell, const matd_t &nods,
                                                 const double rho, const double w)
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
elastic_potential::elastic_potential(const mati_t &tets, const matd_t &nods, Material type,
                                     const double Ym, const double Pr, const double w)
  : tets_(tets), type_(type), dim_(nods.size()), w_(w) {
  vol_.resize(1, tets_.size(2));
  Dm_.resize(9, tets_.size(2));
#pragma omp parallel for
  for (size_t i = 0; i < tets_.size(2); ++i) {
    matd_t edge = nods(colon(), tets_(colon(1, 3), i))-nods(colon(), tets_(0, i))*ones<double>(1, 3);
    matd_t cp_edge = edge;
    vol_[i] = fabs(det(cp_edge))/6.0;
    if ( inv(edge) )
      cerr << "\tdegenerated tet " << i << endl;
    std::copy(edge.begin(), edge.end(), &Dm_(0, i));
  }
  // calculate \lambda and \miu according
  // to Young's modulus and Poisson ratio
  lam_ = Ym*Pr/((1.0+Pr)*(1.0-2.0*Pr));
  miu_ = Ym/(2.0*(1.0+Pr));
}

size_t elastic_potential::Nx() const {
  return dim_;
}

int elastic_potential::Val(const double *x, double *val) const {
  RETURN_WITH_COND_TRUE(w_ == 0.0);
  itr_matrix<const double*> X(3, dim_/3, x);
  for (size_t i = 0; i < tets_.size(2); ++i) {
    matd_t vert = X(colon(), tets_(colon(), i));
    double value = 0;
    switch ( type_ ) {
      case LINEAR:
        tet_linear_(&value, &vert[0], &Dm_(0, i), &vol_[i], &lam_, &miu_);
        break;
      case STVK:
        tet_stvk_(&value, &vert[0], &Dm_(0, i), &vol_[i], &lam_, &miu_);
        break;
      case COROTATIONAL:
        break;
      case NEOHOOKEAN:
        tet_neohookean_(&value, &vert[0], &Dm_(0, i), &vol_[i], &lam_, &miu_);
        break;
      default:
        break;
    }
    *val += w_*value;
  }
  return 0;
}

int elastic_potential::Gra(const double *x, double *gra) const {
  RETURN_WITH_COND_TRUE(w_ == 0.0);
  itr_matrix<const double *> X(3, dim_/3, x);
  itr_matrix<double *> G(3, dim_/3, gra);
  for (size_t i = 0; i < tets_.size(2); ++i) {
    matd_t vert = X(colon(), tets_(colon(), i));
    matd_t grad = zeros<double>(3, 4);
    switch ( type_ ) {
      case LINEAR:
        tet_linear_jac_(&grad[0], &vert[0], &Dm_(0, i), &vol_[i], &lam_, &miu_);
        break;
      case STVK:
        tet_stvk_jac_(&grad[0], &vert[0], &Dm_(0, i), &vol_[i], &lam_, &miu_);
        break;
      case COROTATIONAL:
        break;
      case NEOHOOKEAN:
        tet_neohookean_jac_(&grad[0], &vert[0], &Dm_(0, i), &vol_[i], &lam_, &miu_);
      default:
        break;
    }
    G(colon(), tets_(colon(), i)) += w_*grad;
  }
  return 0;
}

int elastic_potential::Hes(const double *x, vector<Triplet<double>> *hes) const {
  RETURN_WITH_COND_TRUE(w_ == 0.0);
  itr_matrix<const double *> X(3, dim_/3, x);
  for (size_t i = 0; i < tets_.size(2); ++i) {
    matd_t vert = X(colon(), tets_(colon(), i));
    matd_t H = zeros<double>(12, 12);
    switch ( type_ ) {
      case LINEAR:
        tet_linear_hes_(&H[0], nullptr, &Dm_(0, i), &vol_[i], &lam_, &miu_);
        break;
      case STVK:
        tet_stvk_hes_(&H[0], &vert[0], &Dm_(0, i), &vol_[i], &lam_, &miu_);
        break;
      case COROTATIONAL:
        break;
      case NEOHOOKEAN:
        tet_neohookean_(&H[0], &vert[0], &Dm_(0, i), &vol_[i], &lam_, &miu_);
        break;
      default:
        break;
    }
    for (size_t p = 0; p < 12; ++p) {
      for (size_t q = 0; q < 12; ++q) {
        const size_t I = 3*tets_(p/3, i)+p%3;
        const size_t J = 3*tets_(q/3, i)+q%3;
        if ( H(p, q) != 0.0 )
          hes->push_back(Triplet<double>(I, J, w_*H(p, q)));
      }
    }
  }
  return 0;
}
//==============================================================================
voxel_elastic_potential::voxel_elastic_potential(const mati_t &cube, const matd_t &nods, Material type, const double Ym, const double Pr, const double w)
  : dim_(nods.size()), w_(w), type_(type), cube_(cube) {
  h_ = norm(nods(colon(), cube_(0, 0))-nods(colon(), cube(1, 0)));
  lam_ = Ym*Pr/((1.0+Pr)*(1.0-2.0*Pr));
  miu_ = Ym/(2.0*(1.0+Pr));
}

size_t voxel_elastic_potential::Nx() const {
  return dim_;
}

int voxel_elastic_potential::Val(const double *x, double *val) const {
  RETURN_WITH_COND_TRUE(w_ == 0.0);
  itr_matrix<const double *> X(3, dim_/3, x);
  for (size_t i = 0; i < cube_.size(2); ++i) {
    matd_t vert = X(colon(), cube_(colon(), i));
    double value = 0;
    switch ( type_ ) {
      case LINEAR:
        hex_linear_(&value, &vert[0], &h_, &lam_, &miu_);
        break;
      case STVK:
        hex_stvk_(&value, &vert[0], &h_, &lam_, &miu_);
      default:
        break;
    }
    *val += w_*value;
  }
  return 0;
}

int voxel_elastic_potential::Gra(const double *x, double *gra) const {
  RETURN_WITH_COND_TRUE(w_ == 0.0);
  itr_matrix<const double *> X(3, dim_/3, x);
  itr_matrix<double *> G(3, dim_/3, gra);
  for (size_t i = 0; i < cube_.size(2); ++i) {
    matd_t vert = X(colon(), cube_(colon(), i));
    matd_t g = zeros<double>(3, 8);
    switch ( type_ ) {
      case LINEAR:
        hex_linear_jac_(&g[0], &vert[0], &h_, &lam_, &miu_);
        break;
      case STVK:
        hex_stvk_jac_(&g[0], &vert[0], &h_, &lam_, &miu_);
        break;
      default:
        break;
    }
    G(colon(), cube_(colon(), i)) += w_*g;
  }
  return 0;
}

int voxel_elastic_potential::Hes(const double *x, vector<Triplet<double>> *hes) const {
  RETURN_WITH_COND_TRUE(w_ == 0.0);
  itr_matrix<const double *> X(3, dim_/3, x);
  for (size_t i = 0; i < cube_.size(2); ++i) {
    matd_t vert = X(colon(), cube_(colon(), i));
    matd_t H = zeros<double>(24, 24);
    switch ( type_ ) {
      case LINEAR:
        hex_linear_hes_(&H[0], &vert[0], &h_, &lam_, &miu_);
        break;
      case STVK:
        hex_stvk_hes_(&H[0], &vert[0], &h_, &lam_, &miu_);
        break;
      default:
        break;
    }
    for (size_t p = 0; p < 24; ++p) {
      for (size_t q = 0; q < 24; ++q) {
        const size_t I = 3*cube_(p/3, i)+p%3;
        const size_t J = 3*cube_(q/3, i)+q%3;
        if ( H(p, q) != 0.0 )
         hes->push_back(Triplet<double>(I, J, w_*H(p, q)));
      }
    }
  }
  return 0;
}
//==============================================================================
positional_potential::positional_potential(const matd_t &nods, const double w)
  : dim_(nods.size()), w_(w) {}

size_t positional_potential::Nx() const {
  return dim_;
}

int positional_potential::Val(const double *x, double *val) const {
  RETURN_WITH_COND_TRUE(w_ == 0.0 || fixed_.empty());
  Map<const MatrixXd> X(x, 3, dim_/3);
  for (auto &elem : fixed_) {
    const size_t id = elem.first;
    *val += 0.5*w_*(X.col(id)-elem.second).squaredNorm();
  }
  return 0;
}

int positional_potential::Gra(const double *x, double *gra) const {
  RETURN_WITH_COND_TRUE(w_ == 0.0 || fixed_.empty());
  Map<const MatrixXd> X(x, 3, dim_/3);
  Map<MatrixXd> G(gra, 3, dim_/3);
  for (auto &elem : fixed_) {
    const size_t id = elem.first;
    G.col(id) += w_*(X.col(id)-elem.second);
  }
  return 0;
}

int positional_potential::Hes(const double *x, vector<Triplet<double>> *hes) const {
  RETURN_WITH_COND_TRUE(w_ == 0.0 || fixed_.empty());
  for (auto &elem : fixed_) {
    const size_t id = elem.first;
    hes->push_back(Triplet<double>(3*id+0, 3*id+0, w_));
    hes->push_back(Triplet<double>(3*id+1, 3*id+1, w_));
    hes->push_back(Triplet<double>(3*id+2, 3*id+2, w_));
  }
  return 0;
}

int positional_potential::Pin(const size_t id, const double *pos) {
  if ( id < 0 || id >= Nx()/3 )
    return __LINE__;
  fixed_[id] = Vector3d(pos);
  return 0;
}

int positional_potential::Release(const size_t id) {
  if ( id < 0 || id >= Nx()/3 ) {
    cerr << "[info] vertex index is out of range\n";
    return __LINE__;
  }
  auto it = fixed_.find(id);
  if ( it == fixed_.end() ) {
    cerr << "[info] vertex " << id << " is not fixed\n";
    return __LINE__;
  }
  fixed_.erase(it);
  return 0;
}

//==============================================================================
spring_potential::spring_potential(const mati_t &edge, const matd_t &nods, const double w)
  : dim_(nods.size()), w_(w), edge_(edge) {
  len_.resize(edge_.size(2), 1);
#pragma omp parallel for
  for (size_t i = 0; i < edge_.size(2); ++i) {
    matd_t vert = nods(colon(), edge_(colon(), i));
    calc_edge_length_(&len_[i], &vert[0]);
  }
}

size_t spring_potential::Nx() const {
  return dim_;
}

int spring_potential::Val(const double *x, double *val) const {
  RETURN_WITH_COND_TRUE(w_ == 0.0);
  itr_matrix<const double *> X(3, dim_/3, x);
  for (size_t i = 0; i < edge_.size(2); ++i) {
    matd_t vert = X(colon(), edge_(colon(), i));
    double value = 0;
    mass_spring_(&value, &vert[0], &len_[i]);
    *val += w_*value;
  }
  return 0;
}

int spring_potential::Gra(const double *x, double *gra) const {
  RETURN_WITH_COND_TRUE(w_ == 0.0);
  itr_matrix<const double *> X(3, dim_/3, x);
  itr_matrix<double *> G(3, dim_/3, gra);
  for (size_t i = 0; i < edge_.size(2); ++i) {
    matd_t vert = X(colon(), edge_(colon(), i));
    matd_t g = zeros<double>(3, 2);
    mass_spring_jac_(&g[0], &vert[0], &len_[i]);
    G(colon(), edge_(colon(), i)) += w_*g;
  }
  return 0;
}

int spring_potential::Hes(const double *x, vector<Triplet<double>> *hes) const {
  RETURN_WITH_COND_TRUE(w_ == 0.0);
  itr_matrix<const double *> X(3, dim_/3, x);
  for (size_t i = 0; i < edge_.size(2); ++i) {
    matd_t vert = X(colon(), edge_(colon(), i));
    matd_t H = zeros<double>(6, 6);
    double curr = 0;
    calc_edge_length_(&curr, &vert[0]);
    if ( curr < len_[i] ) {
      matd_t g = zeros<double>(6, 1);
      double energy = 0;
      mass_spring_(&energy, &vert[0], &len_[i]);
      if ( false /*energy < 1e-16*/ ) {
        calc_edge_length_jac_(&g[0], &vert[0]);
        g /= sqrt(len_[i]);
      } else {
        mass_spring_jac_(&g[0], &vert[0], &len_[i]);
        g /= 2*sqrt(energy);
      }
      H = 2*g*trans(g);
    } else {
      mass_spring_hes_(&H[0], &vert[0], &len_[i]);
    }
    for (size_t p = 0; p < 6; ++p) {
      for (size_t q = 0; q < 6; ++q) {
        const size_t I = 3*edge_(p/3, i)+p%3;
        const size_t J = 3*edge_(q/3, i)+q%3;
        if ( H(p, q) != 0.0 )
          hes->push_back(Triplet<double>(I, J, w_*H(p, q)));
      }
    }
  }
  return 0;
}
//==============================================================================
fast_mass_spring::fast_mass_spring(const mati_t &edge, const matd_t &nods, const double w)
  : dim_(nods.size()), w_(w), edge_(edge) {
  len_.resize(edge_.size(2), 1);
  d_.resize(3, edge_.size(2));
#pragma omp parallel for
  for (size_t i = 0; i < edge_.size(2); ++i) {
    d_(colon(), i) = nods(colon(), edge_(0, i))-nods(colon(), edge_(1, i));
    len_[i] = norm(d_(colon(), i));
  }
}

size_t fast_mass_spring::Nx() const {
  return dim_;
}

int fast_mass_spring::Val(const double *x, double *val) const {
  itr_matrix<const double *> X(3, dim_/3, x);
  matd_t dx(3, 1);
  for (size_t i = 0; i < edge_.size(2); ++i) {
    dx = X(colon(), edge_(0, i))-X(colon(), edge_(1, i))-d_(colon(), i);
    *val += 0.5*w_*dot(dx, dx);
  }
  return 0;
}

int fast_mass_spring::Gra(const double *x, double *gra) const {
  itr_matrix<const double *> X(3, dim_/3, x);
  itr_matrix<double *> G(3, dim_/3, gra);
  matd_t dx(3, 1);
  for (size_t i = 0; i < edge_.size(2); ++i) {
    dx = X(colon(), edge_(0, i))-X(colon(), edge_(1, i))-d_(colon(), i);
    G(colon(), edge_(0, i)) += w_*dx;
    G(colon(), edge_(1, i)) -= w_*dx;
  }
  return 0;
}

int fast_mass_spring::Hes(const double *x, vector<Triplet<double>> *hes) const {
  for (size_t i = 0; i < edge_.size(2); ++i) {
    add_diag_block<double, 3>(edge_(0, i), edge_(0, i), w_, hes);
    add_diag_block<double, 3>(edge_(0, i), edge_(1, i), -w_, hes);
    add_diag_block<double, 3>(edge_(1, i), edge_(0, i), -w_, hes);
    add_diag_block<double, 3>(edge_(1, i), edge_(1, i), w_, hes);
  }
  return 0;
}

void fast_mass_spring::LocalSolve(const double *x) {
  itr_matrix<const double *> X(3, dim_/3, x);
#pragma omp parallel for
  for (size_t i = 0; i < edge_.size(2); ++i) {
    d_(colon(), i) = X(colon(), edge_(0, i))-X(colon(), edge_(1, i));
    double dnorm = norm(d_(colon(), i));
    d_(colon(), i) *= len_[i]/dnorm;
  }
}
//==============================================================================
surf_bending_potential::surf_bending_potential(const mati_t &diams, const matd_t &nods, const double w)
  : dim_(nods.size()), diams_(diams), w_(w) {
  len_.resize(diams_.size(2), 1);
  angle_.resize(diams_.size(2), 1);
  area_.resize(diams_.size(2), 1);
#pragma omp parallel for
  for (size_t i = 0; i < diams_.size(2); ++i) {
    matd_t vert = nods(colon(), diams_(colon(), i));
    calc_edge_length_(&len_[i], &vert(0, 1));
    calc_dih_angle_(&angle_[i], &vert[0]);
    area_[i] = calc_tri_area(vert(colon(), colon(0, 2)))+calc_tri_area(vert(colon(), colon(1, 3)));
  }
}

size_t surf_bending_potential::Nx() const {
  return dim_;
}

int surf_bending_potential::Val(const double *x, double *val) const {
  RETURN_WITH_COND_TRUE(w_ == 0.0);
  itr_matrix<const double *> X(3, Nx()/3, x);
  for (size_t i = 0; i < diams_.size(2); ++i) {
    matd_t vert = X(colon(), diams_(colon(), i));
    double value = 0;
    surf_bending_(&value, &vert[0], &angle_[i], &len_[i], &area_[i]);
    *val += w_*value;
  }
  return 0;
}

int surf_bending_potential::Gra(const double *x, double *gra) const {
  RETURN_WITH_COND_TRUE(w_ == 0.0);
  itr_matrix<const double *> X(3, Nx()/3, x);
  itr_matrix<double *> G(3, Nx()/3, gra);
  for (size_t i = 0; i < diams_.size(2); ++i) {
    matd_t vert = X(colon(), diams_(colon(), i));
    matd_t grad = zeros<double>(3, 4);
    surf_bending_jac_(&grad[0], &vert[0], &angle_[i], &len_[i], &area_[i]);
    G(colon(), diams_(colon(), i)) += w_*grad;
  }
  return 0;
}

int surf_bending_potential::Hes(const double *x, vector<Triplet<double>> *hes) const {
  RETURN_WITH_COND_TRUE(w_ == 0.0);
  itr_matrix<const double *> X(3, Nx()/3, x);
  for (size_t i = 0; i < diams_.size(2); ++i) {
    matd_t vert = X(colon(), diams_(colon(), i));
    matd_t H = zeros<double>(12, 12);
    surf_bending_hes_(&H[0], &vert[0], &angle_[i], &len_[i], &area_[i]);
    for (size_t p = 0; p < 12; ++p) {
      for (size_t q = 0; q < 12; ++q) {
        if ( H(p, q) != 0.0 ) {
          const size_t I = 3*diams_(p/3, i)+p%3;
          const size_t J = 3*diams_(q/3, i)+q%3;
          hes->push_back(Triplet<double>(I, J, w_*H(p, q)));
        }
      }
    }
  }
  return 0;
}
//==============================================================================
isometric_bending::isometric_bending(const mati_t &diams, const matd_t &nods, const double w)
  : diams_(diams), w_(w), dim_(nods.size()) {
  area_.resize(diams_.size(2), 1);
  cotv_.resize(4, diams_.size(2));
#pragma omp parallel for
  for (size_t i = 0; i < diams_.size(2); ++i) {
    matd_t vert = nods(colon(), diams_(colon(), i));
    area_[i] = calc_tri_area(vert(colon(), colon(0, 2)))+calc_tri_area(vert(colon(), colon(1, 3)));
    double v012 = cal_cot_val(&vert(0, 0), &vert(0, 1), &vert(0, 2));
    double v021 = cal_cot_val(&vert(0, 0), &vert(0, 2), &vert(0, 1));
    double v312 = cal_cot_val(&vert(0, 3), &vert(0, 1), &vert(0, 2));
    double v321 = cal_cot_val(&vert(0, 3), &vert(0, 2), &vert(0, 1));
    cotv_(0, i) = -v012-v021;
    cotv_(1, i) = v021+v321;
    cotv_(2, i) = v012+v312;
    cotv_(3, i) = -v312-v321;
  }
}

size_t isometric_bending::Nx() const {
  return dim_;
}

int isometric_bending::Val(const double *x, double *val) const {
  RETURN_WITH_COND_TRUE(w_ == 0.0);
  itr_matrix<const double *> X(3, dim_/3, x);
  for (size_t i = 0; i < diams_.size(2); ++i) {
    matd_t vert = X(colon(), diams_(colon(), i));
    matd_t lx = vert*cotv_(colon(), i);
    *val += 0.5*w_*3/area_[i]*dot(lx, lx);
  }
  return 0;
}

int isometric_bending::Gra(const double *x, double *gra) const {
  RETURN_WITH_COND_TRUE(w_ == 0.0);
  itr_matrix<const double *> X(3, dim_/3, x);
  itr_matrix<double *> G(3, dim_/3, gra);
  for (size_t i = 0; i < diams_.size(2); ++i) {
    matd_t vert = X(colon(), diams_(colon(), i));
    matd_t lx = vert*cotv_(colon(), i);
    G(colon(), diams_(colon(), i)) += w_*3/area_[i]*(lx*trans(cotv_(colon(), i)));
  }
  return 0;
}

int isometric_bending::Hes(const double *x, vector<Triplet<double>> *hes) const {
  RETURN_WITH_COND_TRUE(w_ == 0.0);
  for (size_t i = 0; i < diams_.size(2); ++i) {
    matd_t H = w_*3/area_[i]*cotv_(colon(), i)*temp(trans(cotv_(colon(), i)));
    for (size_t p = 0; p < 4; ++p) {
      for (size_t q = 0; q < 4; ++q) {
        add_diag_block<double, 3>(diams_(p, i), diams_(q, i), H(p, q), hes);
      }
    }
  }
  return 0;
}
//==============================================================================
ext_force_energy::ext_force_energy(const matd_t &nods, const double w)
  : dim_(nods.size()), w_(w) {
  force_ = zeros<double>(dim_, 1);
}

size_t ext_force_energy::Nx() const {
  return dim_;
}

int ext_force_energy::Val(const double *x, double *val) const {
  itr_matrix<const double *> X(Nx(), 1, x);
  *val += -w_*dot(force_, X);
  return 0;
}

int ext_force_energy::Gra(const double *x, double *gra) const {
  itr_matrix<const double *> X(Nx(), 1, x);
  itr_matrix<double *> G(Nx(), 1, gra);
  G += w_*-force_;
  return 0;
}

int ext_force_energy::ApplyForce(const size_t id, const double *f) {
  if ( id < 0 || id >= Nx()/3 )
    return __LINE__;
  std::copy(f, f+3, &force_[3*id]);
  return 0;
}

int ext_force_energy::RemoveForce(const size_t id) {
  if ( id < 0 || id >= Nx()/3 )
    return __LINE__;
  std::fill(&force_[3*id+0], &force_[3*id+3], 0);
  return 0;
}

//==============================================================================
line_bending_potential::line_bending_potential(const mati_t &edge, const matd_t &nods, const double w)
  : dim_(nods.size()), w_(w), edge_(edge) {
  len_.resize(edge_.size(2), 1);
#pragma omp parallel for
  for (size_t i = 0; i < edge_.size(2); ++i) {
    matd_t vert = nods(colon(), edge_(colon(), i));
    calc_edge_length_(&len_[i], &vert[0]);
  }
}

size_t line_bending_potential::Nx() const {
  return dim_;
}

int line_bending_potential::Val(const double *x, double *val) const {
  RETURN_WITH_COND_TRUE(w_ == 0.0);
  itr_matrix<const double *> X(3, dim_/3, x);
  for (size_t i = 0; i < edge_.size(2)-1; ++i) {
    matd_t vert(3, 4);
    vert(colon(), colon(0, 1)) = X(colon(), edge_(colon(), i));
    vert(colon(), colon(2, 3)) = X(colon(), edge_(colon(), i+1));
    double value = 0;
    line_bending_(&value, &vert[0], &len_[i], &len_[i+1]);
    *val += w_*value;
  }
  return 0;
}

int line_bending_potential::Gra(const double *x, double *gra) const {
  RETURN_WITH_COND_TRUE(w_ == 0.0);
  itr_matrix<const double *> X(3, dim_/3, x);
  itr_matrix<double *> G(3, dim_/3, gra);
  for (size_t i = 0; i < edge_.size(2)-1; ++i) {
    matd_t vert(3, 4);
    vert(colon(), colon(0, 1)) = X(colon(), edge_(colon(), i));
    vert(colon(), colon(2, 3)) = X(colon(), edge_(colon(), i+1));
    matd_t g(3, 4);
    line_bending_jac_(&g[0], &vert[0], &len_[i], &len_[i+1]);
    G(colon(), edge_(colon(), i)) += w_*g(colon(), colon(0, 1));
    G(colon(), edge_(colon(), i+1)) += w_*g(colon(), colon(2, 3));
  }
  return 0;
}

int line_bending_potential::Hes(const double *x, vector<Triplet<double>> *hes) const {
  RETURN_WITH_COND_TRUE(w_ == 0.0);
  itr_matrix<const double *> X(3, dim_/3, x);
  for (size_t i = 0; i < edge_.size(2)-1; ++i) {
    matd_t vert(3, 4);
    vert(colon(), colon(0, 1)) = X(colon(), edge_(colon(), i));
    vert(colon(), colon(2, 3)) = X(colon(), edge_(colon(), i+1));
    matd_t H(12, 12);
    line_bending_hes_(&H[0], &vert[0], &len_[i], &len_[i+1]);
    for (size_t p = 0; p < 12; ++p) {
      for (size_t q = 0; q < 12; ++q) {
        const size_t I = 3*edge_((p%6)/3, i+p/6)+p%3;
        const size_t J = 3*edge_((q%6)/3, i+q/6)+q%3;
        if ( H(p, q) != 0.0 )
          hes->push_back(Triplet<double>(I, J, w_*H(p, q)));
      }
    }
  }
  return 0;
}
//==============================================================================
position_constraint::position_constraint(const matd_t &nods, const double w)
  : w_(sqrt(w)), dim_(nods.size()) {}

int position_constraint::Val(const double *x, double *val) const {
  RETURN_WITH_COND_TRUE(Nf() == 0);
  Map<const MatrixXd> X(x, 3, Nx()/3);
  Map<VectorXd> value(val, Nf());
  size_t cnt = 0;
  for (auto &elem : fixed_) {
    value.segment<3>(3*cnt) += w_*(X.col(elem.first)-elem.second);
    ++cnt;
  }
  return 0;
}

int position_constraint::Jac(const double *x, const size_t off, vector<Triplet<double>> *jac) const {
  RETURN_WITH_COND_TRUE(Nf() == 0);
  size_t cnt = 0;
  for (auto &elem : fixed_) {
    jac->push_back(Triplet<double>(off+cnt++, 3*elem.first+0, w_));
    jac->push_back(Triplet<double>(off+cnt++, 3*elem.first+1, w_));
    jac->push_back(Triplet<double>(off+cnt++, 3*elem.first+2, w_));
  }
  return 0;
}

int position_constraint::Pin(const size_t pid, const double *pos) {
  if ( pid < 0 || pid >= Nx()/3 ) {
    cerr << "[info] point ID is out of range\n";
    return __LINE__;
  }
  fixed_[pid] = Vector3d(pos);
  return 0;
}

int position_constraint::Release(const size_t pid) {
  if ( pid < 0 || pid >= Nx()/3 ) {
    cerr << "[info] point ID is out of range\n";
    return __LINE__;
  }
  auto it = fixed_.find(pid);
  if ( it == fixed_.end() ) {
    cerr << "[info] point " << pid << " is not fixed\n";
    return __LINE__;
  }
  fixed_.erase(it);
  return 0;
}

//==============================================================================
inextensible_constraint::inextensible_constraint(const mati_t &edge, const matd_t &nods, const double w)
  : w_(w), dim_(nods.size()), edges_(edge) {
#pragma omp parallel for
  for (size_t i = 0; i < edges_.size(2); ++i) {
    matd_t vert = nods(colon(), edges_(colon(), i));
    calc_edge_length_(&len_[i], &vert[0]);
  }
}

int inextensible_constraint::Val(const double *x, double *val) const {
  itr_matrix<const double *> X(3, Nx()/3, x);
#pragma omp parallel for
  for (size_t i = 0; i < edges_.size(2); ++i) {
    matd_t vert = X(colon(), edges_(colon(), i));
    double curr = 0;
    calc_edge_length_(&curr, &vert[0]);
    val[i] += curr-len_[i];
  }
  return 0;
}

int inextensible_constraint::Jac(const double *x, const size_t off, vector<Triplet<double>> *jac) const {
  return 0;
}
//==============================================================================
}
