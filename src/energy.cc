#include "energy.h"

#include <iostream>
#include <hjlib/math/blas_lapack.h>
#include <zjucad/matrix/lapack.h>
#include <zjucad/matrix/itr_matrix.h>
#include <zjucad/matrix/io.h>
#include <unsupported/Eigen/KroneckerProduct>
#include <jtflib/mesh/util.h>

#include "mass_matrix.h"
#include "config.h"
#include "geom_util.h"
#include "util.h"
#include "mesh_partition.h"

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

void tet_arap_(double *val, const double *x, const double *Dm, const double *R, const double *vol);
void tet_arap_jac_(double *jac, const double *x, const double *Dm, const double *R, const double *vol);
void tet_arap_hes_(double *hes, const double *x, const double *Dm, const double *R, const double *vol);

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

void bw98_stretch_(double *val, const double *x, const double *invUV, const double *area);
void bw98_stretch_jac_(double *jac, const double *x, const double *invUV, const double *area);
void bw98_stretch_hes_(double *hes, const double *x, const double *invUV, const double *area);

void bw98_shear_(double *val, const double *x, const double *invUV, const double *area);
void bw98_shear_jac_(double *jac, const double *x, const double *invUV, const double *area);
void bw98_shear_hes_(double *hes, const double *x, const double *invUV, const double *area);

void line_bending_(double *val, const double *x, const double *d1, const double *d2);
void line_bending_jac_(double *jac, const double *x, const double *d1, const double *d2);
void line_bending_hes_(double *hes, const double *x, const double *d1, const double *d2);

}

typedef double scalarD;
void const_len_spring(double *out, const double *x, const double *l0) {
  //input
  scalarD a1 = x[0];
  scalarD a2 = x[1];
  scalarD a3 = x[2];
  scalarD b1 = x[3];
  scalarD b2 = x[4];
  scalarD b3 = x[5];
  scalarD r = *l0;

  //temp
  scalarD tt1;
  scalarD tt2;
  scalarD tt3;
  scalarD tt4;

  tt1=a1-b1;
  tt2=a2-b2;
  tt3=a3-b3;
  tt4=1/sqrt(pow(tt3,2)+pow(tt2,2)+pow(tt1,2));
  out[0]=tt1*tt4*r;
  out[1]=tt2*tt4*r;
  out[2]=tt4*tt3*r;
}
void const_len_spring_jac(double *out, const double *x, const double *l0) {
  //input
  scalarD a1 = x[0];
  scalarD a2 = x[1];
  scalarD a3 = x[2];
  scalarD b1 = x[3];
  scalarD b2 = x[4];
  scalarD b3 = x[5];
  scalarD r = *l0;

  //temp
  scalarD tt1;
  scalarD tt2;
  scalarD tt3;
  scalarD tt4;
  scalarD tt5;
  scalarD tt6;
  scalarD tt7;
  scalarD tt8;
  scalarD tt9;
  scalarD tt10;
  scalarD tt11;
  scalarD tt12;
  scalarD tt13;
  scalarD tt14;
  scalarD tt15;
  scalarD tt16;
  scalarD tt17;

  tt1=a1-b1;
  tt2=pow(tt1,2);
  tt3=a2-b2;
  tt4=pow(tt3,2);
  tt5=a3-b3;
  tt6=pow(tt5,2);
  tt7=sqrt(tt6+tt4+tt2);
  tt8=1/pow(tt7,3);
  tt9=1/tt7;
  tt10=tt9*r;
  tt11=-tt1*tt3*tt8*r;
  tt12=-tt1*tt8*tt5*r;
  tt13=-tt3*tt8*tt5*r;
  tt14=-tt9*r;
  tt15=tt1*tt3*tt8*r;
  tt16=tt1*tt8*tt5*r;
  tt17=tt3*tt8*tt5*r;
  out[0]=tt10-tt2*tt8*r;
  out[1]=tt11;
  out[2]=tt12;
  out[3]=tt11;
  out[4]=tt10-tt4*tt8*r;
  out[5]=tt13;
  out[6]=tt12;
  out[7]=tt13;
  out[8]=tt10-tt8*tt6*r;
  out[9]=tt14+tt2*tt8*r;
  out[10]=tt15;
  out[11]=tt16;
  out[12]=tt15;
  out[13]=tt14+tt4*tt8*r;
  out[14]=tt17;
  out[15]=tt16;
  out[16]=tt17;
  out[17]=tt8*tt6*r+tt14;
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
      if ( energy < 1e-16 ) {
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
  S_.resize(aux_dim(), Nx()); {
    vector<Triplet<double>> trips;
    for (size_t i = 0; i < edge_.size(2); ++i) {
      trips.push_back(Triplet<double>(3*i+0, 3*edge_(0, i)+0, 1.0));
      trips.push_back(Triplet<double>(3*i+0, 3*edge_(1, i)+0, -1.0));
      trips.push_back(Triplet<double>(3*i+1, 3*edge_(0, i)+1, 1.0));
      trips.push_back(Triplet<double>(3*i+1, 3*edge_(1, i)+1, -1.0));
      trips.push_back(Triplet<double>(3*i+2, 3*edge_(0, i)+2, 1.0));
      trips.push_back(Triplet<double>(3*i+2, 3*edge_(1, i)+2, -1.0));
    }
    S_.reserve(trips.size());
    S_.setFromTriplets(trips.begin(), trips.end());
  }
}

size_t fast_mass_spring::Nx() const {
  return dim_;
}

int fast_mass_spring::Val(const double *x, double *val) const {
  RETURN_WITH_COND_TRUE(w_ == 0.0);
  itr_matrix<const double *> X(3, dim_/3, x);
  matd_t dx(3, 1);
  for (size_t i = 0; i < edge_.size(2); ++i) {
    dx = X(colon(), edge_(0, i))-X(colon(), edge_(1, i))-d_(colon(), i);
    *val += 0.5*w_*dot(dx, dx);
  }
  return 0;
}

int fast_mass_spring::Gra(const double *x, double *gra) const {
  RETURN_WITH_COND_TRUE(w_ == 0.0);
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
  RETURN_WITH_COND_TRUE(w_ == 0.0);
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

void fast_mass_spring::Project() {
#pragma omp parallel for
  for (size_t i = 0; i < d_.size(2); ++i) {
    double dnorm = norm(d_(colon(), i));
    d_(colon(), i) *= len_[i]/dnorm;
  }
}
//==============================================================================
modified_fms_energy::modified_fms_energy(const mati_t &edge, const matd_t &nods, const double w)
  : dim_(nods.size()), w_(w), edge_(edge) {
  len_.resize(edge_.size(2), 1);
  d_.resize(3, edge_.size(2));
#pragma omp parallel for
  for (size_t i = 0; i < edge_.size(2); ++i) {
    d_(colon(), i) = nods(colon(), edge_(0, i))-nods(colon(), edge_(1, i));
    len_[i] = norm(d_(colon(), i));
  }
  S_.resize(aux_dim(), Nx()); {
    vector<Triplet<double>> trips;
    for (size_t i = 0; i < edge_.size(2); ++i) {
      trips.push_back(Triplet<double>(3*i+0, 3*edge_(0, i)+0, 1.0));
      trips.push_back(Triplet<double>(3*i+0, 3*edge_(1, i)+0, -1.0));
      trips.push_back(Triplet<double>(3*i+1, 3*edge_(0, i)+1, 1.0));
      trips.push_back(Triplet<double>(3*i+1, 3*edge_(1, i)+1, -1.0));
      trips.push_back(Triplet<double>(3*i+2, 3*edge_(0, i)+2, 1.0));
      trips.push_back(Triplet<double>(3*i+2, 3*edge_(1, i)+2, -1.0));
    }
    S_.reserve(trips.size());
    S_.setFromTriplets(trips.begin(), trips.end());
  }
}

size_t modified_fms_energy::Nx() const {
  return dim_;
}

int modified_fms_energy::Val(const double *x, double *val) const {
  RETURN_WITH_COND_TRUE(w_ == 0.0);
  itr_matrix<const double *> X(3, dim_/3, x);
  matd_t dx(3, 1);
  for (size_t i = 0; i < edge_.size(2); ++i) {
    dx = X(colon(), edge_(0, i))-X(colon(), edge_(1, i))-d_(colon(), i);
    *val += 0.5*w_*dot(dx, dx);
  }
  return 0;
}

int modified_fms_energy::Gra(const double *x, double *gra) const {
  RETURN_WITH_COND_TRUE(w_ == 0.0);
  itr_matrix<const double *> X(3, dim_/3, x);
  itr_matrix<double *> G(3, dim_/3, gra);
  // \nabla f = 0 => (S^T-J)(Sx-d) = 0
  matd_t ST(6, 3), JT(6, 3), vert(3, 2), p(3, 1), J(3, 6), grad(6, 1);
  ST(colon(0, 2), colon()) = eye<double>(3);
  ST(colon(3, 5), colon()) = -eye<double>(3);
  for (size_t i = 0; i < edge_.size(2); ++i) {
    vert = X(colon(), edge_(colon(), i));
    p = vert(colon(), 0)-vert(colon(), 1);
    const_len_spring_jac(&J[0], &vert[0], &len_[i]);
    JT = trans(J);
    grad = (ST-JT)*(p-d_(colon(), i));
    G(colon(), edge_(colon(), i)) += w_*itr_matrix<const double *>(3, 2, &grad[0]);
  }
  return 0;
}

int modified_fms_energy::Hes(const double *x, vector<Triplet<double>> *hes) const {
  RETURN_WITH_COND_TRUE(w_ == 0.0);
  for (size_t i = 0; i < edge_.size(2); ++i) {
    add_diag_block<double, 3>(edge_(0, i), edge_(0, i), w_, hes);
    add_diag_block<double, 3>(edge_(0, i), edge_(1, i), -w_, hes);
    add_diag_block<double, 3>(edge_(1, i), edge_(0, i), -w_, hes);
    add_diag_block<double, 3>(edge_(1, i), edge_(1, i), w_, hes);
  }
  return 0;
}

void modified_fms_energy::LocalSolve(const double *x) {
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
tet_arap_energy::tet_arap_energy(const mati_t &tets, const matd_t &nods, const double w)
  : dim_(nods.size()), tets_(tets), w_(w) {
  vol_.resize(tets_.size(2), 1);
  R_.resize(9, tets_.size(2));
  D_.resize(9, tets_.size(2));
#pragma omp parallel for
  for (size_t i = 0; i < tets_.size(2); ++i) {
    matd_t basis = nods(colon(), tets_(colon(1, 3), i))-nods(colon(), tets_(0, i))*ones<double>(1, 3);
    Map<Matrix3d> B(basis.begin());
    Map<Matrix3d>(&D_(0, i)) = B.inverse();
    vol_[i] = std::fabs(B.determinant())/6.0;
    Map<Matrix3d>(&R_(0, i)) = Matrix3d::Identity();
  }
}

size_t tet_arap_energy::Nx() const {
  return dim_;
}

int tet_arap_energy::Val(const double *x, double *val) const {
  RETURN_WITH_COND_TRUE(w_ == 0.0);
  itr_matrix<const double *> X(3, dim_/3, x);
  for (size_t i = 0; i < tets_.size(2); ++i) {
    matd_t vert = X(colon(), tets_(colon(), i));
    double value = 0;
    tet_arap_(&value, &vert[0], &D_(0, i), &R_(0, i), &vol_[i]);
    *val += w_*value;
  }
  return 0;
}

int tet_arap_energy::Gra(const double *x, double *gra) const {
  RETURN_WITH_COND_TRUE(w_ == 0.0);
  itr_matrix<const double *> X(3, dim_/3, x);
  itr_matrix<double *> G(3, dim_/3, gra);
  for (size_t i = 0; i < tets_.size(2); ++i) {
    matd_t vert = X(colon(), tets_(colon(), i));
    matd_t jac = zeros<double>(3, 4);
    tet_arap_jac_(&jac[0], &vert[0], &D_(0, i), &R_(0, i), &vol_[i]);
    G(colon(), tets_(colon(), i)) += w_*jac;
  }
  return 0;
}

int tet_arap_energy::Hes(const double *x, vector<Triplet<double>> *hes) const {
  RETURN_WITH_COND_TRUE(w_ == 0.0);
  for (size_t i = 0; i < tets_.size(2); ++i) {
    matd_t H = zeros<double>(12, 12);
    tet_arap_hes_(&H[0], nullptr, &D_(0, i), &R_(0, i), &vol_[i]);
    for (size_t p = 0; p < 12; ++p) {
      for (size_t q = 0; q < 12; ++q) {
        const size_t I = 3*tets_(p/3, i)+p%3;
        const size_t J = 3*tets_(q/3, i)+q%3;
        hes->push_back(Triplet<double>(I, J, w_*H(p, q)));
      }
    }
  }
  return 0;
}

void tet_arap_energy::LocalSolve(const double *x) {
  itr_matrix<const double *> X(3, dim_/3, x);
#pragma omp parallel for
  for (size_t i = 0; i < tets_.size(2); ++i) {
    matd_t Ds = X(colon(), tets_(colon(1, 3), i))-X(colon(), tets_(0, i))*ones<double>(1, 3);
    matd_t df = Ds*itr_matrix<const double *>(3, 3, &D_(0, i));
    Map<Matrix3d> F(df.begin());
    JacobiSVD<Matrix3d> svd(F, ComputeFullU|ComputeFullV);
    Map<Matrix3d>(&R_(0, i)) = svd.matrixU()*svd.matrixV().transpose();
  }
}
//==============================================================================
bw98_stretch_energy::bw98_stretch_energy(const mati_t &tris, const matd_t &nods, const double w)
  : dim_(nods.size()), w_(w), tris_(tris) {
  // build local frame
  matd_t O(3, tris_.size(2)), T(3, tris_.size(2)), B(3, tris_.size(2)), N(3, tris_.size(2));
  jtf::mesh::cal_face_normal(tris, nods, N, true);
#pragma omp parallel for
  for (size_t i = 0; i < tris_.size(2); ++i) {
    const matd_t vert = nods(colon(), tris_(colon(), i));
    O(colon(), i) = vert*ones<double>(3, 1)/3.0;
    T(colon(), i) = vert(colon(), 1)-vert(colon(), 0);
    T(colon(), i) /= norm(T(colon(), i));
    B(colon(), i) = cross(N(colon(), i), T(colon(), i));
    B(colon(), i) /= norm(B(colon(), i));
  }
  // calc inv(u, v) and area
  invUV_.resize(4, tris_.size(2));
  area_.resize(tris_.size(2), 1);
#pragma omp parallel for
  for (size_t i = 0; i < tris_.size(2); ++i) {
    const matd_t vert = nods(colon(), tris_(colon(), i));
    matd_t uv(2, 3);
    for (size_t j = 0; j < 3; ++j) {
      uv(0, j) = dot(vert(colon(), j)-O(colon(), i), T(colon(), i));
      uv(1, j) = dot(vert(colon(), j)-O(colon(), i), B(colon(), i));
    }
    matd_t base = uv(colon(), colon(1, 2))-uv(colon(), 0)*ones<double>(1, 2);
    if ( inv(base) ) {
      cerr << "\t@degenerated triangle " << i << endl;
      exit(EXIT_FAILURE);
    }
    std::copy(base.begin(), base.end(), &invUV_(0, i));
    area_[i] = calc_tri_area(vert);
  }
}

size_t bw98_stretch_energy::Nx() const {
  return dim_;
}

int bw98_stretch_energy::Val(const double *x, double *val) const {
  RETURN_WITH_COND_TRUE(w_ == 0.0);
  itr_matrix<const double *> X(3, dim_/3, x);
  for (size_t i = 0; i < tris_.size(2); ++i) {
    matd_t vert = X(colon(), tris_(colon(), i));
    double value = 0;
    bw98_stretch_(&value, &vert[0], &invUV_(0, i), &area_[i]);
    *val += w_*value;
  }
  return 0;
}

int bw98_stretch_energy::Gra(const double *x, double *gra) const {
  RETURN_WITH_COND_TRUE(w_ == 0.0);
  itr_matrix<const double *> X(3, dim_/3, x);
  itr_matrix<double *> G(3, dim_/3, gra);
  for (size_t i = 0; i < tris_.size(2); ++i) {
    matd_t vert = X(colon(), tris_(colon(), i));
    matd_t jac = zeros<double>(3, 3);
    bw98_stretch_jac_(&jac[0], &vert[0], &invUV_(0, i), &area_[i]);
    G(colon(), tris_(colon(), i)) += w_*jac;
  }
  return 0;
}

int bw98_stretch_energy::Hes(const double *x, vector<Triplet<double>> *hes) const {
  RETURN_WITH_COND_TRUE(w_ == 0.0);
  itr_matrix<const double *> X(3, dim_/3, x);
  for (size_t i = 0; i < tris_.size(2); ++i) {
    matd_t vert = X(colon(), tris_(colon(), i));
    matd_t H = zeros<double>(9, 9);
    bw98_stretch_hes_(&H[0], &vert[0], &invUV_(0, i), &area_[i]);
    for (size_t p = 0; p < 9; ++p) {
      for (size_t q = 0; q < 9; ++q) {
        const size_t I = 3*tris_(p/3, i)+p%3;
        const size_t J = 3*tris_(q/3, i)+q%3;
        hes->push_back(Triplet<double>(I, J, w_*H(p, q)));
      }
    }
  }
  return 0;
}
//==============================================================================
bw98_shear_energy::bw98_shear_energy(const mati_t &tris, const matd_t &nods, const double w)
  : dim_(nods.size()), w_(w), tris_(tris) {
  // build local frame
  matd_t O(3, tris_.size(2)), T(3, tris_.size(2)), B(3, tris_.size(2)), N(3, tris_.size(2));
  jtf::mesh::cal_face_normal(tris, nods, N, true);
#pragma omp parallel for
  for (size_t i = 0; i < tris_.size(2); ++i) {
    const matd_t vert = nods(colon(), tris_(colon(), i));
    O(colon(), i) = vert*ones<double>(3, 1)/3.0;
    T(colon(), i) = vert(colon(), 1)-vert(colon(), 0);
    T(colon(), i) /= norm(T(colon(), i));
    B(colon(), i) = cross(N(colon(), i), T(colon(), i));
    B(colon(), i) /= norm(B(colon(), i));
  }
  // calc inv(u, v) and area
  invUV_.resize(4, tris_.size(2));
  area_.resize(tris_.size(2), 1);
#pragma omp parallel for
  for (size_t i = 0; i < tris_.size(2); ++i) {
    const matd_t vert = nods(colon(), tris_(colon(), i));
    matd_t uv(2, 3);
    for (size_t j = 0; j < 3; ++j) {
      uv(0, j) = dot(vert(colon(), j)-O(colon(), i), T(colon(), i));
      uv(1, j) = dot(vert(colon(), j)-O(colon(), i), B(colon(), i));
    }
    matd_t base = uv(colon(), colon(1, 2))-uv(colon(), 0)*ones<double>(1, 2);
    inv(base);
    std::copy(base.begin(), base.end(), &invUV_(0, i));
    area_[i] = calc_tri_area(vert);
  }
}

size_t bw98_shear_energy::Nx() const {
  return dim_;
}

int bw98_shear_energy::Val(const double *x, double *val) const {
  RETURN_WITH_COND_TRUE(w_ == 0.0);
  itr_matrix<const double *> X(3, dim_/3, x);
  for (size_t i = 0; i < tris_.size(2); ++i) {
    matd_t vert = X(colon(), tris_(colon(), i));
    double value = 0;
    bw98_shear_(&value, &vert[0], &invUV_(0, i), &area_[i]);
    *val += w_*value;
  }
  return 0;
}

int bw98_shear_energy::Gra(const double *x, double *gra) const {
  RETURN_WITH_COND_TRUE(w_ == 0.0);
  itr_matrix<const double *> X(3, dim_/3, x);
  itr_matrix<double *> G(3, dim_/3, gra);
  for (size_t i = 0; i < tris_.size(2); ++i) {
    matd_t vert = X(colon(), tris_(colon(), i));
    matd_t jac = zeros<double>(3, 3);
    bw98_shear_jac_(&jac[0], &vert[0], &invUV_(0, i), &area_[i]);
    G(colon(), tris_(colon(), i)) += w_*jac;
  }
  return 0;
}

int bw98_shear_energy::Hes(const double *x, vector<Triplet<double>> *hes) const {
  RETURN_WITH_COND_TRUE(w_ == 0.0);
  itr_matrix<const double *> X(3, dim_/3, x);
  for (size_t i = 0; i < tris_.size(2); ++i) {
    matd_t vert = X(colon(), tris_(colon(), i));
    matd_t H = zeros<double>(9, 9);
    bw98_shear_hes_(&H[0], &vert[0], &invUV_(0, i), &area_[i]);
    for (size_t p = 0; p < 9; ++p) {
      for (size_t q = 0; q < 9; ++q) {
        const size_t I = 3*tris_(p/3, i)+p%3;
        const size_t J = 3*tris_(q/3, i)+q%3;
        hes->push_back(Triplet<double>(I, J, w_*H(p, q)));
      }
    }
  }
  return 0;
}
//==============================================================================
low_pass_filter_energy::low_pass_filter_energy(const mati_t &tris, const matd_t &nods, const size_t patch_num, const double w)
  : dim_(nods.size()), w_(w), tris_(tris), patch_num_(patch_num) {
  mesh_partition mp(tris, nods);
  vector<ptn_to_patch> info;
  mp.init(info);
  mp.run(patch_num_, info);
  // allocate patches
  pat_.resize(mp.get_actual_patch_num());
  for (size_t i = 0; i < info.size(); ++i) {
    pat_[info[i].id_patch].push_back(make_pair(i, info[i].dist));
  }
  // calc coef
  double sigma = 10;//2*()*pow(1, 1.0/3);
#pragma omp parallel for
  for (size_t i = 0; i < pat_.size(); ++i) {
    double sum = 0;
    for (size_t j = 0; j < pat_[i].size(); ++j) {
      double d = pat_[i][j].second;
      if ( d < 2*sigma )
        pat_[i][j].second = exp(-0.5*d*d/(sigma*sigma));
      else
        pat_[i][j].second = 0;
      sum += pat_[i][j].second;
    }
    for (size_t j = 0; j < pat_[i].size(); ++j) {
      pat_[i][j].second /= sum;
    }
  }
}

size_t low_pass_filter_energy::Nx() const {
  return dim_;
}

int low_pass_filter_energy::Val(const double *x, double *val) const {
  RETURN_WITH_COND_TRUE(w_ == 0.0);
  Map<const MatrixXd> X(x, 3, dim_/3);
  Map<const MatrixXd> Q(ref_, 3, dim_/3);
  for (auto &arr : pat_) {
    Vector3d lhs = Vector3d::Zero(), rhs = Vector3d::Zero();
    for (auto &elem: arr) {
      const size_t pi = elem.first;
      lhs += elem.second*X.col(pi);
      rhs += elem.second*Q.col(pi);
    }
    *val += 0.5*w_*(lhs-rhs).squaredNorm();
  }
  return 0;
}

int low_pass_filter_energy::Gra(const double *x, double *gra) const {
  RETURN_WITH_COND_TRUE(w_ == 0.0);
  Map<const MatrixXd> X(x, 3, dim_/3);
  Map<const MatrixXd> Q(ref_, 3, dim_/3);
  Map<MatrixXd> G(gra, 3, dim_/3);
  for (auto &arr : pat_) {
    Vector3d lhs = Vector3d::Zero(), rhs = Vector3d::Zero();
    for (auto &elem : arr) {
      const size_t pi = elem.first;
      lhs += elem.second*X.col(pi);
      rhs += elem.second*Q.col(pi);
    }
    for (auto &elem : arr) {
      G.col(elem.first) += w_*elem.second*(lhs-rhs);
    }
  }
  return 0;
}

int low_pass_filter_energy::Hes(const double *x, vector<Triplet<double>> *hes) const {
  RETURN_WITH_COND_TRUE(w_ == 0.0);
  for (auto &arr : pat_) {
    for (size_t i = 0; i < arr.size(); ++i) {
      for (size_t j = 0; j < arr.size(); ++j) {
        add_diag_block<double, 3>(arr[i].first, arr[j].first, w_*arr[i].second*arr[j].second, hes);
      }
    }
  }
  return 0;
}

void low_pass_filter_energy::Update(const double *ref) {
  ref_ = ref;
}
//==============================================================================
}
