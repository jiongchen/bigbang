#include "catenary.h"

#include <iostream>
#include <fstream>
#include <zjucad/matrix/matrix.h>
#include <zjucad/matrix/itr_matrix.h>

#include "util.h"
#include "vtk.h"
#include "config.h"

using namespace std;
using namespace zjucad::matrix;
using namespace Eigen;
using mati_t=zjucad::matrix::matrix<size_t>;
using matd_t=zjucad::matrix::matrix<double>;

namespace bigbang {

extern "C" {
  
void mass_spring_(double *val, const double *x, const double *d);
void mass_spring_jac_(double *jac, const double *x, const double *d);
void mass_spring_hes_(double *hes, const double *x, const double *d);

void curve_bending_(double *val, const double *x, const double *d1, const double *d2);
void curve_bending_jac_(double *jac, const double *x, const double *d1, const double *d2);
void curve_bending_hes_(double *hes, const double *x, const double *d1, const double *d2);

}

struct catenary* create_catenary(const double length,
                                 const size_t vert_num,
                                 const double dens) {
  if ( length < 0.0 || vert_num < 3 || dens < 0.0 )
    return nullptr;
  
  struct catenary *ins = new catenary;
  ins->len = length;
  ins->dens = dens;
  ins->vert_num = vert_num;

  // init positions
  ins->pos = VectorXd::Zero(3*vert_num);
  Vector3d dir = Vector3d(length, 0, 0)/(vert_num-1);
  for (size_t i = 0; i < vert_num; ++i)
    ins->pos.segment<3>(3*i) = i*dir;

  // init velocity
  ins->vel = VectorXd::Zero(3*vert_num);

  // mass matrix
  vector<Triplet<double>> trips;
  for (size_t i = 0; i < vert_num-1; ++i) {
    double mass = ins->dens*(ins->pos.segment<3>(3*i)-ins->pos.segment<3>(3*i+3)).norm();
    add_diag_block<double, 3>(i, i, mass/3.0, &trips);
    add_diag_block<double, 3>(i, i+1, mass/6.0, &trips);
    add_diag_block<double, 3>(i+1, i, mass/6.0, &trips);
    add_diag_block<double, 3>(i+1, i+1, mass/3.0, &trips);
  }
  ins->Mass.resize(3*vert_num, 3*vert_num);
  ins->Mass.reserve(trips.size());
  ins->Mass.setFromTriplets(trips.begin(), trips.end());
  
  return ins;
}

int dump_catenary(const char *filename, const catenary *ins) {
  ofstream ofs(filename);
  if ( ofs.fail() ) {
    cerr << "[Info] can not write " << filename << endl;
    return __LINE__;
  }
  const size_t vert_num = ins->vert_num;
  matrix<size_t> cell(2, vert_num-1);
  cell(0, colon()) = colon(0, vert_num-2);
  cell(1, colon()) = colon(1, vert_num-1);

  line2vtk(ofs, ins->pos.data(), ins->pos.size()/3, &cell[0], cell.size(2));
  ofs.close();
  return 0;
}

//===============================================================================
catenary_strain::catenary_strain(const catenary *ins, const double w)
    : w_(w), dim_(ins->pos.size()) {
  len_ = VectorXd::Zero(ins->vert_num-1);
  for (size_t i = 0; i < len_.size(); ++i)
    len_[i] = (ins->pos.segment<3>(3*i)-ins->pos.segment<3>(3*i+3)).norm();
}

size_t catenary_strain::Nx() const {
  return dim_;
}

int catenary_strain::Val(const double *x, double *val) const {
  RETURN_WITH_COND_TRUE(w_ == 0.0);
  itr_matrix<const double *> X(3, dim_/3, x);
  double value = 0;
  matd_t vert = zeros<double>(3, 2);
  for (size_t i = 0; i < len_.size(); ++i) {
    vert = X(colon(), colon(i, i+1));
    mass_spring_(&value, &vert[0], &len_[i]);
    *val += w_*value;
  }
  return 0;
}

int catenary_strain::Gra(const double *x, double *gra) const {
  RETURN_WITH_COND_TRUE(w_ == 0.0);
  itr_matrix<const double *> X(3, dim_/3, x);
  itr_matrix<double *> G(3, dim_/3, gra);
  matd_t vert = zeros<double>(3, 2);
  matd_t g = zeros<double>(3, 2);
  for (size_t i = 0; i < len_.size(); ++i) {
    vert = X(colon(), colon(i, i+1));
    mass_spring_jac_(&g[0], &vert[0], &len_[i]);
    G(colon(), colon(i, i+1)) += w_*g;
  }
  return 0;
}

int catenary_strain::Hes(const double *x, vector<Triplet<double>> *hes) const {
  RETURN_WITH_COND_TRUE(w_ == 0.0);
  itr_matrix<const double *> X(3, dim_/3, x);
  matd_t vert = zeros<double>(3, 2);
  matd_t H = zeros<double>(6, 6);
  matd_t g = zeros<double>(6, 1);
  for (size_t i = 0; i < len_.size(); ++i) {
    vert = X(colon(), colon(i, i+1));
    double curr_len = norm(vert(colon(), 0)-vert(colon(), 1));
    if ( curr_len < len_[i] ) {
      double value = 0;
      mass_spring_(&value, &vert[0], &len_[i]);
      mass_spring_jac_(&g[0], &vert[0], &len_[i]);
      g /= 2*sqrt(value);
      H = 2*g*trans(g);
    } else {
      mass_spring_hes_(&H[0], &vert[0], &len_[i]);
    }
    for (size_t p = 0; p < 6; ++p) {
      for (size_t q = 0; q < 6; ++q) {
        const size_t I = 3*(i+p/3)+p%3;
        const size_t J = 3*(i+q/3)+q%3;
        hes->push_back(Triplet<double>(I, J, w_*H(p, q)));
      }
    }
  }
  return 0;
}
//===============================================================================
catenary_bending::catenary_bending(const catenary *ins, const double w)
    : w_(w), dim_(ins->pos.size()) {
  len_ = VectorXd::Zero(ins->vert_num-1);
  for (size_t i = 0; i < len_.size(); ++i)
    len_[i] = (ins->pos.segment<3>(3*i)-ins->pos.segment<3>(3*i+3)).norm();
}

size_t catenary_bending::Nx() const {
  return dim_;
}

int catenary_bending::Val(const double *x, double *val) const {
  RETURN_WITH_COND_TRUE(w_ == 0.0);
  itr_matrix<const double *> X(3, dim_/3, x);
  double value = 0;
  matd_t vert = zeros<double>(3, 3);
  for (size_t i = 0; i < len_.size()-1; ++i) {
    vert = X(colon(), colon(i, i+2));
    curve_bending_(&value, &vert[0], &len_[i], &len_[i+1]);
    *val += w_*value;
  }
  return 0;
}

int catenary_bending::Gra(const double *x, double *gra) const {
  RETURN_WITH_COND_TRUE(w_ == 0.0);
  itr_matrix<const double *> X(3, dim_/3, x);
  itr_matrix<double *> G(3, dim_/3, gra);
  matd_t vert = zeros<double>(3, 3);
  matd_t g = zeros<double>(3, 3);
  for (size_t i = 0; i < len_.size()-1; ++i) {
    vert = X(colon(), colon(i, i+2));
    curve_bending_jac_(&g[0], &vert[0], &len_[i], &len_[i+1]);
    G(colon(), colon(i, i+2)) += w_*g;
  }
  return 0;
}

int catenary_bending::Hes(const double *x, vector<Triplet<double>> *hes) const {
  RETURN_WITH_COND_TRUE(w_ == 0.0);
  itr_matrix<const double *> X(3, dim_/3, x);
  matd_t vert = zeros<double>(3, 3);
  matd_t H = zeros<double>(9, 9);
  for (size_t i = 0; i < len_.size()-1; ++i) {
    vert = X(colon(), colon(i, i+2));
    curve_bending_hes_(&H[0], &vert[0], &len_[i], &len_[i+1]);
    for (size_t p = 0; p < 9; ++p) {
      for (size_t q = 0; q < 9; ++q) {
        const size_t I = 3*(i+p/3)+p%3;
        const size_t J = 3*(i+q/3)+q%3;
        hes->push_back(Triplet<double>(I, J, w_*H(p, q)));
      }
    }
  }
  return 0;
}
//===============================================================================
catenary_grav::catenary_grav(const catenary *ins, const double w)
    : w_(w), dim_(ins->pos.size()) {
  m_ = VectorXd::Zero(dim_);
  for (size_t i = 0; i < m_.size(); ++i)
    m_(i) = ins->Mass.row(i).sum();
  g_ = VectorXd::Zero(dim_);
  for (size_t i = 0; i < dim_/3; ++i)
    g_(3*i+1) = 9.8;
}

size_t catenary_grav::Nx() const {
  return dim_;
}

int catenary_grav::Val(const double *x, double *val) const {
  RETURN_WITH_COND_TRUE(w_ == 0.0);
  Map<const VectorXd> X(x, dim_);
  *val += w_*X.dot(m_.asDiagonal()*g_);
  return 0;
}

int catenary_grav::Gra(const double *x, double *gra) const {
  RETURN_WITH_COND_TRUE(w_ == 0.0);
  Map<VectorXd> G(gra, dim_);
  G += w_*m_.asDiagonal()*g_;
  return 0;
}

int catenary_grav::Hes(const double *x, vector<Triplet<double>> *hes) const {
  return __LINE__;
}
//===============================================================================
catenary_handle::catenary_handle(const catenary *ins, const double w)
    : w_(w), dim_(ins->pos.size()) {}

size_t catenary_handle::Nx() const {
  return dim_;
}

int catenary_handle::Val(const double *x, double *val) const {
  RETURN_WITH_COND_TRUE(w_ == 0.0);
  Map<const MatrixXd> X(x, 3, dim_/3);
  for (size_t i = 0; i < indices_.size(); ++i) {
    if ( moves_[i]->valid() )
      *val += 0.5*w_*(X.col(indices_[i])-moves_[i]->move()).squaredNorm();
  }
  return 0;
}

int catenary_handle::Gra(const double *x, double *gra) const {
  RETURN_WITH_COND_TRUE(w_ == 0.0);
  Map<const MatrixXd> X(x, 3, dim_/3);
  Map<MatrixXd> G(gra, 3, dim_/3);
  for (size_t i = 0; i < indices_.size(); ++i) {
    if ( moves_[i]->valid() )
      G.col(indices_[i]) += w_*(X.col(indices_[i])-moves_[i]->move());
  }
  return 0;
}

int catenary_handle::Hes(const double *x, vector<Triplet<double>> *hes) const {
  RETURN_WITH_COND_TRUE(w_ == 0.0);
  for (size_t i = 0; i < indices_.size(); ++i) {
    if ( moves_[i]->valid() )
      add_diag_block<double, 3>(indices_[i], indices_[i], w_, hes);
  }
  return 0;
}

void catenary_handle::PinDown(const size_t idx, const shared_ptr<handle_move> &mv) {
  indices_.push_back(idx);
  moves_.push_back(mv);
}
//===============================================================================

}
