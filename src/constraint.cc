#include "constraint.h"

#include <iostream>
#include <zjucad/matrix/itr_matrix.h>

using namespace std;
using namespace Eigen;
using namespace zjucad::matrix;

namespace bigbang {

//==============================================================================
position_constraint::position_constraint(const mati_t &point, const size_t dim, const double *pos)
  : dim_(dim), constraint_piece(point, 1.0, constraint_piece::EQUAL) {
  pos_.resize(3, 1);
  std::copy(pos, pos+3, pos_.begin());
}

int position_constraint::eval_val(const double *x, double *val) const {
  itr_matrix<const double *> X(3, dim_/3, x);
  matd_t dx = X(colon(), pn_)-pos_;
  *val = dot(dx, dx);
  return 0;
}

int position_constraint::eval_jac(const double *x, double *jac) const {
  itr_matrix<const double *> X(3, dim_/3, x);
  matd_t dx = 2*(X(colon(), pn_)-pos_);
  std::copy(dx.begin(), dx.end(), jac);
  return 0;
}
//==============================================================================
inext_constraint::inext_constraint(const mati_t &edge, const matd_t &nods)
  : dim_(nods.size()), constraint_piece(edge, 1.0, constraint_piece::EQUAL) {
  matd_t vert = nods(colon(), pn_);
  len_ = norm(vert(colon(), 0)-vert(colon(), 1));
}

int inext_constraint::eval_val(const double *x, double *val) const {
  matd_t X = itr_matrix<const double *>(3, dim_/3, x)(colon(), pn_);
  matd_t ed = X(colon(), 0)-X(colon(), 1);
  *val = dot(ed, ed)/len_-len_;
  return 0;
}

int inext_constraint::eval_jac(const double *x, double *jac) const {
  matd_t X = itr_matrix<const double *>(3, dim_/3, x)(colon(), pn_);
  matd_t grad = 2*(X(colon(), 0)-X(colon(), 1))/len_;
  std::copy(grad.begin(), grad.end(), jac);
  grad *= -1.0;
  std::copy(grad.begin(), grad.end(), jac+3);
  return 0;
}
//===============================================================================
asm_constraint::asm_constraint(const vector<shared_ptr<constraint_piece<double>>> &buffer)
  : buffer_(buffer) {
  if ( buffer_.empty() ) {
    cerr << "[error] no input constraints\n";
    exit(EXIT_FAILURE);
  }
  dim_ = buffer_[0]->dim();
  for (auto &elem : buffer_) {
    if ( dim_ != elem->dim() ) {
      cerr << "[error] incompatible constraints\n";
      exit(EXIT_FAILURE);
    }
  }
}

int asm_constraint::Val(const double *x, double *val) const {
#pragma omp parallel for
  for (size_t i = 0; i < buffer_.size(); ++i) {
    buffer_[i]->eval_val(x, val+i);
  }
  return 0;
}

int asm_constraint::Jac(const double *x, const size_t off, vector<Triplet<double>> *jac) const {
  for (size_t i = 0; i < buffer_.size(); ++i) {
    matd_t grad(3*buffer_[i]->pn_.size(), 1);
    buffer_[i]->eval_jac(x, &grad[0]);
    for (size_t j = 0; j < grad.size(); ++j) {
      if ( grad[j]*grad[j] != 0.0 )
        jac->push_back(Triplet<double>(off+i, 3*buffer_[i]->pn_[j/3]+j%3, grad[j]));
    }
  }
  return 0;
}
//==============================================================================
}
