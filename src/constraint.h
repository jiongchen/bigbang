#ifndef CONSTRAINT_H
#define CONSTRAINT_H

#include "def.h"

namespace bigbang {

using mati_t=zjucad::matrix::matrix<size_t>;
using matd_t=zjucad::matrix::matrix<double>;

//// $(x-p)^2=0$, nonlinear constraint
class position_constraint : public constraint_piece<double>
{
public:
  position_constraint(const mati_t &point, const size_t dim, const double *pos);
  size_t dim() const { return dim_; }
  int eval_val(const double *x, double *val) const;
  int eval_jac(const double *x, double *jac) const;
private:
  const size_t dim_;
  matd_t pos_;
};

/// $l_{curr}-l_{rest}=0$, nonlinear constraint
class inext_constraint : public constraint_piece<double>
{
public:
  inext_constraint(const mati_t &edge, const matd_t &nods);
  size_t dim() const { return dim_; }
  int eval_val(const double *x, double *val) const;
  int eval_jac(const double *x, double *jac) const;
private:
  const size_t dim_;
  double len_;
};

class asm_constraint : public Constraint<double>
{
public:
  asm_constraint(const std::vector<std::shared_ptr<constraint_piece<double>>> &buffer);
  size_t Nx() const { return dim_; }
  size_t Nf() const { return buffer_.size(); }
  int Val(const double *x, double *val) const;
  int Jac(const double *x, const size_t off, std::vector<Eigen::Triplet<double>> *jac) const;
private:
  size_t dim_;
  const std::vector<std::shared_ptr<constraint_piece<double>>> &buffer_;
};

}
#endif

