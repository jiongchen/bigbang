#ifndef ENERGY_H
#define ENERGY_H

#include <zjucad/matrix/matrix.h>
#include <Eigen/Sparse>

#include "def.h"

using mati_t=zjucad::matrix::matrix<size_t>;
using matd_t=zjucad::matrix::matrix<double>;

namespace bigbang {

class momentum_potential_imp_euler : public Functional<double>
{
public:
  momentum_potential_imp_euler(const mati_t &cell, const matd_t &nods, const double rho, const double h, const double w=1.0);
  size_t Nx() const;
  int Val(const double *x, double *val) const;
  int Gra(const double *x, double *gra) const;
  int Hes(const double *x, std::vector<Eigen::Triplet<double>> *hes) const;
  void Init(const double *x0, const double *v0);
  void Update(const double *x);
  double QueryKineticEnergy() const;
private:
  const double rho_, h_;
  const size_t dim_;
  double w_;
  Eigen::SparseMatrix<double> M_;
  Eigen::VectorXd xn_, vn_;
};

class momentum_potential_bdf2 : public Functional<double>
{
public:
  momentum_potential_bdf2(const mati_t &cell, const matd_t &nods, const double rho, const double h, const double w=1.0);
  size_t Nx() const;
  int Val(const double *x, double *val) const;
  int Gra(const double *x, double *gra) const;
  int Hes(const double *x, std::vector<Eigen::Triplet<double>> *hes) const;
  void Init(const double *x0, const double *v0);
  void Update(const double *x);
  double QueryKineticEnergy();
private:
  const double rho_, h_;
  const size_t dim_;
  double w_;
  Eigen::SparseMatrix<double> M_;
  Eigen::VectorXd xn_, vn_, xnn_, vnn_;
};

class elastic_potential : public Functional<double>
{
public:
  enum Material {
    LINEAR,
    STVK,
    COROTATIONAL,
    NEOHOOKEAN
  };
  elastic_potential(const mati_t &tets, const matd_t &nods, Material type, const double Ym, const double Pr, const double w);
  size_t Nx() const;
  int Val(const double *x, double *val) const;
  int Gra(const double *x, double *gra) const;
  int Hes(const double *x, std::vector<Eigen::Triplet<double>> *hes) const;
private:
  const size_t dim_;
  double w_;
  Material type_;
  const mati_t &tets_;
  matd_t vol_;
  double lam_, miu_;
  matd_t Dm_;
};

class gravitational_potential : public Functional<double>
{
public:
  gravitational_potential(const mati_t &cell, const matd_t &nods, const double rho, const double w);
  size_t Nx() const;
  int Val(const double *x, double *val) const;
  int Gra(const double *x, double *gra) const;
  int Hes(const double *x, std::vector<Eigen::Triplet<double>> *hes) const { return __LINE__; }
private:
  const size_t dim_;
  double w_;
  Eigen::SparseMatrix<double> M_;
};

class spring_potential;

class bending_potential;

class quadratic_bending_potential;

class position_constraint;

}

#endif
