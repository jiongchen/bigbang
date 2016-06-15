#ifndef CATENARY_H
#define CATENARY_H

#include <Eigen/Sparse>

#include "def.h"

namespace bigbang {

using pfunc_t=std::shared_ptr<Functional<double>>;

struct catenary {
  double len;
  double dens;
  size_t vert_num;
  Eigen::VectorXd pos, vel;
  Eigen::SparseMatrix<double> Mass;
};

struct catenary* create_catenary(const double length,
                                 const size_t vert_num,
                                 const double density);

int dump_catenary(const char *filename, const catenary *ins);

class catenary_strain : public Functional<double>
{
public:
  catenary_strain(const catenary *ins, const double w);
  size_t Nx() const;
  int Val(const double *x, double *val) const;
  int Gra(const double *x, double *gra) const;
  int Hes(const double *x, std::vector<Eigen::Triplet<double>> *hes) const;
private:
  const double w_;
  const size_t dim_;
  Eigen::VectorXd len_;
};

class catenary_bending : public Functional<double>
{
public:
  catenary_bending(const catenary *ins, const double w);
  size_t Nx() const;
  int Val(const double *x, double *val) const;
  int Gra(const double *x, double *gra) const;
  int Hes(const double *x, std::vector<Eigen::Triplet<double>> *hes) const;
private:
  const double w_;
  const size_t dim_;
  Eigen::VectorXd len_;
};

class catenary_grav : public Functional<double>
{
public:
  catenary_grav(const catenary *ins, const double w);
  size_t Nx() const;
  int Val(const double *x, double *val) const;
  int Gra(const double *x, double *gra) const;
  int Hes(const double *x, std::vector<Eigen::Triplet<double>> *hes) const;
private:
  const double w_;
  const size_t dim_;
  Eigen::VectorXd m_, g_;
};

class handle_move
{
public:
  handle_move(const Eigen::Vector3d &x0, const double &timer, const double start)
      : x0_(x0), timer_(timer), start_(start) {}
  virtual Eigen::Vector3d move() const = 0;
  bool valid() const {
    return timer_ >= start_;
  }
protected:
  const Eigen::Vector3d x0_;
  const double &timer_;
  const double start_;
};

class constant_move : public handle_move
{
public:
  constant_move(const Eigen::Vector3d &x0, const double &timer, const double start)
      : handle_move(x0, timer, start) {}
  Eigen::Vector3d move() const {
    return handle_move::x0_;
  }
};

class vertical_sine_move : public handle_move
{
public:
  vertical_sine_move(const Eigen::Vector3d &x0, const double &timer, const double start,
                     const double omega, const double ampitude)
      : omega_(omega), ampitude_(ampitude), handle_move(x0, timer, start) {}
  Eigen::Vector3d move() const {
    Eigen::Vector3d rtn = handle_move::x0_;
    rtn.y() += ampitude_*sin(omega_*(handle_move::timer_-handle_move::start_));
    return rtn;
  }
private:
  const double omega_, ampitude_;
};

class catenary_handle : public Functional<double>
{
public:
  catenary_handle(const catenary *ins, const double w);
  size_t Nx() const;
  int Val(const double *x, double *val) const;
  int Gra(const double *x, double *gra) const;
  int Hes(const double *x, std::vector<Eigen::Triplet<double>> *hes) const;
  void PinDown(const size_t idx, const std::shared_ptr<handle_move> &mv);
private:
  const double w_;
  const size_t dim_;
  std::vector<size_t> indices_;
  std::vector<std::shared_ptr<handle_move>> moves_;
};

}
#endif
