#ifndef COSSERAT_H
#define COSSERAT_H

#include <Eigen/Sparse>

namespace bigbang {

/// @brief generate a rod with N vertices
void generate_rod(const Eigen::Matrix<double, 3, 2> &ends, const double len, const size_t N, Eigen::Matrix3Xd &rod);

/// @brief x=R*cos(omega*t), y = R*sin(omega*t), z = h*t
void init_rod_as_helix(const double radius, const double omega, const double h, const double dt, Eigen::Matrix3Xd &rod);

/// @brief designed as a physics solver
/// rather than a variational solver
class cosserat_solver
{
public:
  cosserat_solver(const Eigen::Matrix3Xd &rest, const double h);
  void init_rod(const Eigen::Matrix3Xd &init);
  void advance();
  Eigen::Matrix3Xd& get_rod_pos() { return r_; }
private:
  // physics part
  Eigen::Matrix3Xd r_, vr_;
  Eigen::Matrix4Xd q_, vq_;
  Eigen::VectorXd Mr_, Mq_;
  double h_;
  // numeric part
};

}

#endif
