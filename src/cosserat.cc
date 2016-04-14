#include "cosserat.h"

#include <iostream>

#include "def.h"
#include "config.h"

using namespace std;
using namespace Eigen;

namespace bigbang {

extern "C" {

void rod_stretch_(double *val, const double *x, const double *d, const double *Es, const double *r);
void rod_stretch_jac_(double *jac, const double *x, const double *d, const double *Es, const double *r);
void rod_stretch_hes_(double *hes, const double *x, const double *d, const double *Es, const double *r);

void rod_bend_(double *val, const double *q, const double *u, const double *d, const double *E, const double *G, const double *r);
void rod_bend_jac_(double *jac, const double *q, const double *u, const double *d, const double *E, const double *G, const double *r);
void rod_bend_hes_(double *hes, const double *q, const double *u, const double *d, const double *E, const double *G, const double *r);

void rod_couple_(double *val, const double *xq, const double *d, const double *kappa);
void rod_couple_jac_(double *jac, const double *xq, const double *d, const double *kappa);
void rod_couple_hes_(double *hes, const double *xq, const double *d, const double *kappa);

}

void generate_rod(const Matrix<double, 3, 2> &ends, const size_t n, Matrix3Xd &rod) {
  rod.resize(NoChange, n);
  for (size_t i = 0; i < n; ++i) {
    rod.col(i) = ends.col(0)+(ends.col(1)-ends.col(0))/(n-1)*i;
  }
}

void init_rod_as_helix(const double radius, const double h, const double omega, const double dt, Matrix3Xd &rod) {
  double t = 0;
  for (size_t i = 0; i < rod.cols(); ++i) {
    rod(0, i) = radius*cos(omega*t);
    rod(1, i) = radius*sin(omega*t);
    rod(2, i) = h*t;
    t += dt;
  }
}

class line_kinetic_energy;
class angular_kinetic_energy;

class rod_stretch_energy : public Functional<double>
{
public:
  rod_stretch_energy(const Matrix3Xd &rod, const Matrix4Xd &q, const double Es, const double r)
    : r_size_(rod.size()), q_size_(q.size()), Es_(Es), r_(r), elem_num_(rod.cols()-1) {
    len_ = VectorXd::Zero(elem_num_);
    for (size_t i = 0; i < elem_num_; ++i) {
      len_(i) = (rod.col(i)-rod.col(i+1)).norm();
    }
  }
  size_t Nx() const {
    return r_size_+q_size_;
  }
  int Val(const double *x, double *val) const {
    Map<const VectorXd> X(x, r_size_+q_size_);
    for (size_t i = 0; i < elem_num_; ++i) {
      Matrix<double, 3, 2> rr;
      rr.col(0) = X.segment<3>(3*i);
      rr.col(1) = X.segment<3>(3*(i+1));
      double value = 0;
      rod_stretch_(&value, rr.data(), &len_(i), &Es_, &r_);
      *val += value;
    }
    return 0;
  }
  int Gra(const double *x, double *gra) const {
    Map<const VectorXd> X(x, r_size_+q_size_);
    Map<VectorXd> G(gra, r_size_+q_size_);
    for (size_t i = 0; i < elem_num_; ++i) {
      Matrix<double, 3, 2> rr;
      rr.col(0) = X.segment<3>(3*i);
      rr.col(1) = X.segment<3>(3*(i+1));
      Matrix<double, 3, 2> g = Matrix<double, 3, 2>::Zero();
      rod_stretch_jac_(g.data(), rr.data(), &len_(i), &Es_, &r_);
      G.segment<3>(3*i) += g.col(0);
      G.segment<3>(3*(i+1)) += g.col(1);
    }
    return 0;
  }
  int Hes(const double *x, vector<Triplet<double>> *hes) const {
    Map<const VectorXd> X(x, r_size_+q_size_);
    for (size_t i = 0; i < elem_num_; ++i) {
      Matrix<double, 3, 2> rr;
      rr.col(0) = X.segment<3>(3*i);
      rr.col(1) = X.segment<3>(3*(i+1));
      Matrix<double, 6, 6> H = Matrix<double, 6, 6>::Zero();
      rod_stretch_hes_(H.data(), rr.data(), &len_(i), &Es_, &r_);
      for (size_t p = 0; p < 6; ++p) {
        for (size_t q = 0; q < 6; ++q) {
          const size_t I = 3*(i+p/3)+p%3;
          const size_t J = 3*(i+q/3)+q%3;
          hes->push_back(Triplet<double>(I, J, H(p, q)));
        }
      }
    }
    return 0;
  }
private:
  const size_t r_size_, q_size_;
  const size_t elem_num_;
  const double Es_, r_;
  VectorXd len_;
};

class rod_bend_energy : public Functional<double>
{
public:
  rod_bend_energy(const Matrix3Xd &rod, const Matrix4Xd &q, const double E, const double G, const double r)
    : r_size_(rod.size()), q_size_(q.size()), elem_num_(q.cols()-2), E_(E), G_(G), r_(r) {
    len_ = VectorXd::Zero(elem_num_);
    for (size_t i = 0; i < elem_num_; ++i) {
      len_(i) = 0.5*((rod.col(i)-rod.col(i+1)).norm()+(rod.col(i+1)-rod.col(i+2)).norm());
    }
  }
  size_t Nx() const {
    return r_size_+q_size_;
  }
  int Val(const double *x, double *val) const {
    Map<const VectorXd> X(x, r_size_+q_size_);
    for (size_t i = 0; i < elem_num_; ++i) {
      Matrix<double, 4, 2> qq;
      qq.col(0) = X.segment<4>(r_size_+4*i);
      qq.col(1) = X.segment<4>(r_size_+4*(i+1));
      double value = 0, u = 1;
      rod_bend_(&value, qq.data(), &u, &len_(i), &E_, &G_, &r_);
      *val += value;
    }
    return 0;
  }
  int Gra(const double *x, double *gra) const {
    Map<const VectorXd> X(x, r_size_+q_size_);
    Map<VectorXd> G(gra, r_size_+q_size_);
    for (size_t i = 0; i < elem_num_; ++i) {
      Matrix<double, 4, 2> qq;
      qq.col(0) = X.segment<4>(r_size_+4*i);
      qq.col(1) = X.segment<4>(r_size_+4*(i+1));
      Matrix<double, 4, 2> g = Matrix<double, 4, 2>::Zero();
      double u = 1;
      rod_bend_jac_(g.data(), qq.data(), &u, &len_(i), &E_, &G_, &r_);
      G.segment<4>(r_size_+4*i) += g.col(0);
      G.segment<4>(r_size_+4*(i+1)) += g.col(1);
    }
    return 0;
  }
  int Hes(const double *x, vector<Triplet<double>> *hes) const {
    Map<const VectorXd> X(x, r_size_+q_size_);
    for (size_t i = 0; i < elem_num_; ++i) {
      Matrix<double, 4, 2> qq;
      qq.col(0) = X.segment<4>(r_size_+4*i);
      qq.col(1) = X.segment<4>(r_size_+4*(i+1));
      Matrix<double, 8, 8> H = Matrix<double, 8, 8>::Zero();
      double u = 1;
      rod_bend_hes_(H.data(), qq.data(), &u, &len_(i), &E_, &G_, &r_);
      for (size_t p = 0; p < 8; ++p) {
        for (size_t q = 0; q < 8; ++q) {
          const size_t I = r_size_+4*(i+p/4)+p%4;
          const size_t J = r_size_+4*(i+q/4)+q%4;
          hes->push_back(Triplet<double>(I, J, H(p, q)));
        }
      }
    }
    return 0;
  }
private:
  const size_t r_size_, q_size_;
  const size_t elem_num_;
  const double E_, G_, r_;
  VectorXd len_;
};

class coupling_energy : public Functional<double>
{
public:
  coupling_energy(const Matrix3Xd &rod, const Matrix4Xd &q, const double kappa)
    : r_size_(rod.size()), q_size_(q.size()), elem_num_(rod.cols()-1), kappa_(kappa) {
    len_ = VectorXd::Zero(elem_num_);
    for (size_t i = 0; i < elem_num_; ++i) {
      len_(i) = (rod.col(i)-rod.col(i+1)).norm();
    }
  }
  size_t Nx() const {
    return r_size_+q_size_;
  }
  int Val(const double *x, double *val) const {
    Map<const VectorXd> XQ(x, r_size_+q_size_);
    for (size_t i = 0; i < elem_num_; ++i) {
      VectorXd rq = VectorXd::Zero(10);
      rq.segment<3>(0) = XQ.segment<3>(3*i);
      rq.segment<3>(3) = XQ.segment<3>(3*(i+1));
      rq.segment<4>(6) = XQ.segment<4>(r_size_+4*i);
      double value = 0;
      rod_couple_(&value, rq.data(), &len_(i), &kappa_);
      *val += value;
    }
    return 0;
  }
  int Gra(const double *x, double *gra) const {
    Map<const VectorXd> XQ(x, r_size_+q_size_);
    Map<VectorXd> G(gra, r_size_+q_size_);
    for (size_t i = 0; i < elem_num_; ++i) {
      VectorXd rq = VectorXd::Zero(10);
      rq.segment<3>(0) = XQ.segment<3>(3*i);
      rq.segment<3>(3) = XQ.segment<3>(3*(i+1));
      rq.segment<4>(6) = XQ.segment<4>(r_size_+4*i);
      VectorXd g = VectorXd::Zero(10);
      rod_couple_jac_(g.data(), rq.data(), &len_(i), &kappa_);
      G.segment<3>(3*i) += g.segment<3>(0);
      G.segment<3>(3*(i+1)) += g.segment<3>(3);
      G.segment<4>(r_size_+4*i) += g.segment<4>(6);
    }
    return 0;
  }
  int Hes(const double *x, vector<Triplet<double>> *hes) const {
    Map<const VectorXd> XQ(x, r_size_+q_size_);
    for (size_t i = 0; i < elem_num_; ++i) {
      VectorXd rq = VectorXd::Zero(10);
      rq.segment<3>(0) = XQ.segment<3>(3*i);
      rq.segment<3>(3) = XQ.segment<3>(3*(i+1));
      rq.segment<4>(6) = XQ.segment<4>(r_size_+4*i);
      Matrix<double, 10, 10> H = Matrix<double, 10, 10>::Zero();
      rod_couple_hes_(H.data(), rq.data(), &len_(i), &kappa_);
      for (size_t p = 0; p < 10; ++p) {
        for (size_t q = 0; q < 10; ++q) {
          const size_t I = p < 6 ? 3*(i+p/3)+p%3 : r_size_+4*i+p-6;
          const size_t J = q < 6 ? 3*(i+q/3)+q%3 : r_size_+4*i+q-6;
          hes->push_back(Triplet<double>(I, J, H(p, q)));
        }
      }
    }
    return 0;
  }
private:
  const size_t r_size_, q_size_;
  const size_t elem_num_;
  const double kappa_;
  VectorXd len_;
};
//==============================================================================
cosserat_solver::cosserat_solver(const Matrix3Xd &rest)
  : rest_(rest) {}

void cosserat_solver::init_rod(const Matrix3Xd &rinit, const Matrix4Xd &qinit) {
  r_ = rinit;
  q_ = qinit;
  vr_.setZero(r_.rows(), r_.cols());
  vq_.setZero(q_.rows(), q_.cols());
}

void cosserat_solver::precompute() {
  buffer_.resize(3);
  buffer_[0] = make_shared<rod_stretch_energy>(rest_, q_, param_.Es, param_.radius);
//  buffer_[1] = make_shared<rod_bend_energy>(rest_, q_, param_.E, param_.G, param_.radius);
  buffer_[2] = make_shared<coupling_energy>(rest_, q_, param_.kappa);
  try {
    potential_ = make_shared<energy_t<double>>(buffer_);
  } catch ( exception &e ) {
    cerr << "[Error] exception " << e.what() << endl;
    exit(EXIT_FAILURE);
  }
}

void cosserat_solver::advance(const size_t max_iter, const double tolerance) {
  const size_t dim = r_.size()+q_.size();
  VectorXd xstar(dim);
  std::copy(r_.data(), r_.data()+r_.size(), xstar.data());
  std::copy(q_.data(), q_.data()+q_.size(), xstar.data()+r_.size());
  // iterative solve
  for (size_t iter = 0; iter < max_iter; ++iter) {
    double value = 0; {
      potential_->Val(xstar.data(), &value);
      cout << "\t@potential value: " << value << endl;
    }
    VectorXd g = VectorXd::Zero(dim); {
      potential_->Gra(xstar.data(), g.data());
      g *= -1;
    }
    SparseMatrix<double> H(dim, dim); {
      vector<Triplet<double>> trips;
      potential_->Hes(xstar.data(), &trips);
      H.setFromTriplets(trips.begin(), trips.end());
    }
    SimplicialCholesky<SparseMatrix<double>> solver;
    solver.compute(H);
    ASSERT(solver.info() == Success);
    VectorXd dx = solver.solve(g);
    ASSERT(solver.info() == Success);
    xstar += dx;
    for (size_t k = 0; k < q_.size()/4; ++k)
      xstar.segment<4>(r_.size()+4*k).normalize();
  }
  std::copy(xstar.data(), xstar.data()+r_.size(), r_.data());
  std::copy(xstar.data()+r_.size(), xstar.data()+dim, q_.data());
//  double len = 0;
//  for (size_t i = 0; i < r_.cols()-1; ++i)
//    len += (r_.col(i)-r_.col(i+1)).norm();
//  cout << "length: " << len << endl;
}

}
