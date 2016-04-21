#include "cosserat.h"

#include <iostream>
#include <unordered_map>

#include "def.h"
#include "config.h"
#include "util.h"
#include "optimizer.h"

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

//void init_rod_as_helix(const double radius, const double h, const double omega, const double dt, Matrix3Xd &rod) {
//  double t = 0;
//  for (size_t i = 0; i < rod.cols(); ++i) {
//    rod(0, i) = radius*cos(omega*t);
//    rod(1, i) = radius*sin(omega*t);
//    rod(2, i) = h*t;
//    t += dt;
//  }
//}

//void init_rod_as_spiral(const double radius, const double omega, const double dt, Matrix3Xd &rod) {
//  double t = 0;
//  for (size_t i = 0; i < rod.cols(); ++i) {
//    rod(0, i) = radius*omega*t*cos(omega*t);
//    rod(1, i) = radius*omega*t*sin(omega*t);
//    rod(2, i) = 0;
//    t += dt;
//  }
//}

//void compute_bishop_frame(const Matrix3Xd &rod, const Matrix3d &u0, Matrix4Xd &frm) {
//  vector<Matrix3d> U(rod.cols()-1);
//  U.front() = u0;
//  for (size_t i = 1; i < U.size(); ++i) {
//    U[i].col(2) = (rod.col(i+1)-rod.col(i)).normalized();
//  }
//  for (size_t i = 0; i < U.size()-1; ++i) {
//    Vector3d v1 = rod.col(i+1)-rod.col(i);
//    double c1 = v1.dot(v1);
//    Vector3d rL = U[i].col(0)-(2/c1)*v1.dot(U[i].col(0))*v1;
//    Vector3d tL = U[i].col(2)-(2/c1)*v1.dot(U[i].col(2))*v1;
//    Vector3d v2 = U[i+1].col(2)-tL;
//    double c2 = v2.dot(v2);
//    U[i+1].col(0) = rL-(2/c2)*v2.dot(rL)*v2;
//    U[i+1].col(1) = U[i+1].col(2).cross(U[i+1].col(0));
//  }
//  frm.resize(NoChange, U.size());
//  for (size_t i = 0; i < frm.cols(); ++i) {
//    Quaterniond q(U[i]);
//    frm.col(i) = q.coeffs();
//  }
//}

class mass_calculator
{
public:
  mass_calculator(const Matrix3Xd &rest, const double rho, const double r)
    : r_size_(rest.size()), q_size_(4*(rest.cols()-1)), rho_(rho), r_(r), I_(3), B_(3) {
    lenr_ = VectorXd::Zero(rest.cols()-1);
    for (size_t i = 0; i < rest.cols()-1; ++i)
      lenr_(i) = (rest.col(i)-rest.col(i+1)).norm();
    lenq_ = VectorXd::Zero(rest.cols()-2);
    for (size_t i = 0; i < rest.cols()-2; ++i)
      lenq_(i) = 0.5*((rest.col(i)-rest.col(i+1)).norm()+(rest.col(i+1)-rest.col(i+2)).norm());
    I_[0] = I_[1] = rho*M_PI*r*r/4.0;
    I_[2] = 2*I_[0];
    B_[0] << 0, 0, 0, 1,
        0, 0, 1, 0,
        0, -1, 0, 0,
        -1, 0, 0, 0;
    B_[1] << 0, 0, -1, 0,
        0, 0, 0, 1,
        1, 0, 0, 0,
        0, -1, 0, 0;
    B_[2] << 0, 1, 0, 0,
        -1, 0, 0, 0,
        0, 0, 0, 1,
        0, 0, -1, 0;
  }
  void get_mass_matrix(const VectorXd &x, SparseMatrix<double> &M) {
    vector<Triplet<double>> trips;
    for (size_t i = 0; i < lenr_.size(); ++i) {
      double w = 0.5*rho_*M_PI*r_*r_*lenr_(i);
      add_diag_block<double, 3>(i, i, w, &trips);
      add_diag_block<double, 3>(i+1, i+1, w, &trips);
    }
    for (size_t i = 0; i < lenq_.size(); ++i) {
      for (size_t j = 0; j < 3; ++j) {
        Vector4d bj = B_[j]*(x.segment<4>(r_size_+4*i)+x.segment<4>(r_size_+4*(i+1)));
        Matrix4d dM = 0.25*lenq_(i)*I_[j]*bj*bj.transpose();
        insert_block<double>(r_size_+4*i, r_size_+4*i, dM.data(), 4, 4, &trips);
        insert_block<double>(r_size_+4*i, r_size_+4*i+4, dM.data(), 4, 4, &trips);
        insert_block<double>(r_size_+4*i+4, r_size_+4*i, dM.data(), 4, 4, &trips);
        insert_block<double>(r_size_+4*i+4, r_size_+4*i+4, dM.data(), 4, 4, &trips);
      }
    }
    M.resize(r_size_+q_size_, r_size_+q_size_);
    M.setFromTriplets(trips.begin(), trips.end());
  }
private:
  const size_t r_size_, q_size_;
  const double rho_, r_;
  VectorXd lenr_, lenq_;
  vector<double> I_;
  vector<Matrix4d> B_;
};

class rod_stretch_energy : public Functional<double>
{
public:
  rod_stretch_energy(const Matrix3Xd &rod, const double Es, const double r)
    : r_size_(rod.size()), q_size_(4*(rod.cols()-1)), elem_num_(rod.cols()-1), Es_(Es), r_(r) {
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
      if ( (rr.col(0)-rr.col(1)).norm() >= len_(i) ) {
        rod_stretch_hes_(H.data(), rr.data(), &len_(i), &Es_, &r_);
      } else {
//        rod_stretch_hes_(H.data(), rr.data(), &len_(i), &Es_, &r_);
        double value = 0;
        rod_stretch_(&value, rr.data(), &len_(i), &Es_, &r_);
        Matrix<double, 6, 1> g = Matrix<double, 6, 1>::Zero();
        rod_stretch_jac_(g.data(), rr.data(), &len_(i), &Es_, &r_);
        g = 0.5*g/sqrt(value);
        H = 2*g*g.transpose();
      }
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
  rod_bend_energy(const Matrix3Xd &rod, const double E, const double G, const double r, const Vector3d &uk)
    : r_size_(rod.size()), q_size_(4*(rod.cols()-1)), elem_num_(rod.cols()-2), E_(E), G_(G), r_(r), uk_(uk) {
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
      double value = 0;
      rod_bend_(&value, qq.data(), uk_.data(), &len_(i), &E_, &G_, &r_);
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
      rod_bend_jac_(g.data(), qq.data(), uk_.data(), &len_(i), &E_, &G_, &r_);
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
      rod_bend_hes_(H.data(), qq.data(), uk_.data(), &len_(i), &E_, &G_, &r_);
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
  const Vector3d uk_;
};

class coupling_energy : public Functional<double>
{
public:
  coupling_energy(const Matrix3Xd &rod, const double kappa)
    : r_size_(rod.size()), q_size_(4*(rod.cols()-1)), elem_num_(rod.cols()-1), kappa_(kappa) {
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

class rod_fix_vert_energy : public Functional<double>
{
public:
  rod_fix_vert_energy(const Matrix3Xd &rod, const double w=1e2)
    : r_size_(rod.size()), q_size_(4*(rod.cols()-1)), w_(w) {}
  size_t Nx() const {
    return r_size_+q_size_;
  }
  int Val(const double *x, double *val) const {
    Map<const VectorXd> X(x, r_size_+q_size_);
    for (auto &elem : fixed_) {
      *val += 0.5*w_*(X.segment<3>(3*elem.first)-elem.second).squaredNorm();
    }
    return 0;
  }
  int Gra(const double *x, double *gra) const {
    Map<const VectorXd> X(x, r_size_+q_size_);
    Map<VectorXd> G(gra, r_size_+q_size_);
    for (auto &elem : fixed_) {
      G.segment<3>(3*elem.first) += w_*(X.segment<3>(3*elem.first)-elem.second);
    }
    return 0;
  }
  int Hes(const double *x, vector<Triplet<double>> *hes) const {
    for (auto &elem : fixed_) {
      const size_t id = elem.first;
      hes->push_back(Triplet<double>(3*id+0, 3*id+0, w_));
      hes->push_back(Triplet<double>(3*id+1, 3*id+1, w_));
      hes->push_back(Triplet<double>(3*id+2, 3*id+2, w_));
    }
    return 0;
  }
  int Pin(const size_t id, const double *pos) {
    if ( id < 0 || id >= r_size_/3 )
      return __LINE__;
    fixed_[id] = Vector3d(pos);
    return 0;
  }
  int Release(const size_t id) {
    if ( id < 0 || id >= r_size_/3 ) {
      cerr << "[Info] vertex index is out of range\n";
      return __LINE__;
    }
    auto it = fixed_.find(id);
    if ( it == fixed_.end() ) {
      cerr << "[Info] vertex " << id << " is not fixed\n";
      return __LINE__;
    }
    fixed_.erase(it);
    return 0;
  }
private:
  const size_t r_size_, q_size_;
  double w_;
  std::unordered_map<size_t, Eigen::Vector3d> fixed_;
};
//==============================================================================
cosserat_solver::cosserat_solver(const Matrix3Xd &rest, const rod_material &param)
  : rest_(rest), param_(param) {
  buffer_.resize(4);
  buffer_[0] = make_shared<rod_stretch_energy>(rest_, param_.Es, param_.radius);
  buffer_[1] = make_shared<rod_bend_energy>(rest_, param_.E, param_.G, param_.radius, Vector3d(param_.u0, param_.u1, param_.u2));
  buffer_[2] = make_shared<coupling_energy>(rest_, param_.kappa);
  buffer_[3] = make_shared<rod_fix_vert_energy>(rest_);
  try {
    potential_ = make_shared<energy_t<double>>(buffer_);
  } catch ( exception &e ) {
    cerr << "[Error] exception " << e.what() << endl;
    exit(EXIT_FAILURE);
  }
  mc_ = make_shared<mass_calculator>(rest_, param_.density, param_.radius);
}

void cosserat_solver::init_rod(const Matrix3Xd &rinit, const Matrix4Xd &qinit) {
  r_ = rinit;
  q_ = qinit;
  vr_.setZero(r_.rows(), r_.cols());
  vq_.setZero(q_.rows(), q_.cols());
}

void cosserat_solver::pin_down_vert(const size_t id, const double *pos) {
  std::dynamic_pointer_cast<rod_fix_vert_energy>(buffer_[3])->Pin(id, pos);
}

void cosserat_solver::precompute() {
}

void cosserat_solver::advance(const size_t max_iter, const double tolerance) {
  const size_t dim = potential_->Nx();
  VectorXd xn(dim), vn(dim);
  std::copy(r_.data(), r_.data()+r_.size(), xn.data());
  std::copy(q_.data(), q_.data()+q_.size(), xn.data()+r_.size());
  std::copy(vr_.data(), vr_.data()+vr_.size(), vn.data());
  std::copy(vq_.data(), vq_.data()+vq_.size(), vn.data()+vr_.size());
  VectorXd xstar = xn;
  // iterative solve
  for (size_t iter = 0; iter < max_iter; ++iter) {
    double value = 0; {
      if ( iter % 100 == 0 ) {
        potential_->Val(xstar.data(), &value);
        cout << "\t@potential value: " << value << endl;
      }
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
    SparseMatrix<double> M; {
      mc_->get_mass_matrix(xstar, M);
    }
    SparseMatrix<double> LHS = M+param_.h*param_.h*H;
    VectorXd rhs = param_.h*param_.h*g-M*(xstar-xn-param_.h*vn);
    SimplicialCholesky<SparseMatrix<double>> solver;
    //solver.setMode(SimplicialCholeskyLLT);
    solver.compute(LHS);
    ASSERT(solver.info() == Success);
    VectorXd dx = solver.solve(rhs);
    ASSERT(solver.info() == Success);
    double xstar_norm = xstar.norm();
    xstar += dx;
    for (size_t k = 0; k < q_.cols(); ++k)
      xstar.segment<4>(r_.size()+4*k).normalize();
    if ( dx.norm() < tolerance*xstar_norm ) {
      cout << "\t@CONVERGED\n";
      break;
    }
  }
  // update velocity
  vn = (xstar-xn)/param_.h;
  xn = xstar;
  std::copy(xn.data(), xn.data()+r_.size(), r_.data());
  std::copy(xn.data()+r_.size(), xn.data()+dim, q_.data());
  std::copy(vn.data(), vn.data()+r_.size(), vr_.data());
  std::copy(vn.data()+vr_.size(), vn.data()+dim, vq_.data());
}

}
