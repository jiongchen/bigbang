#include "optimizer.h"

#include <iostream>
#include <Eigen/UmfPackSupport>

#include "config.h"
#include "def.h"
#include "HLBFGS/HLBFGS.h"
#include "HLBFGS/Lite_Sparse_Matrix.h"

using namespace std;
using namespace Eigen;

namespace bigbang {

static const size_t MAX_ITER = 10000;
static const double EPS = 1e-12;

int newton_solve(double *x, const size_t dim, shared_ptr<Functional<double>> &f) {
  if ( dim != f->Nx() ) {
    cerr << "[error] dim not match\n";
    return __LINE__;
  }
  SimplicialCholesky<SparseMatrix<double>> sol;
  Map<VectorXd> X(x, dim);
  VectorXd Xstar, Xprev;
  Xstar = Xprev = X;
  for (size_t iter = 0; iter < MAX_ITER; ++iter) {
    double value = 0;
    f->Val(&Xstar[0], &value); {
      if ( iter % 100 == 0 ) {
        cout << "\t@iter " << iter << endl;
        cout << "\t@energy value: " << value << endl;
      }
    }
    VectorXd grad(dim); {
      grad.setZero();
      f->Gra(&Xstar[0], &grad[0]);
      if ( grad.norm() <= EPS ) {
        cout << "\t@gradient converged\n";
        break;
      }
    }
    SparseMatrix<double> H(dim, dim); {
      vector<Triplet<double>> trips;
      f->Hes(&Xstar[0], &trips);
      H.reserve(trips.size());
      H.setFromTriplets(trips.begin(), trips.end());
    }
    sol.compute(H);
    ASSERT(sol.info() == Success);
    VectorXd dx = sol.solve(-grad);
    ASSERT(sol.info() == Success);
    if ( dx.norm() <= EPS*Xprev.norm() ) {
      cout << "[info] converged after " << iter << " iterations\n\n";
      break;
    }
    Xprev = Xstar;
    Xstar += dx;
  }
  X = Xstar;
  return 0;
}

typedef void (*lbfgs_func_t)(int N, double* x, double *prev_x, double* f, double* g);

void newiteration(int iter, int call_iter, double *x, double* f, double *g,  double* gnorm) {
  cout << iter <<": " << call_iter <<" " << *f <<" " << *gnorm  << endl;
}

int lbfgs_solve(double *x, const size_t dim, shared_ptr<Functional<double>> &f, const int M, const int T, bool with_hessian) {
  if ( dim != f->Nx() ) {
    cerr << "[error] dim not match\n";
    return __LINE__;
  }

  double parameter[20];
  int info[20];
  //initialize
  INIT_HLBFGS(parameter, info);
  info[4] = MAX_ITER;
  info[6] = T;
  info[7] = with_hessian?1:0;
  info[10] = 0;
  info[11] = 1;

  lbfgs_func_t evalfunc;
  if ( with_hessian ) {
//    HLBFGS();
  } else {
    HLBFGS(dim, M, x, evalfunc, 0, HLBFGS_UPDATE_Hessian, newiteration, parameter, info);
  }
  return 0;
}

int constrained_newton_solve(double *x, const size_t dim, shared_ptr<Functional<double>> &f, shared_ptr<Constraint<double>> &c) {
  if ( dim != f->Nx() || dim != c->Nx() ) {
    cerr << "[error] dim not match\n";
    return __LINE__;
  }
  const size_t fdim = c->Nf();
  UmfPackLU<SparseMatrix<double>> sol;
  VectorXd X = VectorXd::Zero(dim+fdim);
  std::copy(x, x+dim, X.data());
  VectorXd xprev = X.head(dim);
  for (size_t iter = 0; iter < MAX_ITER; ++iter) {
    VectorXd rhs(dim+fdim); {
      rhs.setZero();
      f->Gra(&X[0], &rhs[0]);
      c->Val(&X[0], &rhs[dim]);
    }
    SparseMatrix<double> LHS(dim+fdim, dim+fdim); {
      vector<Triplet<double>> trips;
      f->Hes(&X[0], &trips);
      const auto begin = trips.end();
      c->Jac(&X[0], dim, &trips);
      const auto end = trips.end();
      for (auto it = begin; it != end; ++it) {

      }
    }
  }
  std::copy(X.data(), X.data()+dim, x);
  return 0;
}

int gauss_newton_solve(double *x, const size_t dim, std::shared_ptr<Constraint<double>> &f) {
  if ( dim != f->Nx() ) {
    return __LINE__;
  }
  const size_t fdim = f->Nf();
  for (size_t i = 0; i < MAX_ITER; ++i) {
    VectorXd value(fdim); {
      f->Val(x, &value[0]);
    }
    SparseMatrix<double> J(fdim, dim); {

    }

  }
  return 0;
}

}
