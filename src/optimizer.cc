#include "optimizer.h"

#include <iostream>

#include "config.h"
#include "def.h"

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
        cout << "\t@energy value: " << value << endl << endl;
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
      cout << "[info] converged\n\n";
      break;
    }
    Xprev = Xstar;
    Xstar += dx;
  }
  X = Xstar;
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
