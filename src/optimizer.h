#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include <memory>
#include <Eigen/Sparse>

namespace bigbang {

template <typename T>
class Functional;

template <typename T>
class Constraint;

struct opt_args {
  size_t max_iter;
  double eps;
  bool lineseach;
};

using pfunc=std::shared_ptr<Functional<double>>;
using pcons=std::shared_ptr<Constraint<double>>;

int newton_solve(double *x, const size_t dim, const pfunc &f, const opt_args &args);

int newton_solve_with_constrained_dofs(double *x, const size_t dim, const pfunc &f, const std::vector<size_t> &g2l, const opt_args &args);

int lbfgs_solve(double *x, const size_t dim, const pfunc &f, const opt_args &args);

int constrained_newton_solve(double *x, const size_t dim, const pfunc &f, const pcons &c, const opt_args &args);

int gauss_newton_solve(double *x, const size_t dim, const pcons &f);

int apply_jacobi(const Eigen::SparseMatrix<double, Eigen::RowMajor> &A, const Eigen::VectorXd &rhs, Eigen::VectorXd &x);

int apply_gauss_seidel(const Eigen::SparseMatrix<double, Eigen::RowMajor> &A, const Eigen::VectorXd &rhs, Eigen::VectorXd &x, bool increase=true);

}
#endif
