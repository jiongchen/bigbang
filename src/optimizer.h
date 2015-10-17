#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include <memory>

namespace bigbang {

template <typename T>
class Functional;

template <typename T>
class Constraint;

int newton_solve(double *x, const size_t dim, std::shared_ptr<Functional<double>> &f);

int lbfgs_solve(double *x, const size_t dim, std::shared_ptr<Functional<double>> &f, const int M, const int T, bool with_hessian);

int constrained_newton_solve(double *x, const size_t dim, std::shared_ptr<Functional<double>> &f, std::shared_ptr<Constraint<double>> &c);

int gauss_newton_solve(double *x, const size_t dim, std::shared_ptr<Constraint<double>> &f);

}
#endif
