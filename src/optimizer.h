#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include <memory>

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

int lbfgs_solve(double *x, const size_t dim, const pfunc &f, const opt_args *args);

int constrained_newton_solve(double *x, const size_t dim, const pfunc &f, const pcons &c);

int gauss_newton_solve(double *x, const size_t dim, const pcons &f);

}
#endif
