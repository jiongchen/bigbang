#ifndef MASS_MATRIX_H
#define MASS_MATRIX_H

#include <zjucad/matrix/matrix.h>
#include <Eigen/Sparse>

namespace bigbang {

using mati_t = zjucad::matrix::matrix<size_t>;
using matd_t = zjucad::matrix::matrix<double>;
using spmat_t = Eigen::SparseMatrix<double>;

int calc_mass_matrix(const mati_t &cell,
                     const matd_t &nods,
                     const double rho,
                     const size_t dim,
                     spmat_t *M,
                     bool lumped);

}

#endif
