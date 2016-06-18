#ifndef EXAMPLE_BASED_ELASTIC_H
#define EXAMPLE_BASED_ELASTIC_H

#include <zjucad/matrix/matrix.h>

namespace bigbang {

using mati_t=zjucad::matrix::matrix<size_t>;
using matd_t=zjucad::matrix::matrix<double>;

void strain_calc_test();
void numeric_diff_test();

}

#endif
