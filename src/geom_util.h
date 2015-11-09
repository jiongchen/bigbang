#ifndef GEOMETRY_UTIL_H
#define GEOMETRY_UTIL_H

#include <zjucad/matrix/matrix.h>

using mati_t=zjucad::matrix::matrix<size_t>;
using matd_t=zjucad::matrix::matrix<double>;

namespace bigbang {

void get_edge_elem(const mati_t &tris, mati_t &edge);

void get_diam_elem(const mati_t &tris, mati_t &diam);

double calc_tri_area(const matd_t &vert);

}
#endif
