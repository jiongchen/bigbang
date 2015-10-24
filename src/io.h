#ifndef IO_H
#define IO_H

#include <zjucad/matrix/matrix.h>

using mati_t=zjucad::matrix::matrix<size_t>;
using matd_t=zjucad::matrix::matrix<double>;

namespace bigbang {

int hex_mesh_read_from_vtk(const char *path, matd_t *node=nullptr, mati_t *hex=nullptr);

}
#endif
