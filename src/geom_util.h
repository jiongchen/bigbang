#ifndef GEOMETRY_UTIL_H
#define GEOMETRY_UTIL_H

#include <Eigen/Sparse>
#include <zjucad/matrix/matrix.h>

using mati_t=zjucad::matrix::matrix<size_t>;
using matd_t=zjucad::matrix::matrix<double>;
using csc_t=Eigen::SparseMatrix<double>;

namespace bigbang {

void get_edge_elem(const mati_t &tris, mati_t &edge);

void get_diam_elem(const mati_t &tris, mati_t &diam);

double calc_tri_area(const matd_t &vert);

void eval_tri_rot(const double *x0, const double *x1, double *R);

int remove_extra_verts(mati_t &cell, matd_t &nods);

void extract_tet_surface(const mati_t &tets, const matd_t &nods, mati_t &tris, matd_t &snods);

void shrink_surface(const mati_t &tris, matd_t &nods, const double dist);

void subdivide_surface(const mati_t &tris, const matd_t &nods, const mati_t &out_tris, matd_t &out_nods);

int interp_pts_in_tets(const matd_t &v, const mati_t &tet, const matd_t &pts, csc_t &coef);

struct partition_info {
  size_t cid;
  size_t pid;
  double d;
};

}
#endif
