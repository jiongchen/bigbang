#include "geom_util.h"

#include <jtflib/mesh/mesh.h>

using namespace std;
using namespace jtf::mesh;
using namespace zjucad::matrix;

namespace bigbang {

void get_edge_elem(const mati_t &tris, mati_t &edge) {
  edge2cell_adjacent *e2c = edge2cell_adjacent::create(tris, false);
  edge.resize(2, e2c->edges_.size());
#pragma omp parallel
  for (size_t i = 0; i < edge.size(2); ++i) {
    edge(0, i) = e2c->edges_[i].first;
    edge(1, i) = e2c->edges_[i].second;
  }
  delete e2c;
}

void get_diam_elem(const mati_t &tris, mati_t &diam) {
  edge2cell_adjacent *ea = edge2cell_adjacent::create(tris, false);
  mati_t bd_ed_id;
  get_boundary_edge_idx(*ea, bd_ed_id);
  diam.resize(4, ea->edges_.size()-bd_ed_id.size());
  for(size_t ei = 0, di = 0; ei < ea->edges_.size(); ++ei) {
    pair<size_t, size_t> nb_tr_id = ea->edge2cell_[ei];
    if( ea->is_boundary_edge(nb_tr_id) ) continue;
    diam(colon(1, 2), di) = ea->get_edge(ei);
    // orient
    bool need_swap = true;
    for(size_t k = 0; k < 3; ++k) {
      if( diam(1, di) == tris(k, nb_tr_id.first) ) {
        if( diam(2, di) != tris((k+1)%3, nb_tr_id.first) )
          need_swap = false;
      }
    }
    if( need_swap )
      swap(diam(1, di), diam(2, di));
    diam(0, di) = zjucad::matrix::sum(tris(colon(), nb_tr_id.first))
        - zjucad::matrix::sum(diam(colon(1, 2), di));
    diam(3, di) = zjucad::matrix::sum(tris(colon(), nb_tr_id.second))
        - zjucad::matrix::sum(diam(colon(1, 2), di));
    ++di;
  }
  delete ea;
}

double calc_tri_area(const matd_t &vert) {
  matd_t edge = vert(colon(), colon(1, 2))-vert(colon(), colon(0, 1));
  return 0.5*norm(cross(edge(colon(), 0), edge(colon(), 1)));
}

}
