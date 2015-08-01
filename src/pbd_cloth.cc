#include "pbd_cloth.h"

#include <jtflib/mesh/mesh.h>

using namespace std;
using namespace zjucad::matrix;
using namespace jtf::mesh;

namespace bigbang {

extern "C" {

void calc_streth_(double *val, const double *x);
void calc_streth_jac_(double *jac, const double *x);

void calc_bend_(double *val, const double *x);
void calc_bend_jac_(double *jac, const double *x);

}

void get_dihedral_elements(const matrix<size_t> &tris, matrix<size_t> &ele)
{
  unique_ptr<edge2cell_adjacent> ea(edge2cell_adjacent::create(tris));
  matrix<size_t> bd_ed_id;
  get_boundary_edge_idx(*ea, bd_ed_id);
  ele.resize(4, ea->edges_.size()-bd_ed_id.size());
  for(size_t ei = 0, di = 0; ei < ea->edges_.size(); ++ei) {
    pair<size_t, size_t> nb_tr_id = ea->edge2cell_[ei];
    if(ea->is_boundary_edge(nb_tr_id)) continue;
    ele(colon(1, 2), di) = ea->get_edge(ei);
    // orient
    bool need_swap = true;
    for(size_t k = 0; k < 3; ++k) {
      if(ele(1, di) == tris(k, nb_tr_id.first)) {
        if(ele(2, di) != tris((k+1)%3, nb_tr_id.first))
          need_swap = false;
      }
    }
    if(need_swap)
      swap(ele(1, di), ele(2, di));
    ele(0, di) = zjucad::matrix::sum(tris(colon(), nb_tr_id.first))
      - zjucad::matrix::sum(ele(colon(1, 2), di));
    ele(3, di) = zjucad::matrix::sum(tris(colon(), nb_tr_id.second))
      - zjucad::matrix::sum(ele(colon(1, 2), di));
    ++di;
  }
}


}
