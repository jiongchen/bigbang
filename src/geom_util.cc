#include "geom_util.h"

#include <ANN/ANN.h>
#include <jtflib/mesh/mesh.h>
#include <jtflib/mesh/util.h>
#include <Eigen/Geometry>
#include <zjucad/matrix/io.h>
#include <hjlib/math/blas_lapack.h>
#include <zjucad/matrix/lapack.h>

using namespace std;
using namespace jtf::mesh;
using namespace zjucad::matrix;
using namespace Eigen;

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

int remove_extra_verts(mati_t &cell, matd_t &nods) {
  set<size_t> used(cell.begin(), cell.end());
  if ( used.size() == nods.size(2) )
    return 0;
  mati_t small_to_big(used.size(), 1);
  std::copy(used.begin(), used.end(), small_to_big.begin());
  mati_t big_to_small(used.size(), 1);
  big_to_small(small_to_big) = colon(0, used.size()-1);

  for (auto &pid : cell)
    pid = big_to_small(pid);

  matd_t newnods(nods.size(1), used.size());
  for (size_t i = 0; i < newnods.size(2); ++i)
    newnods(colon(), i) = nods(colon(), small_to_big[i]);
  nods = newnods;
  return 0;
}

void extract_tet_surface(const mati_t &tets, const matd_t &nods, mati_t &tris, matd_t &snods) {
  using jtf::mesh::face2tet_adjacent;
  shared_ptr<face2tet_adjacent> f2t(face2tet_adjacent::create(tets));
  jtf::mesh::get_outside_face(*f2t, tris, true);
  snods = nods;
  remove_extra_verts(tris, snods);
}

void shrink_surface(const mati_t &tris, matd_t &nods, const double dist) {
  matd_t normal;
  jtf::mesh::cal_point_normal(tris, nods, normal);
  nods -= dist*normal;
}

void subdivide_surface(const mati_t &tris, const matd_t &nods, mati_t &ftris, matd_t &fnods) {
  using jtf::mesh::edge2cell_adjacent;
  unique_ptr<edge2cell_adjacent> e2c(edge2cell_adjacent::create(tris, false));
  ftris.resize(3, 4 * tris.size(2));
  fnods.resize(3, nods.size(2) + e2c->edges_.size());

  vector<int> vis(e2c->edges_.size());
  std::fill(vis.begin(), vis.end(), -1);
  fnods(colon(), colon(0, nods.size(2) - 1)) = nods;

  int ptris = 0, pnods = nods.size(2);
  for (size_t i = 0; i < tris.size(2); ++i) {
    vector<size_t> mid(3);
    for (size_t j = 0; j < 3; ++j) {
      size_t P = tris(j, i);
      size_t Q = tris((j + 1) % 3, i);
      size_t eid = e2c->get_edge_idx(P, Q);
      if ( vis[eid] == -1 ) {
        mid[j] = vis[eid] = pnods;
        fnods(colon(), pnods) = 0.5 * (nods(colon(), P) + nods(colon(), Q));
        ++pnods;
      } else {
        mid[j] = vis[eid];
      }
    }
    /// @ guarantee the order of surface wouldn't change during subdivision
    const size_t small_tris0[] = {mid[1], mid[2], mid[0]};
    ftris(colon(), ptris++) = itr_matrix<const size_t *>(3, 1, small_tris0);
    const size_t small_tris1[] = {tris(0, i), mid[0], mid[2]};
    ftris(colon(), ptris++) = itr_matrix<const size_t *>(3, 1, small_tris1);
    const size_t small_tris2[] = {tris(1, i), mid[1], mid[0]};
    ftris(colon(), ptris++) = itr_matrix<const size_t *>(3, 1, small_tris2);
    const size_t small_tris3[] = {tris(2, i), mid[2], mid[1]};
    ftris(colon(), ptris++) = itr_matrix<const size_t *>(3, 1, small_tris3);
  }
}

int interp_pts_in_tets(const matd_t &v, const mati_t &tet, const matd_t &pts, csc_t &coef)
{
  const size_t tn = tet.size(2), pn = pts.size(2);

  vector<double*> pv(tn);
  matrix<double> tet_center(3, tn); {
    for(int i = 0; i < tn; ++i) {
      tet_center(colon(), i) = v(colon(), tet(colon(), i))*ones<double>(4, 1)/4;
      pv[i] = &tet_center(0, i);
    }
  }

  auto_ptr<ANNkd_tree> kdt(new ANNkd_tree(&pv[0], tn, v.size(1), 32));
  matrix<matrix<double> > bary_op(tn); {
    for(int i = 0; i < tn; ++i) {
      matrix<double> v44 = ones<double>(4, 4);
      for(int j = 0; j < 4; ++j)
        v44(colon(0, 2), j) = v(colon(), tet(j, i));
      inv(v44);
      bary_op[i] = v44;
    }
    cout << "create bary-coor operators success." << endl;
  }

  vector<Triplet<double>> trips;

  matrix<double> pt, w;
  const int ave_k = 40, iter_n = 4;
  const int max_k = static_cast<int>(40*floor(pow(2.0, iter_n)+0.5));
  matrix<double> dist(max_k);
  matrix<int> idx(max_k);
  double min_good = 1;
  int outside_cnt = 0;

  for(int pi = 0; pi < pn; ++pi) {
    if((pi%1000) == 0)
      cerr << "process " << pi << endl;

    pt = pts(colon(), pi);
    pair<int, double> best_t(-1, -10);

    for(int ki = 0, k = ave_k; ki < iter_n && k < max_k; ++ki, k*=2) {
      if(k > max_k)
        k = max_k;
      const double r2 = 1e1;
      kdt->annkSearch(&pt[0], max_k, &idx[0], &dist[0], 1e-10);
      for(int ti = (k > 40)?k/2:0; ti < k; ++ti) {
        int t_idx = idx[ti];
        w = bary_op[t_idx](colon(0, 3), colon(0, 2))*pt + bary_op[t_idx](colon(), 3);
        double good = min(w);
        if(best_t.second < good) {
          best_t.second = good;
          best_t.first = t_idx;
        }
        if(best_t.second >= 0)
          break;
      }
      if(best_t.second >= 0)
        break;
    }

    if(best_t.second < 0)
      ++outside_cnt;
    if(best_t.second < min_good)
      min_good = best_t.second;
    if(best_t.first < 0) {
      cout << "Wow, very bad point!!" << endl;
      return __LINE__;
    }

    w = bary_op[best_t.first](colon(0, 3), colon(0, 2))*pt + bary_op[best_t.first](colon(), 3);

    if(fabs(sum(w)-1) > 1e-9) {

      cout << "strange weight." << trans(w);
      cout << "sum : " << sum(w) << endl;
    }
    trips.push_back(Triplet<double>(tet(0, best_t.first), pi, w[0]));
    trips.push_back(Triplet<double>(tet(1, best_t.first), pi, w[1]));
    trips.push_back(Triplet<double>(tet(2, best_t.first), pi, w[2]));
    trips.push_back(Triplet<double>(tet(3, best_t.first), pi, w[3]));
  }
  cout << "outside pt num is: " << outside_cnt << " min_good is: " << min_good << endl;
  coef.resize(v.size(2), pn);
  coef.reserve(trips.size());
  coef.setFromTriplets(trips.begin(), trips.end());
  return 0;
}

}
