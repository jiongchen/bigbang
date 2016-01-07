#include "mesh_partition.h"

#include <iostream>
#include <fstream>
#include <limits>
#include <jtflib/mesh/mesh.h>

#include "vtk.h"

using namespace std;
using namespace zjucad::matrix;
using namespace jtf::mesh;

namespace bigbang {

mesh_partition::mesh_partition(const mati_t &tris, const matd_t &nods)
  : g_(make_shared<Graph>()), act_num_(-1), nods_(nods) {
  shared_ptr<edge2cell_adjacent> e2c(edge2cell_adjacent::create(tris, true));
  for (size_t i = 0; i < nods.size(2); ++i)
    boost::add_vertex(i, *g_);
  for (auto &elem : e2c->edges_) {
    double len = norm(nods(colon(), elem.first)-nods(colon(), elem.second));
    boost::add_edge(elem.first, elem.second, len, *g_);
  }
}

int mesh_partition::init(vector<ptn_to_patch> &result) {
  result.resize(boost::num_vertices(*g_));
  for (auto &elem : result)
    elem.dist = numeric_limits<double>::max();
}

int mesh_partition::run(const size_t cluster_num, vector<ptn_to_patch> &result) {
  vector<double> distances(boost::num_vertices(*g_));

  IndexMap indexMap = boost::get(boost::vertex_index, *g_);
  DistanceMap distanceMap(&distances[0], indexMap);

  size_t start = 0, next, iter;
  for (iter = 0; iter < cluster_num; ++iter) {
    boost::dijkstra_shortest_paths(*g_, start, boost::distance_map(distanceMap));
    // update
    bool updated = false;
    double farest_dist = 0.0;
    for (size_t i = 0; i < result.size(); ++i) {
      if ( distances[i] >= result[i].dist )
        continue;
      result[i].id_patch = iter;
      result[i].id_center = start;
      result[i].dist = distances[i];
      updated = true;
      if ( result[i].dist > farest_dist ) {
        farest_dist = result[i].dist;
        next = i;
      }
    }
    if ( !updated )
      break;
    start = next;
  }
  act_num_ = iter;
  return 0;
}

int mesh_partition::visualize_patches(const char *directory, vector<ptn_to_patch> &result) const {
  vector<vector<size_t>> cluster(act_num_);
  size_t i = 0;
  for (auto &elem : result)
    cluster[elem.id_patch].push_back(i++);

  char outfile[256];
  for (size_t i = 0; i < act_num_; ++i) {
    sprintf(outfile, "%s/patch_%zu.vtk", directory, i);
    ofstream os(outfile);
    point2vtk(os, &nods_[0], nods_.size(2), &cluster[i][0], cluster[i].size());
  }
  return 0;
}

}
