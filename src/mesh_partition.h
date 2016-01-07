#ifndef MESH_PARTITION_H
#define MESH_PARTITION_H

#include <zjucad/matrix/matrix.h>
#include <boost/config.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/dijkstra_shortest_paths.hpp>
#include <boost/graph/graph_traits.hpp>
#include <boost/graph/iteration_macros.hpp>
#include <boost/graph/properties.hpp>
#include <boost/property_map/property_map.hpp>

namespace bigbang {

typedef zjucad::matrix::matrix<size_t> mati_t;
typedef zjucad::matrix::matrix<double> matd_t;
typedef double Weight;
typedef boost::property<boost::edge_weight_t, Weight> WeightProperty;
typedef boost::property<boost::vertex_name_t, size_t> NameProperty;
typedef boost::adjacency_list<boost::listS, boost::vecS, boost::undirectedS, NameProperty, WeightProperty> Graph;
typedef boost::graph_traits<Graph>::vertex_descriptor Vertex;
typedef boost::property_map<Graph, boost::vertex_index_t>::type IndexMap;
typedef boost::property_map<Graph, boost::vertex_name_t>::type NameMap;
typedef boost::iterator_property_map<Vertex*, IndexMap, Vertex, Vertex&> PredecessorMap;
typedef boost::iterator_property_map<Weight*, IndexMap, Weight, Weight&> DistanceMap;

struct ptn_to_patch {
  size_t id_patch;
  size_t id_center;
  double dist;
};

class mesh_partition
{
public:
  mesh_partition(const mati_t &tris, const matd_t &nods);
  int init(std::vector<ptn_to_patch> &result);
  int run(const size_t cluster_num, std::vector<ptn_to_patch> &result);
  size_t get_actual_patch_num() const { return act_num_; }
  int visualize_patches(const char *directory, std::vector<ptn_to_patch> &result) const;
private:
  size_t act_num_;
  const matd_t &nods_;
  std::shared_ptr<Graph> g_;
};

}

#endif
