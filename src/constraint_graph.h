#ifndef CONSTRAINT_GRAPH_H
#define CONSTRAINT_GRAPH_H

#include <vector>
#include <cstddef>
#include <memory>

namespace bigbang {

template <typename>
class constraint_piece;

using pcons_t=std::shared_ptr<constraint_piece<double>>;

struct graph {
  size_t pts_num;
  size_t *u, *v;
  size_t *first, *next;
};

class segmenter
{
public:
  segmenter(const std::vector<pcons_t> &cons);
};

}

#endif
