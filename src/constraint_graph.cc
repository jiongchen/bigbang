#include "constraint_graph.h"

namespace bigbang {

static graph*
init_graph(const size_t numpts) {

}

static void
release_graph(graph *G) {
  free(G->u);

}



}
