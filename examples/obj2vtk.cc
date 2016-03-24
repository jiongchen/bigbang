#include <iostream>
#include <jtflib/mesh/io.h>

#include "src/vtk.h"

using namespace std;
using namespace zjucad::matrix;

int main(int argc, char *argv[])
{
  if ( argc != 3 ) {
    cerr << "# usage: obj2vtk model.obj model.vtk\n";
    return __LINE__;
  }
  matrix<size_t> tris;
  matrix<double> nods;
  jtf::mesh::load_obj(argv[1], tris, nods);
  ofstream os(argv[2]);
  tri2vtk(os, &nods[0], nods.size(2), &tris[0], tris.size(2));
  cout << "success\n";
  return 0;
}
