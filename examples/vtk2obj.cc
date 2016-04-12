#include <iostream>
#include <jtflib/mesh/io.h>

#include "src/vtk.h"

using namespace std;
using namespace zjucad::matrix;

static int tri_mesh_read_from_vtk(const char *path, matrix<double> *node = 0, matrix<size_t> *tris = 0) {
  ifstream ifs(path);
  if ( ifs.fail() ) {
    cerr << "[info] " << "can not open file" << path << endl;
    return __LINE__;
  }
  matrix<double> node0;
  matrix<int> tet1;

  string str;
  int point_num = 0,cell_num = 0;

  while(!ifs.eof()){
    ifs >> str;
    if(str == "POINTS"){
      ifs >> point_num >> str;
      node0.resize(3, point_num);
      for(size_t i = 0;i < point_num; ++i){
        for(size_t j = 0;j < 3; ++j)
          ifs >> node0(j, i);
      }
      continue;
    }
    if(str == "CELLS"){
      ifs >> cell_num >> str;
      int point_number_of_cell = 0;
      vector<size_t> tet_temp;
      for(size_t ci = 0; ci < cell_num; ++ci){
        ifs >> point_number_of_cell;
        if(point_number_of_cell != 3){
          for(size_t i = 0; i < point_number_of_cell; ++i)
            ifs >> str;
        }else{
          int p;
          for(size_t i = 0; i < point_number_of_cell; ++i){
            ifs >> p;
            tet_temp.push_back(p);
          }
        }
      }
      tet1.resize(3, tet_temp.size()/3);
      copy(tet_temp.begin(), tet_temp.end(), tet1.begin());
    }
  }
  ifs.close();
  *node = node0;
  *tris = tet1;
  return 0;
}

int main(int argc, char *argv[])
{
  if ( argc != 3 ) {
    cerr << "# usage: obj2vtk model.vtk model.obj\n";
    return __LINE__;
  }
  matrix<size_t> tris;
  matrix<double> nods;
  tri_mesh_read_from_vtk(argv[1], &nods, &tris);
  jtf::mesh::save_obj(argv[2], tris, nods);
  cout << "success\n";
  return 0;
}
