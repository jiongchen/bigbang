#include "io.h"

#include <iostream>
#include <fstream>

using namespace std;
using namespace zjucad::matrix;

namespace bigbang {

int read_fixed_verts(const char *filename, std::vector<size_t> &fixed) {
  fixed.clear();
  ifstream ifs(filename);
  if ( ifs.fail() ) {
    cerr << "[error] can not open " << filename << endl;
    return __LINE__;
  }
  size_t temp;
  while ( ifs >> temp ) {
    fixed.push_back(temp);
  }
  cout << "[info] fixed verts number: " << fixed.size() << endl;
  ifs.close();
  return 0;
}

int hex_mesh_read_from_vtk(const char *path, matd_t *node, mati_t *hex) {
  ifstream ifs(path);
  if(ifs.fail()) {
    cerr << "[info] " << "can not open file" << path << endl;
    return __LINE__;
  }

  matrix<double> node0;
  matrix<int> hex1;

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
      vector<size_t> hex_temp;
      for(size_t ci = 0; ci < cell_num; ++ci){
        ifs >> point_number_of_cell;
        if(point_number_of_cell != 8){
          for(size_t i = 0; i < point_number_of_cell; ++i)
            ifs >> str;
        }else{
          int p;
          for(size_t i = 0; i < point_number_of_cell; ++i){
            ifs >> p;
            hex_temp.push_back(p);
          }
        }
      }
      hex1.resize(8, hex_temp.size()/8);
      copy(hex_temp.begin(), hex_temp.end(), hex1.begin());
    }
  }

  vector<size_t> one_hex(hex1.size(1));
  for(size_t hi = 0; hi < hex1.size(2); ++hi){
    copy(hex1(colon(),hi).begin(), hex1(colon(), hi).end(), one_hex.begin());
    hex1(0,hi) = one_hex[6];
    hex1(1,hi) = one_hex[5];
    hex1(2,hi) = one_hex[7];
    hex1(3,hi) = one_hex[4];
    hex1(4,hi) = one_hex[2];
    hex1(5,hi) = one_hex[1];
    hex1(6,hi) = one_hex[3];
    hex1(7,hi) = one_hex[0];
  }

  *node = node0;
  *hex = hex1;
  return 0;
}

}
