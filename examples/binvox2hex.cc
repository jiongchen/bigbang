//
// This example program reads a .binvox file and writes
// the extracted hexhedral mesh to vtk format
//
// 0 = empty voxel
// 1 = filled voxel
// A newline is output after every "dim" voxels (depth = height = width = dim)
//
// Note that this ASCII version is not supported by "viewvox" and "thinvox"
//
// The x-axis is the most significant axis, then the z-axis, then the y-axis.
//

#include <string>
#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <vector>
#include <algorithm>
#include <zjucad/matrix/matrix.h>

#include "src/config.h"
#include "src/vtk.h"

using namespace std;
using namespace zjucad::matrix;

typedef unsigned char byte;
static int version;
static int depth, height, width;
static int size;
static byte *voxels = 0;
static float tx, ty, tz;
static float scale;

#define SQUARE(x) ((x)*(x))

struct point {
  double x, y, z;
  bool operator ==(const point &other) const {
    return SQUARE(x-other.x)+SQUARE(y-other.y)+SQUARE(z-other.z) < 1e-8;
  }
};

int read_binvox(string filespec)
{
  ifstream *input = new ifstream(filespec.c_str(), ios::in | ios::binary);

  //
  // read header
  //
  string line;
  *input >> line;  // #binvox
  if (line.compare("#binvox") != 0) {
    cout << "Error: first line reads [" << line << "] instead of [#binvox]" << endl;
    delete input;
    return 0;
  }
  *input >> version;
  cout << "reading binvox version " << version << endl;

  depth = -1;
  int done = 0;
  while(input->good() && !done) {
    *input >> line;
    if (line.compare("data") == 0) done = 1;
    else if (line.compare("dim") == 0) {
      *input >> depth >> height >> width;
    }
    else if (line.compare("translate") == 0) {
      *input >> tx >> ty >> tz;
    }
    else if (line.compare("scale") == 0) {
      *input >> scale;
    }
    else {
      cout << "  unrecognized keyword [" << line << "], skipping" << endl;
      char c;
      do {  // skip until end of line
        c = input->get();
      } while(input->good() && (c != '\n'));

    }
  }
  if (!done) {
    cout << "  error reading header" << endl;
    return 0;
  }
  if (depth == -1) {
    cout << "  missing dimensions in header" << endl;
    return 0;
  }

  size = width * height * depth;
  voxels = new byte[size];
  if (!voxels) {
    cout << "  error allocating memory" << endl;
    return 0;
  }

  //
  // read voxel data
  //
  byte value;
  byte count;
  int index = 0;
  int end_index = 0;
  int nr_voxels = 0;

  input->unsetf(ios::skipws);  // need to read every byte now (!)
  *input >> value;  // read the linefeed char

  while((end_index < size) && input->good()) {
    *input >> value >> count;

    if (input->good()) {
      end_index = index + count;
      if (end_index > size) return 0;
      for(int i=index; i < end_index; i++) voxels[i] = value;

      if (value) nr_voxels += count;
      index = end_index;
    }  // if file still ok

  }  // while

  input->close();
  cout << "  read " << nr_voxels << " voxels" << endl;

  return 1;

}

int binvox_to_hexmesh(matrix<size_t> &hexs, matrix<double> &nods) {
  const size_t nbrhex = std::count(voxels, voxels+size, 1);
  hexs.resize(8, nbrhex);
  cout << "[info] hex number: " << nbrhex << endl;

  const size_t dim = depth;
  const double hd = 0.5/dim;
  const double dir[8][3] = {{-hd,-hd,-hd}, {+hd,-hd,-hd}, {-hd,+hd,-hd}, {+hd,+hd,-hd},
                            {-hd,-hd,+hd}, {+hd,-hd,+hd}, {-hd,+hd,+hd}, {+hd,+hd,+hd}};
  // extract eight vertices of a cube
  vector<point> temp;
  size_t hex_count = 0;
  for (size_t x = 0; x < dim; ++x) {
    for (size_t y = 0; y < dim; ++y) {
      for (size_t z = 0; z < dim; ++z) {
        const size_t idx = x*dim*dim+z*dim+y;
        if ( voxels[idx] ) {
          double xyz[3] = {(x+0.5)/dim, (y+0.5)/dim, (z+0.5)/dim};
          for (size_t k = 0; k < 8; ++k) {
            point curr = {xyz[0]+dir[k][0], xyz[1]+dir[k][1], xyz[2]+dir[k][2]};
            auto it = std::find(temp.begin(), temp.end(), curr);
            if ( it == temp.end() ) {
              temp.push_back(curr);
              hexs(k, hex_count) = temp.size()-1;
            } else {
              hexs(k, hex_count) = it-temp.begin();
            }
          }
          ++hex_count;
        }
      }
    }
  }
  ASSERT(hex_count == nbrhex);
  cout << "[info] vertices number: " << temp.size() << endl;

  nods.resize(3, temp.size());
#pragma omp parallel for
  for (size_t i = 0; i < nods.size(2); ++i) {
    nods(0, i) = temp[i].x;
    nods(1, i) = temp[i].y;
    nods(2, i) = temp[i].z;
  }
  // scale and translate the coordinates
  nods *= scale;
  nods(0, colon()) += tx;
  nods(1, colon()) += ty;
  nods(2, colon()) += tz;

  printf("\ndim: %zu\n", dim);
  printf("translate: (%lf, %lf, %lf)\n", tx, ty, tz);
  printf("scale: %lf\n", scale);

  return 0;
}


int main(int argc, char **argv)
{
  if ( argc != 3 ) {
    cout << "Usage: binvox2hex <binvox filename> <hexmesh filename>\n\n";
    return __LINE__;
  }

  if ( !read_binvox(argv[1]) ) {
    cout << "Error reading [" << argv[1] << "]" << endl << endl;
    return __LINE__;
  }

  matrix<size_t> hexs;
  matrix<double> nods;
  binvox_to_hexmesh(hexs, nods);

  ofstream os(argv[2]);
  hex2vtk(os, &nods[0], nods.size(2), &hexs[0], hexs.size(2));

  cout << "\ndone" << endl << endl;
  return 0;
}
