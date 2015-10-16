//
// This example program reads a .binvox file and writes
// an ASCII version of the same file called "voxels.txt"
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
#include <zjucad/matrix/matrix.h>

#include "src/vtk.h"

using namespace std;
using namespace zjucad::matrix;
using mati_t=zjucad::matrix::matrix<size_t>;
using matd_t=zjucad::matrix::matrix<double>;

typedef unsigned char byte;
static int version;
static int depth, height, width;
static int size;
static byte *voxels = 0;
static float tx, ty, tz;
static float scale;

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

int from_point_to_hex(const mati_t &pts, mati_t &hex, matd_t &nods) {

}

int binvox_to_hex_vtk(const char *file) {
  ifstream ifs(file);
  if ( ifs.fail() ) {
    cerr << "[error] can not open " << file << endl;
    return __LINE__;
  }

  const size_t MAX_LEN = 1024;
  char buff[MAX_LEN];
  size_t dim;
  double t[3];
  double s;
  char flag;
  vector<char> has;

  ifs.getline(buff, MAX_LEN);
  ifs >> buff >> dim >> dim >> dim;
  ifs >> buff >> t[0] >> t[1] >> t[2];
  ifs >> buff >> s;
  ifs >> buff;
  while ( !ifs.eof() ) {
    ifs >> flag;
    has.push_back(flag);
  }

  vector<double> coord;
  for (size_t x = 0; x < dim; ++x) {
    for (size_t y = 0; y < dim; ++y) {
      for (size_t z = 0; z < dim; ++z) {
        const size_t idx = x*dim*dim+z*dim+y;
        if ( has[idx] != '0' ) {
          double xyz[3] = {(x+0.5)/dim, (y+0.5)/dim, (z+0.5)/dim};
          coord.push_back(xyz[0]);
          coord.push_back(xyz[1]);
          coord.push_back(xyz[2]);
        }
      }
    }
  }
  const size_t pts_num = coord.size()/3;
  matrix<size_t> pts = colon(0, pts_num-1);
  ofstream os("./extracted_points.vtk");
  point2vtk(os, &coord[0], pts_num, &pts[0], pts_num);

  // extract hex
  mati_t hexs(8, pts_num);
  matd_t nods(3, 5);
  for (size_t i = 0; i < pts_num; ++i) {

  }
//  nods *= s;
//  nods += t;

  printf("\ndim: %zu\n", dim);
  printf("translate: (%lf, %lf, %lf)\n", t[0], t[1], t[2]);
  printf("scale: %lf\n", s);
  printf("number of points: %zu\n", pts_num);

  return 0;
}


int main(int argc, char **argv)
{
  if (argc != 2) {
    cout << "Usage: read_binvox <binvox filename>" << endl << endl;
    exit(1);
  }

  if (!read_binvox(argv[1])) {
    cout << "Error reading [" << argv[1] << "]" << endl << endl;
    exit(1);
  }

  //
  // now write the data to as ASCII
  //
  ofstream *out = new ofstream("voxels.txt");
  if(!out->good()) {
    cout << "Error opening [voxels.txt]" << endl << endl;
    exit(1);
  }

  cout << "Writing voxel data to ASCII file..." << endl;

  *out << "#binvox ASCII data" << endl;
  *out << "dim " << depth << " " << height << " " << width << endl;
  *out << "translate " << tx << " " << ty << " " << tz << endl;
  *out << "scale " << scale << endl;
  *out << "data" << endl;

  for(int i=0; i < size; i++) {
    *out << (char) (voxels[i] + '0') << " ";
    if (((i + 1) % width) == 0) *out << endl;
  }

  out->close();

  binvox_to_hex_vtk("voxels.txt");

  cout << "\ndone" << endl << endl;

  return 0;
}
