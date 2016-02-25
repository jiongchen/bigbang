#include <iostream>
#include <boost/filesystem.hpp>
#include <zjucad/matrix/itr_matrix.h>
#include <jtflib/mesh/util.h>

#include "src/readOBJ.h"
#include "src/vtk.h"

using namespace std;
using namespace Eigen;
using namespace zjucad::matrix;

int main(int argc, char *argv[])
{
  Matrix<size_t, -1, 3, RowMajor> f, ft, fn;
  Matrix<double, -1, 3, RowMajor> v, vt3, vn;
  Matrix<double, -1, 2, RowMajor> vt;
  bool flag = igl::readOBJ("../../dat/guy.obj", v, vt, vn, f, ft, fn);
  cout << flag << endl;

  boost::filesystem::create_directory("./IO");
  {
    ofstream os("./IO/orig.vtk");
    tri2vtk(os, v.data(), v.rows(), f.data(), f.rows());
  }
  vt3.setZero(vt.rows(), 3);
  vt3.topLeftCorner(vt3.rows(), 2) = vt;
  {
    ofstream os("./IO/param.vtk");
    tri2vtk(os, vt3.data(), vt3.rows(), ft.data(), ft.rows());
  }

  itr_matrix<const size_t *> F(3, f.rows(), f.data()), FT(3, ft.rows(), ft.data());
  itr_matrix<const double *> V(3, v.rows(), v.data()), VT(3, vt3.rows(), vt3.data());
  for (size_t i = 0; i < F.size(2); ++i) {
    double area3d = jtf::mesh::cal_face_area(V(colon(), F(colon(), i)));
    double area2d = jtf::mesh::cal_face_area(VT(colon(), FT(colon(), i)));
    cout << "diff: " << area3d-area2d << endl;
  }

  cout << "done\n";
  return 0;
}
