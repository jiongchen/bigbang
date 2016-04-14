#include <iostream>
#include <fstream>
#include <boost/filesystem.hpp>
#include <Eigen/Geometry>

#include "src/json.h"
#include "src/cosserat.h"
#include "src/vtk.h"

using namespace std;
using namespace Eigen;
using namespace bigbang;

static int draw_rod(const char *file, Matrix3Xd &rod) {
  Matrix<size_t, 2, -1> cell;
  cell.resize(NoChange, rod.cols()-1);
  for (size_t i = 0; i < cell.cols(); ++i) {
    cell(0, i) = i; cell(1, i) = i+1;
  }
  ofstream ofs(file);
  if ( ofs.fail() )
    return __LINE__;
  line2vtk(ofs, rod.data(), rod.cols(), cell.data(), cell.cols());
  ofs.close();
  return 0;
}

static int draw_frame(const char *file, const Matrix3Xd &rod, const Matrix4Xd &q) {
  Matrix<size_t, 2, -1> cell;
  cell.resize(NoChange, 3*(rod.cols()-1));
  Matrix<double, 3, -1> nods;
  nods.resize(NoChange, 4*(rod.cols()-1));
  for (size_t i = 0; i < rod.cols()-1; ++i) {
    double len = 0.5*(rod.col(i)-rod.col(i+1)).norm();
    Matrix3d R = Quaternion<double>(q.col(i)).toRotationMatrix();
    nods.col(4*i+0) = (rod.col(i)+rod.col(i+1))/2;
    nods.col(4*i+1) = nods.col(4*i+0)+R.col(0).normalized()*len;
    nods.col(4*i+2) = nods.col(4*i+0)+R.col(1).normalized()*len;
    nods.col(4*i+3) = nods.col(4*i+0)+R.col(2).normalized()*len;
  }
  for (size_t i = 0; i < rod.cols()-1; ++i) {
    cell(0, 3*i+0) = 4*i+0;
    cell(1, 3*i+0) = 4*i+1;
    cell(0, 3*i+1) = 4*i+0;
    cell(1, 3*i+1) = 4*i+2;
    cell(0, 3*i+2) = 4*i+0;
    cell(1, 3*i+2) = 4*i+3;
  }
  ofstream ofs(file);
  if ( ofs.fail() )
    return __LINE__;
  line2vtk(ofs, nods.data(), nods.cols(), cell.data(), cell.cols());
  ofs.close();
  return 0;
}

int main(int argc, char *argv[])
{
  if ( argc != 2 ) {
    cerr << "# usage: ./test_cosserat config.json\n";
    return __LINE__;
  }
  Json::Reader reader;
  Json::Value json;
  ifstream ifs(argv[1]);
  if ( ifs.fail() ) {
    cerr << "[Error] can't open " << argv[1] << endl;
    return __LINE__;
  }
  if ( !reader.parse(ifs, json) ) {
    cerr << "[Error] " << reader.getFormattedErrorMessages() << endl;
    return __LINE__;
  }
  ifs.close();
  boost::filesystem::create_directories(json["outdir"].asString());

  Matrix<double, 3, 2> ends = Matrix<double, 3, 2>::Zero();
  ends(0, 1) = json["length"].asDouble();

  Matrix3Xd rod;
  generate_rod(ends, json["size"].asUInt(), rod);
  cosserat_solver solver(rod);
  {
    string outfile = json["outdir"].asString()+"/rest.vtk";
    draw_rod(outfile.c_str(), rod);
  }
  Matrix4Xd q;
  init_rod_as_helix(1.0, 0.2, M_PI/4, 1, rod);
  q.setRandom(4, rod.cols()-1);
  for (size_t i = 0; i < q.cols(); ++i)
    q.col(i) = q.col(i).normalized();
  solver.init_rod(rod, q);
  {
    string outfile = json["outdir"].asString()+"/init_rod.vtk";
    draw_rod(outfile.c_str(), rod);
    outfile = json["outdir"].asString()+"/init_frame.vtk";
    draw_frame(outfile.c_str(), rod, q);
  }

  rod_material material = {
    json["timestep"].asDouble(),
    json["radius"].asDouble(),
    json["density"].asDouble(),
    json["young_modulus"].asDouble(),
    json["shearing_modulus"].asDouble(),
    json["stretch_modulus"].asDouble(),
    json["spring_const"].asDouble()
  };
  solver.config_material(material);

  solver.precompute();
  solver.advance(json["max_iter"].asUInt(), json["tolerance"].asDouble());
  {
    string outfile = json["outdir"].asString()+"/deform.vtk";
    draw_rod(outfile.c_str(), solver.get_rod_pos());
  }

  cout << "[Info] done\n";
  return 0;
}
