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
    double len = 0.1;//*(rod.col(i)-rod.col(i+1)).norm();
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

  // write config file first
  ofstream ofs(json["outdir"].asString()+"/config.json");
  Json::StyledWriter styledWriter;
  ofs << styledWriter.write(json) << endl;
  ofs.close();

  Matrix<double, 3, 2> ends = Matrix<double, 3, 2>::Zero();
  ends(0, 1) = json["length"].asDouble();
  Matrix3Xd rod;
  generate_rod(ends, json["size"].asUInt(), rod);

  rod_material material = {
    json["timestep"].asDouble(),
    json["radius"].asDouble(),
    json["density"].asDouble(),
    json["young_modulus"].asDouble(),
    json["shearing_modulus"].asDouble(),
    json["stretch_modulus"].asDouble(),
    json["spring_const"].asDouble()
  };

  cosserat_solver solver(rod, material);
  {
    string outfile = json["outdir"].asString()+"/rest.vtk";
    draw_rod(outfile.c_str(), rod);
  }

  // parse transform
  rod *= json["transform"]["scale"].asDouble();
  // initialize frame
  Matrix4Xd frm;
  frm.resize(NoChange, rod.cols()-1);
  Matrix3d u0;
  u0 << 0, 0, 1,
      1, 0, 0,
      0, 1, 0;
  const double frm_cycle = json["frame_cycle"].asDouble();
  cout << "[Info] frame cycle: " << frm_cycle << endl;
  for (size_t i = 0; i < frm.cols(); ++i) {
    Matrix3d rot = AngleAxisd(i*frm_cycle*M_PI/(frm.cols()-1), -Vector3d::UnitX()).toRotationMatrix();
    frm.col(i) = Quaterniond(rot*u0).coeffs();
  }

  // init solver
  solver.init_rod(rod, frm);
  {
    string outfile = json["outdir"].asString()+"/init_rod.vtk";
    draw_rod(outfile.c_str(), rod);
    outfile = json["outdir"].asString()+"/init_frame.vtk";
    draw_frame(outfile.c_str(), rod, frm);
  }

  // parse handles
  for (int i = 0; i < json["handles"].size(); ++i) {
    const size_t id = json["handles"][i].asUInt();
    if ( id >= rod.cols() ) {
      cerr << "[Error] vert index over range\n";
      exit(EXIT_FAILURE);
    }
    cout << "[Info] pin vertex " << id << endl;
    solver.pin_down_vert(id, &rod(0, id));
  }

  for (size_t i = 0; i < json["frames"].asUInt(); ++i) {
    cout << "[Info] frame " << i << endl;
    {
      char outfile[256];
      sprintf(outfile, "%s/deform_%zu.vtk", json["outdir"].asString().c_str(), i);
      draw_rod(outfile, solver.get_rod_pos());
      sprintf(outfile, "%s/deform_frm_%zu.vtk", json["outdir"].asString().c_str(), i);
      draw_frame(outfile, solver.get_rod_pos(), solver.get_frame());
    }
    solver.advance(json["max_iter"].asUInt(), json["tolerance"].asDouble());
  }

  cout << "[Info] done\n";
  return 0;
}
