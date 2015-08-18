#include <iostream>
#include <fstream>
#include <boost/filesystem.hpp>

#include "src/pbd_cloth.h"

using namespace std;
using namespace bigbang;

int main(int argc, char *argv[])
{
  if ( argc != 2 ) {
    cerr << "# Usage: " << argv[0] << " model.obj\n";
    return __LINE__;
  }
  boost::filesystem::create_directory("./PBD");

  pbd_cloth_solver solver;
  solver.load_model_from_obj(argv[1]);
  solver.save_model_to_vtk("./PBD/rest.vtk");

  solver.test_calc_length();
  solver.test_calc_dihedral_angle();
//  solver.test_edge_extraction();
//  solver.test_diamond_extraction();

  solver.init();
  solver.set_mass_matrix(1.0);
  solver.set_time_step(0.01);
  solver.apply_gravity();
  solver.attach_vert(0);
  solver.attach_vert(1);

  solver.precompute();
  char file[256];
  for (size_t i = 0; i < 500; ++i) {
    cout << "[info] frame " << i << endl;
    sprintf(file, "./PBD/frame_%zu.vtk", i);
    solver.save_model_to_vtk(file);

    solver.advance();
  }

  cout << "[info] done\n";
  return 0;
}
