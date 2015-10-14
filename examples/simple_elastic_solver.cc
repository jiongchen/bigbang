#include <iostream>
#include <fstream>
#include <boost/filesystem.hpp>
#include <jtflib/mesh/io.h>

#include "src/vtk.h"
#include "src/optimizer.h"
#include "src/energy.h"

using namespace std;
using namespace bigbang;
using namespace zjucad::matrix;
using mati_t=zjucad::matrix::matrix<size_t>;
using matd_t=zjucad::matrix::matrix<double>;

const double RHO = 1;
const double H = 0.1;
const double YOUNG = 1e4;
const double POISSON = 0.45;

int read_fixed_verts(const char *filename, vector<size_t> &fixed) {
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

#define IMPL_EULER

int main(int argc, char *argv[])
{
  if ( argc != 4 ) {
    cerr << "# Usage: " << "./prog model.tet fix_verts.fv output_prefix\n";
    return __LINE__;
  }
  boost::filesystem::create_directory("./simple");

  mati_t tets;
  matd_t nods;
  jtf::mesh::tet_mesh_read_from_zjumat(argv[1], &nods, &tets); {
    ofstream os("./simple/rest.vtk");
    tet2vtk(os, &nods[0], nods.size(2), &tets[0], tets.size(2));
  }
  vector<size_t> fixed;
  read_fixed_verts(argv[2], fixed);

  // assemble energies
  vector<shared_ptr<Functional<double>>> ebf(4);
  shared_ptr<Functional<double>> energy;
#ifdef IMPL_EULER
  ebf[0] = make_shared<momentum_potential_imp_euler>(tets, nods, RHO, H, 1e0);
#else
  ebf[0] = make_shared<momentum_potential_bdf2>(tets, nods, RHO, H, 1e0);
#endif
  ebf[1] = make_shared<elastic_potential>(tets, nods, elastic_potential::STVK, YOUNG, POISSON, 1e0);
  ebf[2] = make_shared<gravitational_potential>(tets, nods, RHO, 2.0);
  ebf[3] = make_shared<positional_potential>(fixed, nods, 1e3);
  try {
    energy = make_shared<energy_t<double>>(ebf);
  } catch ( exception &e ) {
    cerr << e.what() << endl;
    exit(EXIT_FAILURE);
  }

//  vector<shared_ptr<Constraint<double>>> cbf(1);
//  shared_ptr<Constraint<double>> constraint;

  char outfile[256];
  for (size_t i = 0; i < 200; ++i) {
    cout << "[info] frame " << i << endl;
    sprintf(outfile, "./simple/%s_%zu.vtk", argv[3], i);
    ofstream os(outfile);
    tet2vtk(os, &nods[0], nods.size(2), &tets[0], tets.size(2));

    newton_solve(&nods[0], nods.size(), energy);
    dynamic_pointer_cast<momentum_potential>(ebf[0])->Update(&nods[0]);
  }

  cout << "[info] all done\n";
  return 0;
}
