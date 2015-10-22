#include <iostream>
#include <fstream>
#include <boost/filesystem.hpp>

#include "src/vtk.h"
#include "src/optimizer.h"
#include "src/energy.h"

using namespace std;
using namespace bigbang;
using namespace zjucad::matrix;
using mati_t=zjucad::matrix::matrix<size_t>;
using matd_t=zjucad::matrix::matrix<double>;

const size_t SEGS = 12;
const double RHO = 1;
const double H = 0.01;

//#define IMPL_EULER

int main(int argc, char *argv[])
{
  if ( argc != 2 ) {
    cerr << "# usage: " << argv[1] << " output_prefix\n";
    return __LINE__;
  }
  boost::filesystem::create_directory("./mass_spring");

  mati_t line(2, SEGS);
  matd_t nods(3, SEGS+1); {
    line(0, colon()) = colon(0, SEGS-1);
    line(1, colon()) = colon(1, SEGS);
    nods(colon(), 0) = zeros<double>(3, 1);
    nods(0, SEGS) = 1; nods(1, SEGS) = 0; nods(2, SEGS) = 0;
    for (size_t i = 1; i < SEGS; ++i)
      nods(0, i) = nods(0, i-1)+1.0/SEGS;
    ofstream os("./mass_spring/rest.vtk");
    line2vtk(os, &nods[0], nods.size(2), &line[0], line.size(2));
  }

  vector<size_t> fixed_vert{0, SEGS};

  // assemble energies
  vector<shared_ptr<Functional<double>>> ebf(5);
  shared_ptr<Functional<double>> energy;
#ifdef IMPL_EULER
  ebf[0] = make_shared<momentum_potential_imp_euler>(line, nods, RHO, H, 1e0);
#else
  ebf[0] = make_shared<momentum_potential_bdf2>(line, nods, RHO, H, 1e0);
#endif
  ebf[1] = make_shared<spring_potential>(line, nods, 1e2);
  ebf[2] = make_shared<line_bending_potential>(line, nods, 1e-1);
  ebf[3] = make_shared<gravitational_potential>(line, nods, RHO, 1.0);
  ebf[4] = make_shared<positional_potential>(fixed_vert, nods, 1e3);
  try {
    energy = make_shared<energy_t<double>>(ebf);
  } catch ( exception &e ) {
    cerr << e.what() << endl;
    exit(EXIT_FAILURE);
  }

  // give initial value

  // simulate
  char outfile[256];
  for (size_t i = 0; i < 200; ++i) {
    cout << "[info] frame " << i << endl;
    sprintf(outfile, "./mass_spring/%s_%zu.vtk", argv[1], i);
    ofstream os(outfile);
    line2vtk(os, &nods[0], nods.size(2), &line[0], line.size(2));

    newton_solve(&nods[0], nods.size(), energy);
//    lbfgs_solve(&nods[0], nods.size(), energy);
    dynamic_pointer_cast<momentum_potential>(ebf[0])->Update(&nods[0]);
  }

  cout << "[info] all done\n";
  return 0;
}
