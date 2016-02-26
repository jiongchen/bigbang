#include <iostream>
#include <fstream>
#include <jtflib/mesh/io.h>

#include <boost/filesystem.hpp>
#include <boost/program_options/options_description.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/variables_map.hpp>

#include "src/vtk.h"
#include "src/optimizer.h"
#include "src/energy.h"
#include "src/io.h"

using namespace std;
using namespace bigbang;
using namespace zjucad::matrix;
using mati_t=zjucad::matrix::matrix<size_t>;
using matd_t=zjucad::matrix::matrix<double>;
namespace po=boost::program_options;

namespace test_ses {
struct argument {
  string input_mesh;
  string input_cons;
  string output_folder;
  double density;
  double timestep;
  size_t total_frame;
  double young_modulus;
  double poisson_ratio;
  double we;
  double wg;
  double wp;
};
}

#define APPLY_FORCE(frame, id, f)                                       \
  if ( i == frame )                                                     \
    dynamic_pointer_cast<ext_force_energy>(ebf[4])->ApplyForce(id, f);

#define REMOVE_FORCE(frame, id)                                         \
  if ( i == frame )                                                     \
    dynamic_pointer_cast<ext_force_energy>(ebf[4])->RemoveForce(id);

static opt_args optparam = {10000, 1e-8, false};

int main(int argc, char *argv[])
{
  po::options_description desc("Available options");
  desc.add_options()
      ("help,h", "produce help message")
      ("input_mesh,i", po::value<string>(), "set the input mesh")
      ("input_cons,c", po::value<string>(), "set the input positional constraints")
      ("output_folder,o", po::value<string>(), "set the output folder")
      ("density,d", po::value<double>()->default_value(1.0), "set the density")
      ("timestep,t", po::value<double>()->default_value(0.01), "set the timestep")
      ("total_frame,n", po::value<size_t>()->default_value(200), "set the frame number")
      ("young_modulus,y", po::value<double>()->default_value(1e4), "set the young's modulus")
      ("poisson_ratio,p", po::value<double>()->default_value(0.45), "set poisson ratio")
      ("we", po::value<double>()->default_value(1.0), "set the number of input files to read")
      ("wg", po::value<double>()->default_value(1.0), "set the gravity weight")
      ("wp", po::value<double>()->default_value(1e3), "set the weight of position penalty")
      ;
  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);
  if ( vm.count("help") ) {
    cout << desc << endl;
    return __LINE__;
  }
  test_ses::argument args = {
    vm["input_mesh"].as<string>(),
    vm["input_cons"].as<string>(),
    vm["output_folder"].as<string>(),
    vm["density"].as<double>(),
    vm["timestep"].as<double>(),
    vm["total_frame"].as<size_t>(),
    vm["young_modulus"].as<double>(),
    vm["poisson_ratio"].as<double>(),
    vm["we"].as<double>(),
    vm["wg"].as<double>(),
    vm["wp"].as<double>()
  };

  if ( !boost::filesystem::exists(args.output_folder) )
    boost::filesystem::create_directory(args.output_folder);

  mati_t tets;
  matd_t nods;
  jtf::mesh::tet_mesh_read_from_zjumat(args.input_mesh.c_str(), &nods, &tets); {
    char path[256];
    sprintf(path, "%s/rest.vtk", args.output_folder.c_str());
    ofstream os(path);
    tet2vtk(os, &nods[0], nods.size(2), &tets[0], tets.size(2));
  }
  vector<size_t> fixed;
  read_fixed_verts(args.input_cons.c_str(), fixed);

  // assemble energies
  vector<shared_ptr<Functional<double>>> ebf(5);
  shared_ptr<Functional<double>> energy;
  ebf[0] = make_shared<momentum_potential_imp_euler>(tets, nods, args.density, args.timestep, 1e0);
  ebf[1] = make_shared<elastic_potential>(tets, nods, elastic_potential::COROTATIONAL, args.young_modulus, args.poisson_ratio, args.we);
  ebf[2] = make_shared<gravitational_potential>(tets, nods, args.density, args.wg);
  ebf[3] = make_shared<positional_potential>(nods, args.wp);
  ebf[4] = make_shared<ext_force_energy>(nods, 1e0);
  try {
    energy = make_shared<energy_t<double>>(ebf);
  } catch ( exception &e ) {
    cerr << e.what() << endl;
    exit(EXIT_FAILURE);
  }
  // boudary conditions
  for (auto &elem : fixed)
    dynamic_pointer_cast<positional_potential>(ebf[3])->Pin(elem, &nods(0, elem));

  const vector<size_t> driver{149, 150, 151, 152,
                             153, 154, 155, 156, 157, 158, 159, 160,
                             161, 162, 163, 164, 165, 166, 167};
  const double intensity = 35;

  char outfile[256];
  for (size_t i = 0; i < args.total_frame; ++i) {
    cout << "[info] frame " << i << endl;
    sprintf(outfile, "%s/frame_%zu.vtk", args.output_folder.c_str(), i);
    ofstream os(outfile);
    tet2vtk(os, &nods[0], nods.size(2), &tets[0], tets.size(2));

    matd_t n = cross(nods(colon(), 167)-nods(colon(), 166), nods(colon(), 161)-nods(colon(), 167));
    matd_t o = (nods(colon(), 167)+nods(colon(), 166)+nods(colon(), 161))*ones<double>(3, 1)/3.0;
    // apply twist
    if ( i < 80 ) {
      for (auto &pi : driver) {
        matd_t force = cross(n, nods(colon(), pi)-o);
        force = intensity*force/norm(force);
        APPLY_FORCE(i, pi, &force[0]);
      }
    }
    // release twist
    for (auto &pi : driver) {
      REMOVE_FORCE(80, pi);
    }

    newton_solve(&nods[0], nods.size(), energy, optparam);
    dynamic_pointer_cast<momentum_potential>(ebf[0])->Update(&nods[0]);
  }

  cout << "[info] all done\n";
  return 0;
}
