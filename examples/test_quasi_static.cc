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
#include "src/geom_util.h"
#include "src/io.h"

using namespace std;
using namespace bigbang;
using namespace zjucad::matrix;
namespace po=boost::program_options;

namespace test_qs {
struct argument {
  string input_mesh;
  string input_cons;
  string output_folder;
  double density;
  size_t total_frame;
  double ws, wb, wg, wp;
};
}

static opt_args optparam = {1, 1e-8, false};

int main(int argc, char *argv[])
{
  po::options_description desc("Available options");
  desc.add_options()
      ("help,h", "produce help message")
      ("input_mesh,i", po::value<string>(), "set the input mesh")
      ("input_cons,c", po::value<string>(), "set the input positional constraints")
      ("output_folder,o", po::value<string>(), "set the output folder")
      ("density,d", po::value<double>()->default_value(1.0), "set the density")
      ("total_frame,n", po::value<size_t>()->default_value(200), "set the frame number")
      ("ws", po::value<double>()->default_value(1e3), "set the stretch weight")
      ("wb", po::value<double>()->default_value(1e-1), "set the bending weight")
      ("wg", po::value<double>()->default_value(1.0), "set the gravity weight")
      ("wp", po::value<double>()->default_value(1e3), "set the position weight")
      ;
  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);
  if ( vm.count("help") ) {
    cout << desc << endl;
    return __LINE__;
  }
  test_qs::argument args = {
    vm["input_mesh"].as<string>(),
    vm["input_cons"].as<string>(),
    vm["output_folder"].as<string>(),
    vm["density"].as<double>(),
    vm["total_frame"].as<size_t>(),
    vm["ws"].as<double>(),
    vm["wb"].as<double>(),
    vm["wg"].as<double>(),
    vm["wp"].as<double>()
  };

  if ( !boost::filesystem::exists(args.output_folder) )
    boost::filesystem::create_directory(args.output_folder);

  // load input
  mati_t tris; matd_t nods;
  jtf::mesh::load_obj(args.input_mesh.c_str(), tris, nods);
  vector<size_t> fixed;
  read_fixed_verts(args.input_cons.c_str(), fixed);

  // extract elements
  mati_t edges, diams;
  get_edge_elem(tris, edges);
  get_diam_elem(tris, diams);

  // assemble energies
  vector<shared_ptr<Functional<double>>> ebf(4);
  shared_ptr<Functional<double>> energy;
  ebf[0] = make_shared<spring_potential>(edges, nods, args.ws);
  ebf[1] = make_shared<surf_bending_potential>(diams, nods, args.wb);
  ebf[2] = make_shared<gravitational_potential>(tris, nods, args.density, args.wg);
  ebf[3] = make_shared<positional_potential>(nods, args.wp);
  try {
    energy = make_shared<energy_t<double>>(ebf);
  } catch ( exception &e ) {
    cerr << e.what() << endl;
    exit(EXIT_FAILURE);
  }

  // initial boudary conditions
  for (auto &id : fixed) {
    dynamic_pointer_cast<positional_potential>(ebf[3])->Pin(id, &nods(0, id));
  }

  char outfile[256];
  for (size_t i = 0; i < args.total_frame; ++i) {
    cout << "[info] frame " << i << endl;
    sprintf(outfile, "%s/iter_%zu.vtk", args.output_folder.c_str(), i);
    ofstream os(outfile);
    tri2vtk(os, &nods[0], nods.size(2), &tris[0], tris.size(2));

    newton_solve(&nods[0], nods.size(), energy, optparam);
  }

  cout << "[info] all done\n";
  return 0;
}
