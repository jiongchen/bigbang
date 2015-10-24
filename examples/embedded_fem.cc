#include <iostream>
#include <fstream>
#include <zjucad/matrix/io.h>

#include <boost/filesystem.hpp>
#include <boost/program_options/options_description.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/variables_map.hpp>

#include "src/energy.h"
#include "src/io.h"
#include "src/vtk.h"
#include "src/optimizer.h"

using namespace std;
using namespace zjucad::matrix;
using namespace bigbang;
namespace po=boost::program_options;

struct argument {
  string input_mesh;
  string input_cons;
  string output_folder;
  size_t frame_num;
  double density;
  double timestep;
  double young_modulus;
  double poisson_ratio;
};

static int read_fixed_verts(const char *filename, vector<size_t> &fixed) {
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

int main(int argc, char *argv[])
{
  po::options_description desc("Available options");
  desc.add_options()
      ("help,h", "produce help message")
      ("input_mesh,i", po::value<string>(), "set the input mesh")
      ("input_cons,c", po::value<string>(), "set the constraints")
      ("output_folder,o", po::value<string>(), "set the output folder")
      ("frames,n", po::value<size_t>()->default_value(100), "set the total frame number")
      ("density,d", po::value<double>()->default_value(1.0), "set the density")
      ("timestep,t", po::value<double>()->default_value(0.01), "set the timestep")
      ("young,E", po::value<double>()->default_value(1e4), "set the Young's modulus")
      ("poisson,v", po::value<double>()->default_value(0.45), "set the poisson ratio")
      ;
  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);
  if ( vm.count("help") ) {
    cout << desc << endl;
    return 1;
  }
  argument args = {
    vm["input_mesh"].as<string>(),
    vm["input_cons"].as<string>(),
    vm["output_folder"].as<string>(),
    vm["frames"].as<size_t>(),
    vm["density"].as<double>(),
    vm["timestep"].as<double>(),
    vm["young"].as<double>(),
    vm["poisson"].as<double>()
  };

  if ( !boost::filesystem::exists(args.output_folder) )
    boost::filesystem::create_directory(args.output_folder);

  mati_t hexs;
  matd_t nods;
  hex_mesh_read_from_vtk(args.input_mesh.c_str(), &nods, &hexs); {
    char path[256];
    sprintf(path, "%s/rest.vtk", args.output_folder.c_str());
    ofstream os(path);
    hex2vtk(os, &nods[0], nods.size(2), &hexs[0], hexs.size(2));
  }
  vector<size_t> fixed;
  read_fixed_verts(args.input_cons.c_str(), fixed);

  vector<shared_ptr<Functional<double>>> ebf(4);
  shared_ptr<Functional<double>> energy;
  ebf[0] = make_shared<momentum_potential_imp_euler>(hexs, nods, args.density, args.timestep, 1e0);
  ebf[1] = make_shared<voxel_elastic_potential>
      (hexs, nods, voxel_elastic_potential::STVK, args.young_modulus, args.poisson_ratio, 1e0);
  ebf[2] = make_shared<gravitational_potential>(hexs, nods, args.density, 1e0);
  ebf[3] = make_shared<positional_potential>(fixed, nods, 1e3);
  try {
    energy = make_shared<energy_t<double>>(ebf);
  } catch ( exception &e ) {
    cerr << e.what() << endl;
    exit(EXIT_FAILURE);
  }

  char outfile[256];
  for (size_t i = 0; i < args.frame_num; ++i) {
    cout << "[info] frame " << i << endl;
    sprintf(outfile, "%s/frame_%zu.vtk", args.output_folder.c_str(), i);
    ofstream os(outfile);
    hex2vtk(os, &nods[0], nods.size(2), &hexs[0], hexs.size(2));

    newton_solve(&nods[0], nods.size(), energy);
    dynamic_pointer_cast<momentum_potential>(ebf[0])->Update(&nods[0]);
  }

  cout << "[info] all done\n";
  return 0;
}
