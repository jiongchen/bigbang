#include <iostream>
#include <fstream>
#include <jtflib/mesh/io.h>

#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>

#include "src/vtk.h"
#include "src/optimizer.h"
#include "src/energy.h"
#include "src/io.h"
#include "src/geom_util.h"

using namespace std;
using namespace bigbang;
using namespace zjucad::matrix;
using namespace Eigen;
using mati_t=zjucad::matrix::matrix<size_t>;
using matd_t=zjucad::matrix::matrix<double>;
namespace po=boost::program_options;

#define APPLY_FORCE(frame, id, f)                                     \
  if ( i == frame )                                                   \
    dynamic_pointer_cast<ext_force_energy>(tbf[4])->ApplyForce(id, f);

#define REMOVE_FORCE(frame, id)                                       \
  if ( i == frame )                                                   \
    dynamic_pointer_cast<ext_force_energy>(tbf[4])->RemoveForce(id);

#define RELEASE_VERT(frame, id)                                       \
  if ( i == frame )                                                   \
    dynamic_pointer_cast<positional_potential>(tbf[3])->Release(id);

namespace test_emb_thin_shell {
struct argument {
  string input_mesh;
  string input_cons;
  string output_folder;
  // for tet
  double density;
  double timestep;
  size_t total_frame;
  double young_modulus;
  double poisson_ratio;
  double we;
  double wg;
  double wp;
  // for shell
  double ws;
  double wb;
  double wf;
};
}

static opt_args optparam = {10000, 1e-8, true};

int main(int argc, char *argv[])
{
  po::options_description desc("Available options");
  desc.add_options()
      ("help,h", "produce help message")
      ("input_mesh,i",    po::value<string>(), "set the input mesh")
      ("input_cons,c",    po::value<string>(), "set the input positional constraints")
      ("output_folder,o", po::value<string>(), "set the output folder")
      ("density,d",       po::value<double>()->default_value(1.0), "set the density")
      ("timestep,t",      po::value<double>()->default_value(0.01), "set the timestep")
      ("total_frame,n",   po::value<size_t>()->default_value(200), "set the frame number")
      ("young_modulus,y", po::value<double>()->default_value(1e4), "set the young's modulus")
      ("poisson_ratio,p", po::value<double>()->default_value(0.45), "set poisson ratio")
      ("we", po::value<double>()->default_value(1.0), "set weight for elasticity")
      ("wg", po::value<double>()->default_value(1.0), "set weight for gravity")
      ("wp", po::value<double>()->default_value(1e3), "set weight for position penalty")
      ("ws", po::value<double>()->default_value(1e3), "set weight for stretch")
      ("wb", po::value<double>()->default_value(1e0), "set weight for bending")
      ("wf", po::value<double>()->default_value(1e3), "set weight for filter")
      ;
  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);
  if ( vm.count("help") ) {
    cout << desc << endl;
    return __LINE__;
  }
  test_emb_thin_shell::argument args = {
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
    vm["wp"].as<double>(),
    vm["ws"].as<double>(),
    vm["wb"].as<double>(),
    vm["wf"].as<double>()
  };
  if ( !boost::filesystem::exists(args.output_folder) )
    boost::filesystem::create_directory(args.output_folder);

  // read interior tetrahedral mesh
  mati_t tets; matd_t nods;
  jtf::mesh::tet_mesh_read_from_zjumat(args.input_mesh.c_str(), &nods, &tets);
  vector<size_t> fixed;
  read_fixed_verts(args.input_cons.c_str(), fixed);

  // gernerate surface triangle mesh
  mati_t tris; matd_t snods;
  extract_tet_surface(tets, nods, tris, snods);
  shrink_surface(tris, snods, 0.005); {
    char outfile[256];
    sprintf(outfile, "%s/shell.obj", args.output_folder.c_str());
    jtf::mesh::save_obj(outfile, tris, snods);
  }
  mati_t edges, diams;
  get_edge_elem(tris, edges);
  get_diam_elem(tris, diams);

  // build the interpolation between shell and tet
  SparseMatrix<double> B;
  interp_pts_in_tets(nods, tets, snods, B);

  // ASSEMBLE TET ENERGY
  vector<shared_ptr<Functional<double>>> tbf(5);
  shared_ptr<Functional<double>> energy;
  tbf[0] = make_shared<momentum_potential_imp_euler>(tets, nods, args.density, args.timestep, 1e0);
  tbf[1] = make_shared<elastic_potential>(tets, nods, elastic_potential::STVK, args.young_modulus, args.poisson_ratio, args.we);
  tbf[2] = make_shared<gravitational_potential>(tets, nods, args.density, args.wg);
  tbf[3] = make_shared<positional_potential>(nods, args.wp);
  tbf[4] = make_shared<ext_force_energy>(nods, 1.0);
  try {
    energy = make_shared<energy_t<double>>(tbf);
  } catch ( exception &e ) {
    cerr << e.what() << endl;
    exit(EXIT_FAILURE);
  }
  // boudary simulation conditions
  for (auto &elem : fixed)
    dynamic_pointer_cast<positional_potential>(tbf[3])->Pin(elem, &nods(0, elem));

  // ASSEMBLE SHELL ENERGY
  vector<shared_ptr<Functional<double>>> sbf(3);
  shared_ptr<Functional<double>> shell_energy;
  sbf[0] = make_shared<spring_potential>(edges, snods, args.ws);
  sbf[1] = make_shared<isometric_bending>(diams, snods, args.wb);
//  sbf[2] = make_shared<low_pass_filter_energy>();
  try {
    shell_energy = make_shared<energy_t<double>>(sbf);
  } catch ( exception &e ) {
    cerr << e.what() << endl;
    exit(EXIT_FAILURE);
  }

  char outfile[256];
  double force[3] = {0, 0, -25};
  for (size_t i = 0; i < args.total_frame; ++i) {
    cout << "[INFO] frame " << i << endl;
    // interior dynamics
    {
      sprintf(outfile, "%s/tet_frame_%zu.vtk", args.output_folder.c_str(), i);
      ofstream os(outfile);
      tet2vtk(os, &nods[0], nods.size(2), &tets[0], tets.size(2));
    }
    // some operation here
    for (size_t pi = 0; pi <= 482; ++pi) {
      APPLY_FORCE(0, pi, force);
    }
    for (size_t pi = 0; pi <= 482; ++pi) {
      REMOVE_FORCE(50, pi);
    }
    newton_solve(&nods[0], nods.size(), energy, optparam);
    dynamic_pointer_cast<momentum_potential>(tbf[0])->Update(&nods[0]);

    // exterior statics
    {
      sprintf(outfile, "%s/shell_frame_%zu.vtk", args.output_folder.c_str(), i);
      ofstream os(outfile);
      tri2vtk(os, &snods[0], snods.size(2), &tris[0], tris.size(2));
    }
    Map<MatrixXd>(&snods[0], 3, snods.size(2)) = Map<const MatrixXd>(&nods[0], 3, nods.size(2))*B;
//    newton_solve(&snods[0], snods.size(), shell_energy, optparam);
  }

  cout << "[INFO] all done\n";
  return 0;
}
