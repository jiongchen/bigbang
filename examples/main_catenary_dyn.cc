#include <iostream>
#include <fstream>
#include <Eigen/CholmodSupport>
#include <boost/program_options.hpp>

#include "src/catenary.h"
#include "src/config.h"

using namespace std;
using namespace bigbang;
using namespace Eigen;
namespace po=boost::program_options;

static shared_ptr<catenary> g_ins;
static double  g_h;
static pfunc_t g_energy;
static size_t  g_max_iter;
static double  g_tolerance;

static void advance() {
  const size_t dim = g_ins->pos.size();
  VectorXd xstar = g_ins->pos, dx = VectorXd::Zero(dim);
  
  CholmodSimplicialLDLT<SparseMatrix<double>> solver; 
  for (size_t iter = 0; iter < g_max_iter; ++iter) {
    double value = 0; {
      g_energy->Val(&xstar[0], &value);
      if ( iter % 100 == 0 )
        cout << "\t@potential value: " << value << endl;
    }
    VectorXd g = VectorXd::Zero(dim); {
      g_energy->Gra(&xstar[0], g.data());
    }
    SparseMatrix<double> H(dim, dim); {
      vector<Triplet<double>> trips;
      g_energy->Hes(&xstar[0], &trips);
      H.reserve(trips.size());
      H.setFromTriplets(trips.begin(), trips.end());
    }
    SparseMatrix<double> LHS = g_ins->Mass+g_h*g_h*H;
    VectorXd rhs = -g_ins->Mass*(xstar-g_ins->pos-g_h*g_ins->vel)-g_h*g_h*g;
    solver.compute(LHS);
    ASSERT(solver.info() == Eigen::Success);
    dx = solver.solve(rhs);
    ASSERT(solver.info() == Eigen::Success);
    const double xnorm = xstar.norm();
    xstar += dx;
    if ( dx.norm() <= g_tolerance*xnorm ) {
      cout << "\t@converged after " << iter+1 << " iterations\n";
      break;
    }
  }
  g_ins->vel = (xstar-g_ins->pos)/g_h;
  g_ins->pos = xstar;
}

int main(int argc, char *argv[])
{
  po::options_description desc("Available options"); {
    desc.add_options()
        ("help,h", "help message")
        ("length",   po::value<double>(), "set length")
        ("vert_num", po::value<size_t>(), "set vertices number")
        ("density",  po::value<double>(), "set density")
        ("timestep", po::value<double>(), "timestep")
        ("out_dir",  po::value<string>(), "output folder")
        ("frames",   po::value<size_t>()->default_value(100),  "frames")
        ("ws",       po::value<double>()->default_value(1e3),  "strain stiffness")
        ("wb",       po::value<double>()->default_value(1e-1), "bending stiffness")
        ("wg",       po::value<double>()->default_value(1e0),  "gravity stiffness")
        ("wp",       po::value<double>()->default_value(1e3),  "handle stiffness")
        ("max_iter", po::value<size_t>()->default_value(1000), "maximum iterations")
        ("tolerance",po::value<double>()->default_value(1e-8), "converged tolerance")
        ;
  }
  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);
  if ( vm.count("help") ) {
    cout << desc << endl;
    return __LINE__;
  }

  const double length = vm["length"].as<double>();
  const size_t vert_num = vm["vert_num"].as<size_t>();
  const double dens = vm["density"].as<double>();
  g_ins.reset(create_catenary(length, vert_num, dens));
  if ( !g_ins.get() ) {
    cerr << "[Error] fail to create a catenary\n";
    return __LINE__;
  }

  cout << "[Info] assemble potentials...\n";
  vector<pfunc_t> buffer(4);
  buffer[0] = make_shared<catenary_strain>(g_ins.get(), vm["ws"].as<double>());
  buffer[1] = make_shared<catenary_bending>(g_ins.get(), vm["wb"].as<double>());
  buffer[2] = make_shared<catenary_grav>(g_ins.get(), vm["wg"].as<double>());
  buffer[3] = make_shared<catenary_handle>(g_ins.get(), vm["wp"].as<double>());
  try {
    g_energy = make_shared<energy_t<double>>(buffer);
  } catch ( exception &e ) {
    cerr << "[Excp] " << e.what() << endl;
    exit(EXIT_FAILURE);
  }

  double timer = 0;
  g_h         =  vm["timestep"].as<double>();
  g_max_iter  =  vm["max_iter"].as<size_t>();
  g_tolerance =  vm["tolerance"].as<double>();

  // handles
  {
    const size_t idx = 0;
    const Vector3d x0 = g_ins->pos.segment<3>(3*idx);
    shared_ptr<handle_move> han =
        make_shared<constant_move>(x0, timer, timer);
    dynamic_pointer_cast<catenary_handle>(buffer[3])->PinDown(idx, han);
  }
  {
    const size_t idx = g_ins->vert_num-1;
    const Vector3d x0 = g_ins->pos.segment<3>(3*idx);
    shared_ptr<handle_move> han =
        make_shared<vertical_sine_move>(x0, timer, timer, M_PI/16.0/g_h, 0.2);
    dynamic_pointer_cast<catenary_handle>(buffer[3])->PinDown(idx, han);
  }
  
  char outfile[256];
  for (size_t i = 0; i < vm["frames"].as<size_t>(); ++i) {
    cout << "[Info] frame " << i << endl;
    sprintf(outfile, "%s/frame_%04zu.vtk", vm["out_dir"].as<string>().c_str(), i);
    dump_catenary(outfile, g_ins.get());

    advance();
    timer += g_h;
  }
  
  cout << "[Info] done\n";
  return 0;
}
