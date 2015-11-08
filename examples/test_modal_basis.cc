#include <iostream>
#include <jtflib/mesh/io.h>
#include <zjucad/matrix/itr_matrix.h>
#include <boost/filesystem.hpp>

#include "src/mass_matrix.h"
#include "src/modal_analysis.h"
#include "src/energy.h"
#include "src/vtk.h"

using namespace std;
using namespace zjucad::matrix;
using namespace Eigen;
using namespace bigbang;

static size_t numBase = 60;

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
  boost::filesystem::create_directory(argv[3]);

  matrix<size_t> tets;
  matrix<double> nods;
  jtf::mesh::tet_mesh_read_from_zjumat(argv[1], &nods, &tets);

  vector<size_t> fix;
  read_fixed_verts(argv[2], fix);

  SparseMatrix<double> K, M;
  calc_mass_matrix(tets, nods, 1.0, 3, &M, true);

  shared_ptr<elastic_potential> energy
      = make_shared<elastic_potential>(tets, nods, elastic_potential::LINEAR, 1e3, 0.45, 1.0);
  K.resize(energy->Nx(), energy->Nx()); {
    vector<Triplet<double>> trips;
    energy->Hes(&nods[0], &trips);
    K.reserve(trips.size());
    K.setFromTriplets(trips.begin(), trips.end());
  }

  // fix verts
  unordered_set<size_t> fixDOF;
  for (size_t i = 0; i < fix.size(); ++i) {
    fixDOF.insert(3*fix[i]+0);
    fixDOF.insert(3*fix[i]+1);
    fixDOF.insert(3*fix[i]+2);
  }
  basis_builder solver(M, K, fixDOF);
  MatrixXd U; VectorXd lambda;
  solver.compute(numBase, U, lambda);

  // visualize basis
  cout << "[info] frequency:\n" << lambda.transpose() << endl;
  itr_matrix<const double *> u(U.rows(), U.cols(), U.data());
  char out[256];
  for (size_t i = 0; i < numBase; ++i) {
    cout << "[info] basis " << i << endl;
    sprintf(out, "%s/basis_%zu.vtk", argv[3], i);
    matd_t vert = itr_matrix<const double *>(nods.size(), 1, nods.begin())+u(colon(), i);
    ofstream os(out);
    tet2vtk(os, &vert[0], vert.size()/3, &tets[0], tets.size(2));
  }

  cout << "done\n";
  return 0;
}
