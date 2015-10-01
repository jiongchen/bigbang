#include <iostream>
#include <zjucad/ptree/ptree.h>
#include <boost/filesystem.hpp>
#include <Eigen/Sparse>
#include <unsupported/Eigen/KroneckerProduct>
#include <jtflib/mesh/io.h>
#include <zjucad/matrix/io.h>
#include <unordered_set>
#include <hjlib/math/blas_lapack.h>
#include <zjucad/matrix/lapack.h>

#include "src/mass_matrix.h"
#include "src/config.h"
#include "src/vtk.h"
#include "src/energy.h"

using namespace std;
using namespace Eigen;
using namespace bigbang;
using namespace zjucad::matrix;
using boost::property_tree::ptree;
using mati_t = zjucad::matrix::matrix<size_t>;
using matd_t = zjucad::matrix::matrix<double>;

int test_kronecker_product(ptree &pt) {
  SparseMatrix<double> A(3, 3);
  A.setIdentity();
  SparseMatrix<double> I(3, 3);
  I.setIdentity();
  cout << A << endl << endl;
  SparseMatrix<double> B;
  kroneckerProduct(A, I).evalTo(B);
  cout << B << endl;
  return 0;
}

int test_mass_matrix(ptree &pt) {
  srand(time(NULL));
  {
    cout << "==================================================\n";
    mati_t line = colon(0, 1);
    matd_t nods = rand<double>(3, 2);
    matd_t edge = nods(colon(), 1)-nods(colon(), 0);
    double len = norm(edge);
    cout << "[info] length: " << len << endl;
    SparseMatrix<double> M;
    calc_mass_matrix(line, nods, 1.0/len, 1, &M, false);
    cout << M << endl;
    calc_mass_matrix(line, nods, 1.0/len, 1, &M, true);
    cout << M << endl;
  }
  {
    cout << "==================================================\n";
    mati_t tris = colon(0, 2);
    matd_t nods = rand<double>(3, 3);
    matd_t edge = nods(colon(), colon(0, 1))-nods(colon(), 2)*ones<double>(1, 2);
    double area = 0.5*norm(cross(edge(colon(), 0), edge(colon(), 1)));
    cout << "[info] area: " << area << endl;
    SparseMatrix<double> M;
    calc_mass_matrix(tris, nods, 1.0/area, 1, &M, false);
    cout << M << endl;
    calc_mass_matrix(tris, nods, 1.0/area, 1, &M, true);
    cout << M << endl;
  }
  {
    cout << "==================================================\n";
    mati_t tets = colon(0, 3);
    matd_t nods = rand<double>(3, 4);
    matd_t edge = nods(colon(), colon(0, 2))-nods(colon(), 3)*ones<double>(1, 3);
    double vol = fabs(det(edge))/6.0;
    cout << "[info] volume: " << vol << endl;
    SparseMatrix<double> M;
    calc_mass_matrix(tets, nods, 1.0/vol, 1, &M, false);
    cout << M << endl;
    calc_mass_matrix(tets, nods, 1.0/vol, 1, &M, true);
    cout << M << endl;
  }
  {
    cout << "==================================================\n";
    mati_t tets;
    matd_t nods;
    jtf::mesh::tet_mesh_read_from_zjumat("../../dat/cube.tet", &nods, &tets);
    ofstream os("./unitest/cube.vtk");
    tet2vtk(os, nods.begin(), nods.size(2), tets.begin(), tets.size(2));
    SparseMatrix<double> M;
    calc_mass_matrix(tets, nods, 1.0, 1, &M, true);
    cout << "[info] size: " << M.rows() << endl;
    cout << M << endl;
  }
  cout << "[info] done\n";
  return 0;
}

int test_energy(ptree &pt) {
  shared_ptr<momentum_potential_imp_euler> sol;
  sol->Nx();
  return 0;
}

int main(int argc, char *argv[])
{
  ptree pt;
  boost::filesystem::create_directory("./unitest");
  try {
    zjucad::read_cmdline(argc, argv, pt);
    CALL_SUB_PROG(test_kronecker_product);
    CALL_SUB_PROG(test_mass_matrix);
  } catch (const boost::property_tree::ptree_error &e) {
    cerr << "Usage: " << endl;
    zjucad::show_usage_info(std::cerr, pt);
  } catch (const std::exception &e) {
    cerr << "# " << e.what() << endl;
  }
  return 0;
}
