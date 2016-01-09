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
#include <zjucad/matrix/itr_matrix.h>
#include <zjucad/matrix/io.h>
#include <omp.h>
#include <Eigen/Geometry>
#include <unsupported/Eigen/MatrixFunctions>
#include <time.h>

#include "src/mass_matrix.h"
#include "src/config.h"
#include "src/vtk.h"
#include "src/energy.h"
#include "src/optimizer.h"
#include "src/jacobi.h"

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
  mati_t tets;
  matd_t nods;
  jtf::mesh::tet_mesh_read_from_zjumat("../../dat/beam.tet", &nods, &tets);
  ofstream os("./unitest/beam.vtk");
  tet2vtk(os, &nods[0], nods.size(2), &tets[0], tets.size(2));

  cout << "done\n";
  return 0;
}

extern "C" {
  void hex_stvk_(double *val, const double *x, const double *h, const double *lam, const double *miu);
}

int test_hex_elastic(ptree &pt) {
  double h = 1.0;
  double a = 1.0;
  double b = 1.0;
  double value = 0.0;

  const double G[24] = {-1,-1,-1, 1,-1,-1, -1,1,-1, 1,1,-1, -1,-1,1, 1,-1,1, -1,1,1, 1,1,1};
  Map<const Matrix<double, 3, 8>> Ge(G);
  cout << "Ge:\n" << Ge << endl << endl;
  Matrix<double, 3, 8> Gee = 1.0/(4*h)*Ge;

  const double x[24] = {0,0,0, 1,0,0, 0,1,0, 1,1,0, 0,0,1, 1,0,1, 0,1,1, 1,1,1};
  Map<const Matrix<double, 3, 8>> X(x);
  cout << "X:\n" << X << endl << endl;

  Matrix3d F = X*Gee.transpose();
  cout << F << endl << endl;

  Matrix3d E = 0.5*(F.transpose()*F-Matrix3d::Identity());
  double energy = h*h*h*(a*E.squaredNorm()+0.5*b*E.trace()*E.trace());
  cout << "energy value: " << energy << endl;

  hex_stvk_(&value, x, &h, &a, &b);
  cout << "energy value: " << value << endl;

  cout << "\ndone\n";
  return 0;
}

int test_omp_num_threads(ptree &pt) {
#pragma omp parallel for
  for (size_t i = 0; i < 1; ++i) {
    printf("this is %zu\n", i);
  }
  return 0;
}

int test_mat_add_scalar(ptree &pt) {
  matd_t mat = rand<double>(3, 2);
  cout << mat << endl << endl;
  mat += 3.0;
  cout << mat << endl << endl;
  mat(colon(), 0) += 1.0;
  cout << mat << endl << endl;
  return 0;
}

int test_matrix_log(ptree &pt) {
  srand(time(NULL));
  Vector3d axis = Vector3d::Random();
  axis /= axis.norm();
  cout << "axis:\n" << axis << endl << endl;
  double angle = 1.0;
  cout << "angle:\n" << angle << endl << endl;
  Matrix3d rot = AngleAxisd(angle, axis).toRotationMatrix();
  cout << "log(R):\n" << rot.log() << endl << endl;
  return 0;
}

int test_iterative_solve(ptree &pt) {
  MatrixXd A(4, 4);
//  A << 10, -1, 2, 0.0,
//      -1, 11, -1, 3,
//      2, -1, 10, -1,
//      0, 3, -1, 8;
  A << 5, 1, -1, -2,
      2, 8, 1, 3,
      1, -2, -4, -1,
      -1, 3, 2, 7;
  cout << A << endl;
  VectorXd b(4);
//  b << 6, 25, -11, 15;
  b << -2, -6, 6, 12;
  VectorXd x(4);
  x.setZero();
  SparseMatrix<double, RowMajor> AA = A.sparseView();

  for (size_t i = 0; i < 2000; ++i)
    apply_gauss_seidel(AA, b, x);

  cout << x << endl;
  cout << "residual: " << (b-AA*x).lpNorm<Infinity>() << endl;
  return 0;
}

#ifdef USE_CUDA
int test_cuda_jacobi(ptree &pt) {
  srand(time(NULL));
  const size_t dim = pt.get<int>("dim.value");

  MatrixXd A = MatrixXd::Random(dim, dim);
  MatrixXd ATA = A.transpose()*A;
  for (size_t i = 0; i < dim; ++i)
    ATA(i, i) += 3000;
  SparseMatrix<double, RowMajor> M = ATA.sparseView();

  VectorXd b = VectorXd::Random(dim);
  VectorXd x = VectorXd::Random(dim);

  shared_ptr<cuda_jacobi_solver> solver = make_shared<cuda_jacobi_solver>(M);
  time_t start, end;
  start = clock();
  if ( pt.get<int>("method.value") == 0 ) {
    for (size_t i = 0; i < pt.get<int>("maxiter.value"); ++i)
      solver->apply(b, x);
  } else {
    for (size_t i = 0; i < pt.get<int>("maxiter.value"); ++i)
      apply_jacobi(M, b, x);
  }
  end = clock();
  printf("time: %lf\n", (double)(end-start)/CLOCKS_PER_SEC);

  cout << "residual: " << (b-M*x).lpNorm<Infinity>() << endl;
  return 0;
}
#endif

int main(int argc, char *argv[])
{
  ptree pt;
  boost::filesystem::create_directory("./unitest");
  try {
    zjucad::read_cmdline(argc, argv, pt);
    CALL_SUB_PROG(test_kronecker_product);
    CALL_SUB_PROG(test_mass_matrix);
    CALL_SUB_PROG(test_energy);
    CALL_SUB_PROG(test_hex_elastic);
    CALL_SUB_PROG(test_mat_add_scalar);
    CALL_SUB_PROG(test_omp_num_threads);
    CALL_SUB_PROG(test_matrix_log);
    CALL_SUB_PROG(test_iterative_solve);
    CALL_SUB_PROG(test_cuda_jacobi);
  } catch (const boost::property_tree::ptree_error &e) {
    cerr << "Usage: " << endl;
    zjucad::show_usage_info(std::cerr, pt);
  } catch (const std::exception &e) {
    cerr << "# " << e.what() << endl;
  }
  return 0;
}
