#include "example_elastic.h"

#include <iostream>
#include <hjlib/math/blas_lapack.h>
#include <zjucad/matrix/lapack.h>
#include <zjucad/matrix/io.h>
#include <zjucad/matrix/itr_matrix.h>

using namespace std;
using namespace zjucad::matrix;

namespace bigbang {

extern "C" {

void green_strain(const double *x, const double *X, double *out);
void green_strain_jac_rest(const double *x, const double *X, double *out);
void green_strain_jac_curr(const double *x, const double *X, double *out);

}

static inline void voigt3(const double *mat, double *vec) {
  vec[0] = mat[0];
  vec[1] = mat[4];
  vec[2] = mat[8];
  vec[3] = mat[7];
  vec[4] = mat[6];
  vec[5] = mat[3];
}

void strain_calc_test() {
  srand(time(NULL));
  cout << "test strain calculation\n";
  
  matd_t rest = rand<double>(3, 4);
  matd_t curr = rand<double>(3, 4);

  matd_t e1(6, 1); {
    matd_t Ds = curr(colon(), colon(1, 3))-curr(colon(), 0)*ones<double>(1, 3);
    matd_t Dm = rest(colon(), colon(1, 3))-rest(colon(), 0)*ones<double>(1, 3);
    inv(Dm);
    matd_t F = Ds*Dm;
    matd_t E = 0.5*(trans(F)*F-eye<double>(3));
    voigt3(&E[0], &e1[0]);
  }
  
  matd_t e2(6, 1);
  green_strain(&curr[0], &rest[0], &e2[0]);
  
  cout << e1 << endl;
  cout << e2 << endl;
}

void numeric_diff_test() {
  srand(time(NULL));
  cout << "numerical differential test...\n";
  
  const double eps = 1e-6;

  matd_t rest = rand<double>(3, 4);
  matd_t curr = rand<double>(3, 4);
  matd_t E(6, 1);
  green_strain(&curr[0], &rest[0], &E[0]);  
  {
    matd_t dE(6, 12);
    green_strain_jac_curr(&curr[0], &rest[0], &dE[0]);

    matd_t dEn(6, 12);
    for (size_t i = 0; i < curr.size(); ++i) {
      matd_t curr_d = curr; curr_d[i] += eps;
      matd_t E_d(6, 1);
      green_strain(&curr_d[0], &rest[0], &E_d[0]);
      dEn(colon(), i) = (E_d-E)/eps;
    }
    cout << "curr diff norm: " << norm(dE-dEn)/norm(dE) << endl;
  }
  {
    matd_t dE(6, 12);
    green_strain_jac_rest(&curr[0], &rest[0], &dE[0]);

    matd_t dEn(6, 12);
    for (size_t i = 0; i < rest.size(); ++i) {
      matd_t rest_d = rest; rest_d[i] += eps;
      matd_t E_d(6, 1);
      green_strain(&curr[0], &rest_d[0], &E_d[0]);
      dEn(colon(), i) = (E_d-E)/eps;
    }
    cout << "rest diff norm: " << norm(dE-dEn)/norm(dE) << endl;
  }
}

}
