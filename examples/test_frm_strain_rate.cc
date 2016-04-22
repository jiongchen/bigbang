#include <iostream>
#include <fstream>
#include <Eigen/Eigen>

using namespace std;
using namespace Eigen;

// x---q---x---q---x---q---x---q---x

Matrix3Xd x;      // point
Matrix3Xd frm;  // frame
Matrix4Xd q;      // quaternion
Vector3d a_u;    // analytic strain rate
Matrix3Xd n_u;  // numeric strain rate

/// given a helix -> r(t) = (acos t, asin t, bt)
/// parameterizing by arclength
/// r(s) = (a*cos s/sqrt(a^2+b^2), a*sin s/sqrt(a^2+b^2), b*s/sqrt(a^2+b^2))
/// s = t*sqrt(a^2+b^2)
size_t N;
double a = 0.1;
double b = 0.2;
double T = 6*M_PI;

void init_helix() {
  x.resize(NoChange, N);
  for (size_t i = 0; i < x.cols(); ++i) {
    double t = T/(N-1)*i;
    x(0, i) = a*cos(t);
    x(1, i) = a*sin(t);
    x(2, i) = b*t;
  }
}

/// return frenet frame [N, B, T]
/// T = dr/ds
/// N = \frac{dT/ds}{\|dT/ds\|}
/// B = T x N
void calc_frenet_frame() {
  frm.resize(NoChange, 3*(x.cols()-1));
  double vel = sqrt(a*a+b*b);
  for (size_t i = 0; i < x.cols()-1; ++i) {
    double t = T/(N-1)*(i+0.5);
    double s = vel*t;
    frm.col(3*i+2) = Vector3d(-a*sin(s/vel)/vel, a*cos(s/vel)/vel, b/vel);
    frm.col(3*i+0) = Vector3d(-a*cos(s/vel)/(a*a+b*b), -a*sin(s/vel)/(a*a+b*b), 0).normalized();
    frm.col(3*i+1) = frm.col(3*i+2).cross(frm.col(3*i+0));
  }
}

void convert_frame_to_quaternion() {
  cout << "[Info] Frenet frame in quaternion:\n\n";
  q.resize(NoChange, x.cols()-1);
  for (size_t i = 0; i < q.cols(); ++i) {
    q.col(i) = Quaterniond(frm.block<3, 3>(0, 3*i)).coeffs();
    if ( i >= 1 ) {
      double cosQ = q.col(i-1).dot(q.col(i));
      q.col(i) *= cosQ >= 0 ? 1 : -1;
    }
    printf("%4zu: %16lf %16lf %16lf %16lf\n", i, q(0, i), q(1, i), q(2, i), q(3, i));
  }
  cout << endl;
}

/// Darboux vector u = \kappa*B + \tau*T
/// and u_k = u . d_k
void calc_analytic_strain_rate() {
  const double kappa = a/(a*a+b*b);
  const double tau = b/(b*b+a*a);
  a_u = Vector3d(0, kappa, tau);
  cout << "[Info] curvature of helix: " << kappa << endl << endl;
  cout << "[Info] torsion of helix: " << tau << endl << endl;
  cout << "[Info] Value of real $u_k$: [" << a_u.transpose() << "]\n\n";
}

// x_{j}---q_{j}---x_{j+1}---q_{j+1}---x_{j+2} -> u_k
void calc_numeric_strain_rate() {
  n_u.resize(3, x.cols()-2);
  Matrix4d B0, B1, B2;
  B0 << 0, 0, 0, 1,
      0, 0, 1, 0,
      0, -1, 0, 0,
      -1, 0, 0, 0;
  B1 << 0, 0, -1, 0,
      0, 0, 0, 1,
      1, 0, 0, 0,
      0, -1, 0, 0;
  B2 << 0, 1, 0, 0,
      -1, 0, 0, 0,
      0, 0, 0, 1,
      0, 0, -1, 0;
  for (size_t i = 0; i < n_u.cols(); ++i) {
    double len = 0.5*((x.col(i)-x.col(i+1)).norm()+(x.col(i+1)-x.col(i+2)).norm());
    n_u(0, i) = (B0*(q.col(i)+q.col(i+1))).dot(q.col(i+1)-q.col(i))/len;
    n_u(1, i) = (B1*(q.col(i)+q.col(i+1))).dot(q.col(i+1)-q.col(i))/len;
    n_u(2, i) = (B2*(q.col(i)+q.col(i+1))).dot(q.col(i+1)-q.col(i))/len;
  }
}

int main(int argc, char *argv[])
{
  if ( argc != 2 ) {
    cerr << "#usage: ./test number_of_vertices\n";
    return __LINE__;
  }
  N = atoi(argv[1]);
  if ( N < 3 )
    exit(EXIT_FAILURE);

  cout << "[Info] Vertices number: " << N << endl << endl;
  init_helix();
  calc_frenet_frame();
  convert_frame_to_quaternion();
  calc_analytic_strain_rate();
  calc_numeric_strain_rate();

  cout << "[Info] Approximations of $u_k$:\n\n";
  for (size_t i = 0; i < n_u.cols(); ++i) {
    printf("%4zu: %16lf %16lf %16lf\n", i, n_u(0, i), n_u(1, i), n_u(2, i));
  }
  cout << "\n[Info] Done\n";
  return 0;
}
