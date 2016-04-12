#include "cosserat.h"

using namespace std;
using namespace Eigen;

namespace bigbang {

extern "C" {

void rod_stretch_(double *val, const double *x, const double *d, const double *Es, const double *r);
void rod_stretch_jac_(double *jac, const double *x, const double *d, const double *Es, const double *r);
void rod_stretch_hes_(double *hes, const double *x, const double *d, const double *Es, const double *r);

void rod_bend_(double *val, const double *q, const double *u, const double *d, const double *E, const double *G, const double *r);
void rod_bend_jac_(double *jac, const double *q, const double *u, const double *d, const double *E, const double *G, const double *r);
void rod_bend_hes_(double *hes, const double *q, const double *u, const double *d, const double *E, const double *G, const double *r);

void rod_couple_(double *val, const double *xq, const double *d, const double *kappa);
void rod_couple_jac_(double *jac, const double *xq, const double *d, const double *kappa);
void rod_couple_hes_(double *hes, const double *xq, const double *d, const double *kappa);

}
class line_kinetic_energy;
class angular_kinetic_energy;
class rod_stretch_energy;
class rod_bend_energy;
class coupling_energy;

}
