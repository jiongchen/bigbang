#include "inhomogeneous_elastic.h"

#include "def.h"

namespace bigbang {

extern "C" {
void linear_tet_energy_(double *val, const double *X, const double *D, const double *vol);
}

class linear_elasticity : public Functional<double>
{

};

}
