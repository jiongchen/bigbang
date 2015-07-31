#ifndef STABLE_FLUIDS_H
#define STABLE_FLUIDS_H

#include <Eigen/Sparse>
#include <boost/property_tree/ptree.hpp>

namespace bigbang {

#define INDEX(i, j, N) ( (i)*(N)+(j) )
#define VALID(i, j, N) ( 0<=(i) && i<(N) && 0<=(j) && (j)<(N) )

class stable_fluid
{
public:
  typedef Eigen::VectorXd vec_t;
  typedef Eigen::MatrixXd mat_t;
  typedef Eigen::SparseMatrix<double> spmat_t;
  stable_fluid();
  stable_fluid(boost::property_tree::ptree &pt);
  int step();
private:
  int vstep();
  int sstep();
  int calc_lap_op(spmat_t &L);
  int calc_div_op(spmat_t &D);
  mat_t u1_, u0_;
  vec_t s1_, s0_;
};

}
#endif
