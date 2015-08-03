#include "pbd_cloth.h"

#include <jtflib/mesh/mesh.h>

#include "def.h"

using namespace std;
using namespace zjucad::matrix;
using namespace jtf::mesh;
using namespace Eigen;
using mati_t = matrix<size_t>;
using matd_t = matrix<double>;

namespace bigbang {

extern "C" {

void calc_streth_(double *val, const double *x);
void calc_streth_jac_(double *jac, const double *x);

void calc_bend_(double *val, const double *x);
void calc_bend_jac_(double *jac, const double *x);

}

void get_dihedral_elements(const mati_t &tris, mati_t &ele)
{
  unique_ptr<edge2cell_adjacent> ea(edge2cell_adjacent::create(tris, false));
  mati_t bd_ed_id;
  get_boundary_edge_idx(*ea, bd_ed_id);
  ele.resize(4, ea->edges_.size()-bd_ed_id.size());
  for(size_t ei = 0, di = 0; ei < ea->edges_.size(); ++ei) {
    pair<size_t, size_t> nb_tr_id = ea->edge2cell_[ei];
    if( ea->is_boundary_edge(nb_tr_id) ) continue;
    ele(colon(1, 2), di) = ea->get_edge(ei);
    // orient
    bool need_swap = true;
    for(size_t k = 0; k < 3; ++k) {
      if( ele(1, di) == tris(k, nb_tr_id.first) ) {
        if( ele(2, di) != tris((k+1)%3, nb_tr_id.first) )
          need_swap = false;
      }
    }
    if( need_swap )
      swap(ele(1, di), ele(2, di));
    ele(0, di) = zjucad::matrix::sum(tris(colon(), nb_tr_id.first))
        - zjucad::matrix::sum(ele(colon(1, 2), di));
    ele(3, di) = zjucad::matrix::sum(tris(colon(), nb_tr_id.second))
        - zjucad::matrix::sum(ele(colon(1, 2), di));
    ++di;
  }
}

class constraint_piece
{
public:
  virtual ~constraint_piece();
  virtual size_t dim() const;
  virtual int eval_val(const double *x, double *val) const;
  virtual int eval_jac(const double *x, double *jac) const;
  const mati_t pn_;
};

class stretch_cons : public constraint_piece
{
public:
  stretch_cons(const mati_t &edge, const matd_t &nods)
    : pn_(edge), dim_(nods.size()), len_(0.0) {
    matd_t X = nods(colon(), pn_);
    calc_streth_(&len_, &X[0]);
  }
  size_t dim() const {
    return dim_;;
  }
  int eval_val(const double *x, double *val) const {
    matd_t X = itr_matrix<const double*>(3, dim_/3, x)(colon(), pn_);
    double curr_len = 0;
    calc_streth_(&curr_len, &X[0]);
    *val = curr_len-len_;
    return 0;
  }
  int eval_jac(const double *x, double *jac) const {
    matd_t X = itr_matrix<const double*>(3, dim_/3, x)(colon(), pn_);
    calc_streth_jac_(jac, &X[0]);
    return 0;
  }
private:
  const size_t dim_;
  double len_;
};

class bend_cons : public constraint_piece
{
public:
  bend_cons(const mati_t &dias, const matd_t &nods)
    : pn_(dias), dim_(nods.size()), dih_(0.0) {
    matd_t X = nods(colon(), pn_);
    calc_bend_(&dih_, &X[0]);
  }
  size_t dim() const {
    return dim_;
  }
  int eval_val(const double *x, double *val) const {
    matd_t X = itr_matrix<const double*>(3, dim_/3, x)(colon(), pn_);
    double curr_dih = 0;
    calc_bend_(&curr_dih, &X[0]);
    *val = curr_dih-dih_;
    return 0;
  }
  int eval_jac(const double *x, double *jac) const {
    matd_t X = itr_matrix<const double*>(3, dim_/3, x)(colon(), pn_);
    calc_bend_jac_(jac, &X[0]);
    return 0;
  }
private:
  const size_t dim_;
  double dih_;
};

class constraint_collector
{
public:
  enum ConstraintType {

  };
  void operator ()(const mati_t &tris, const matd_t &nods, ConstraintType type) {

  }
  vector<shared_ptr<constraint_piece>> buff_;
private:
  void add_stretch_cons();
  void add_bend_cons();
  void add_collision_cons();
  mati_t edge_;
  mati_t dia_;
};

int pbd_cloth_solver::precompute() {
  return 0;
}

int pbd_cloth_solver::project_constraints(vec_t &x) {
  itr_matrix<double*> X(3, x.rows()/3, x.data());
  for (auto &co : collect_->buff_) {
    double val = 0.0;
    co->eval_val(&X[0], &val);
    matd_t jac = zeros<double>(3, co->pn_.size(2));
    co->eval_jac(&X[0], &jac[0]);
    X(colon(), co->pn) += dx;
  }
  return 0;
}

int pbd_cloth_solver::advance() {
  Map<VectorXd> X(nods_.begin(), nods_.size());
  VectorXd Xstar = X;
  vel += h_*Minv_*fext_;
  // damp vel;
  Xstar += h_*vel_;
  // generate collision constraints

  for (size_t i = 0; i < MAX_ITER; ++i) {
    // project constraints
  }
  vel_ = (Xstar-X)/h_;
  X = Xstar;
  // velocity update
  return 0;
}

}
