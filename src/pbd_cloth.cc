#include "pbd_cloth.h"

#include <jtflib/mesh/mesh.h>
#include <jtflib/mesh/io.h>
#include <zjucad/matrix/io.h>

#include "def.h"
#include "vtk.h"
#include "mass_matrix.h"

using namespace std;
using namespace zjucad::matrix;
using namespace jtf::mesh;
using namespace Eigen;
using mati_t = matrix<size_t>;
using matd_t = matrix<double>;

namespace bigbang {

extern "C" {

void calc_stretch_(double *val, const double *x);
void calc_stretch_jac_(double *jac, const double *x);

void calc_bend_(double *val, const double *x);
void calc_bend_jac_(double *jac, const double *x);

}

void get_edge_elements(const mati_t &tris, mati_t &ele) {
  unique_ptr<edge2cell_adjacent> ea(edge2cell_adjacent::create(tris, false));
  ele.resize(2, ea->edges_.size());
#pragma omp parallel for
  for (size_t i = 0; i < ele.size(2); ++i) {
    ele(0, i) = ea->edges_[i].first;
    ele(1, i) = ea->edges_[i].second;
  }
}

void get_dihedral_elements(const mati_t &tris, mati_t &ele) {
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
  enum cons_type {
    EQUAL,
    GREATER
  };
  constraint_piece(const mati_t &pn, const double k, const cons_type type) : pn_(pn), k_(k), type_(type) {}
  virtual ~constraint_piece() {}
  virtual size_t dim() const = 0;
  virtual int eval_val(const double *x, double *val) const = 0;
  virtual int eval_jac(const double *x, double *jac) const = 0;
  const mati_t pn_;
  const double k_;
  const cons_type type_;
};

class stretch_cons : public constraint_piece
{
public:
  stretch_cons(const mati_t &edge, const matd_t &nods, const double k, const cons_type type=EQUAL)
    : dim_(nods.size()), len_(0.0), constraint_piece(edge, k, type) {
    matd_t X = nods(colon(), pn_);
    calc_stretch_(&len_, &X[0]);
  }
  size_t dim() const {
    return dim_;
  }
  int eval_val(const double *x, double *val) const {
    matd_t X = itr_matrix<const double*>(3, dim_/3, x)(colon(), pn_);
    double curr_len = 0;
    calc_stretch_(&curr_len, &X[0]);
    *val = curr_len-len_;
    return 0;
  }
  int eval_jac(const double *x, double *jac) const {
    matd_t X = itr_matrix<const double*>(3, dim_/3, x)(colon(), pn_);
    calc_stretch_jac_(jac, &X[0]);
    return 0;
  }
private:
  const size_t dim_;
  double len_;
};

class bend_cons : public constraint_piece
{
public:
  bend_cons(const mati_t &dia, const matd_t &nods, const double k, const cons_type type=EQUAL)
    : dim_(nods.size()), dih_(0.0), constraint_piece(dia, k, type) {
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

pbd_cloth_solver::pbd_cloth_solver() {}

int pbd_cloth_solver::load_model_from_obj(const char *filename) {
  return jtf::mesh::load_obj(filename, tris_, nods_);
}

int pbd_cloth_solver::save_model_to_obj(const char *filename) const {
  return jtf::mesh::save_obj(filename, tris_, nods_);
}

int pbd_cloth_solver::load_model_from_vtk(const char *filename) {
  ifstream is(filename);
  if ( is.fail() )
    return __LINE__;
  return 0;
}

int pbd_cloth_solver::save_model_to_vtk(const char *filename) const {
  ofstream os(filename);
  if ( os.fail() )
    return __LINE__;
  tri2vtk(os, nods_.begin(), nods_.size(2), tris_.begin(), tris_.size(2));
  return 0;
}

int pbd_cloth_solver::init() {
  vel_.setZero(nods_.size());
  fext_.setZero(nods_.size());
  grav_.setZero(nods_.size());
}

void pbd_cloth_solver::set_mass_matrix(const double rho) {
  calc_mass_matrix(tris_, nods_, rho, 3, &M_, true);
  Minv_.resize(M_.cols());
  for (size_t i = 0; i < Minv_.size(); ++i)
    Minv_[i] = 1.0/M_.coeff(i, i);
}

void pbd_cloth_solver::apply_gravity() {
#pragma omp parallel for
  for (size_t i = 0; i < grav_.size()/3; ++i)
    grav_[3*i+1] = -M_.coeff(3*i, 3*i)*9.81;
  fext_ += grav_;
}

void pbd_cloth_solver::erase_gravity() {

}

void pbd_cloth_solver::attach_vert(const size_t id, const double *pos) {
  Minv_[3*id+0] = Minv_[3*id+1] = Minv_[3*id+2] = 0.0;
  if ( pos ) {
    nods_(0, id) = pos[0];
    nods_(1, id) = pos[1];
    nods_(2, id) = pos[2];
  }
}

double pbd_cloth_solver::query_constraint_squared_norm(const double *x) const {
  matd_t Cv = zeros<double>(buff_.size(), 1);
#pragma omp parallel for
  for (size_t i = 0; i < buff_.size(); ++i) {
    buff_[i]->eval_val(x, &Cv[i]);
  }
  return dot(Cv, Cv);
}

int pbd_cloth_solver::precompute() {
  /// construct constraints with rest configuration
  add_strecth_constraints(tris_, nods_);
//  add_bend_constraints(tris_, nods_);
  cout << "[info] constraint number: " << buff_.size() << endl;
  return 0;
}

int pbd_cloth_solver::project_constraints(vec_t &x, const size_t iter_num) {
  itr_matrix<double*> X(3, x.rows()/3, x.data());
  for (auto &co : buff_) {
    double val = 0.0;
    co->eval_val(x.data(), &val);
    if ( val == 0.0 )
      continue;
    if ( co->type_ == constraint_piece::EQUAL || (co->type_ == constraint_piece::GREATER && val < 0.0) ) {
      matd_t jac = zeros<double>(3, co->pn_.size());
      co->eval_jac(&X[0], &jac[0]);
      double s = 0;
      for (size_t i = 0; i < co->pn_.size(); ++i)
        s += Minv_[3*co->pn_[i]]*dot(jac(colon(), i), jac(colon(), i));
      if ( s == 0.0 )
        continue;
      s = val/s;
      for (size_t i = 0; i < co->pn_.size(); ++i)
        X(colon(), co->pn_[i]) += -s*Minv_[3*co->pn_[i]]*jac(colon(), i);
    }
  }
  return 0;
}

int pbd_cloth_solver::advance() {
  Map<VectorXd> X(nods_.begin(), nods_.size());
  VectorXd Xstar = X;
  vel_ += h_*Minv_.asDiagonal()*fext_;
  /// --------------------
  /// ----damp velocity---
  /// --------------------
  Xstar += h_*vel_;
  /// --------------------
  /// -gen collision cosn-
  /// --------------------
  for (size_t i = 0; i < MAX_ITER; ++i) {
    double cons_sqr = query_constraint_squared_norm(Xstar.data());
    cout << "\t@constraint norm: " << cons_sqr << endl;
    if ( cons_sqr < 1e-8 ) {
      cout << "\t@converged\n";
      break;
    }
    project_constraints(Xstar, i);
  }
  vel_ = (Xstar-X)/h_;
  X = Xstar;
  /// --------------------
  /// --velocity update---
  /// --------------------
  return 0;
}

int pbd_cloth_solver::add_strecth_constraints(const mati_t &tris, const matd_t &nods) {
  mati_t edge;
  get_edge_elements(tris, edge);
  for (size_t i = 0; i < edge.size(2); ++i) {
    buff_.push_back(std::make_shared<stretch_cons>(edge(colon(), i), nods, 1.0));
  }
  return 0;
}

int pbd_cloth_solver::add_bend_constraints(const mati_t &tris, const matd_t &nods) {
  mati_t dias;
  get_dihedral_elements(tris, dias);
  for (size_t i = 0; i < dias.size(2); ++i) {
    buff_.push_back(std::make_shared<bend_cons>(dias(colon(), i), nods, 1.0));
  }
  return 0;
}

//==============================================================================
int pbd_cloth_solver::test_calc_length() {
  const double x[6] = {0,1,0, 1,0,0};
  double value = 0;
  calc_stretch_(&value, x);
  cout << value << endl;
  return 0;
}

int pbd_cloth_solver::test_calc_dihedral_angle() {
  const double x[12] = {0,-1,1, 0,0,0, 1,0,0, 0,1,0};
  double value = 0;
  calc_bend_(&value, x);
  cout << value/M_PI*180 << endl;
  return 0;
}

int pbd_cloth_solver::test_edge_extraction() {
  mati_t edge;
  get_edge_elements(tris_, edge);
  cout << edge << endl;
  return 0;
}

int pbd_cloth_solver::test_diamond_extraction() {
  mati_t dias;
  get_dihedral_elements(tris_, dias);
  cout << dias << endl;
  return 0;
}

}
