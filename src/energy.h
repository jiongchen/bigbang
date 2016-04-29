#ifndef ENERGY_H
#define ENERGY_H

#include <zjucad/matrix/matrix.h>
#include <Eigen/Sparse>
#include <unordered_map>

#include "def.h"

using mati_t=zjucad::matrix::matrix<size_t>;
using matd_t=zjucad::matrix::matrix<double>;

namespace bigbang {

class momentum_potential : public Functional<double>
{
public:
  ~momentum_potential() {}
  virtual size_t Nx() const = 0;
  virtual int Val(const double *x, double *val) const = 0;
  virtual int Gra(const double *x, double *gra) const = 0;
  virtual int Hes(const double *x, std::vector<Eigen::Triplet<double>> *hes) const = 0;
  virtual void Update(const double *x) = 0;
  virtual double QueryKineticEnergy() const = 0;
  virtual const Eigen::SparseMatrix<double>& MassMatrix() const = 0;
  virtual const Eigen::VectorXd& CurrVelocity() const = 0;
  virtual double timestep() const = 0;
};

class momentum_potential_imp_euler : public momentum_potential
{
public:
  momentum_potential_imp_euler(const mati_t &cell, const matd_t &nods, const double rho, const double h, const double w=1.0);
  size_t Nx() const;
  int Val(const double *x, double *val) const;
  int Gra(const double *x, double *gra) const;
  int Hes(const double *x, std::vector<Eigen::Triplet<double>> *hes) const;
  void Init(const double *x0, const double *v0);
  void Update(const double *x);
  double QueryKineticEnergy() const;
  const Eigen::SparseMatrix<double>& MassMatrix() const { return M_; }
  const Eigen::VectorXd& CurrVelocity() const { return vn_; }
  double timestep() const { return h_; }
private:
  const double rho_, h_;
  const size_t dim_;
  double w_;
  Eigen::SparseMatrix<double> M_;
  Eigen::VectorXd xn_, vn_;
};

class momentum_potential_bdf2 : public momentum_potential
{
public:
  momentum_potential_bdf2(const mati_t &cell, const matd_t &nods, const double rho, const double h, const double w=1.0);
  size_t Nx() const;
  int Val(const double *x, double *val) const;
  int Gra(const double *x, double *gra) const;
  int Hes(const double *x, std::vector<Eigen::Triplet<double>> *hes) const;
  void Init(const double *x0, const double *v0);
  void Update(const double *x);
  double QueryKineticEnergy() const;
  const Eigen::SparseMatrix<double>& MassMatrix() const { return M_; }
  const Eigen::VectorXd& CurrVelocity() const { return vn_; }
  double timestep() const { return h_; }
private:
  const double rho_, h_;
  const size_t dim_;
  double w_;
  Eigen::SparseMatrix<double> M_;
  Eigen::VectorXd xn_, vn_, xnn_, vnn_;
};

class elastic_potential : public Functional<double>
{
public:
  enum Material {
    LINEAR,
    STVK,
    COROTATIONAL,
    NEOHOOKEAN
  };
  elastic_potential(const mati_t &tets, const matd_t &nods, Material type, const double Ym, const double Pr, const double w);
  size_t Nx() const;
  int Val(const double *x, double *val) const;
  int Gra(const double *x, double *gra) const;
  int Hes(const double *x, std::vector<Eigen::Triplet<double>> *hes) const;
private:
  const size_t dim_;
  double w_;
  Material type_;
  const mati_t &tets_;
  matd_t vol_;
  double lam_, miu_;
  matd_t Dm_;
  matd_t rest_;
};

class voxel_elastic_potential : public Functional<double>
{
public:
  enum Material {
    LINEAR,
    STVK
  };
  voxel_elastic_potential(const mati_t &cube, const matd_t &nods, Material type,
                          const double Ym, const double Pr, const double w);
  size_t Nx() const;
  int Val(const double *x, double *val) const;
  int Gra(const double *x, double *gra) const;
  int Hes(const double *x, std::vector<Eigen::Triplet<double>> *hes) const;
private:
  const size_t dim_;
  double w_;
  Material type_;
  const mati_t &cube_;
  double h_, lam_, miu_;
};

class gravitational_potential : public Functional<double>
{
public:
  gravitational_potential(const mati_t &cell, const matd_t &nods, const double rho, const double w, const int direction=1);
  size_t Nx() const;
  int Val(const double *x, double *val) const;
  int Gra(const double *x, double *gra) const;
  int Hes(const double *x, std::vector<Eigen::Triplet<double>> *hes) const { return __LINE__; }
private:
  const int direction_;
  const size_t dim_;
  double w_;
  Eigen::SparseMatrix<double> M_;
};

class positional_potential : public Functional<double>
{
public:
  positional_potential(const matd_t &nods, const double w);
  size_t Nx() const;
  int Val(const double *x, double *val) const;
  int Gra(const double *x, double *gra) const;
  int Hes(const double *x, std::vector<Eigen::Triplet<double>> *hes) const;
  int Pin(const size_t id, const double *pos);
  int Release(const size_t id);
private:
  const size_t dim_;
  double w_;
  std::unordered_map<size_t, Eigen::Vector3d> fixed_;
};

class spring_potential : public Functional<double>
{
public:
  spring_potential(const mati_t &edge, const matd_t &nods, const double w);
  size_t Nx() const;
  int Val(const double *x, double *val) const;
  int Gra(const double *x, double *gra) const;
  int Hes(const double *x, std::vector<Eigen::Triplet<double>> *hes) const;
  void ResetWeight(const double w) { w_ = w; }
  int ResetEdgeMaterial(const size_t p, const size_t q, const double scale_w, const double scale_len);
private:
  const size_t dim_;
  matd_t w_;
  const mati_t &edge_;
  matd_t len_;
};

class gauss_newton_spring : public Functional<double>
{
public:
  gauss_newton_spring(const mati_t &edge, const matd_t &nods, const double w);
  size_t Nx() const;
  int Val(const double *x, double *val) const;
  int Gra(const double *x, double *gra) const;
  int Hes(const double *x, std::vector<Eigen::Triplet<double>> *hes) const;
  void ResetWeight(const double w) { w_ = w; }
private:
  const size_t dim_;
  matd_t w_;
  const mati_t &edge_;
  matd_t len_;
};

template<typename T>
void hash_combine(size_t &seed, T const &v) {
  seed ^= std::hash<T>()(v)+0x9e3779b9+(seed<<6)+(seed>>2);
}

struct pair_hash {
public:
  template<typename T, typename U>
  size_t operator()(const std::pair<T, U> &rhs) const {
    size_t retval = std::hash<T>()(rhs.first);
    hash_combine(retval, rhs.second);
    return retval;
  }
};

class fast_mass_spring : public Functional<double>
{
public:
  fast_mass_spring(const mati_t &edge, const matd_t &nods, const double w);
  size_t Nx() const;
  int Val(const double *x, double *val) const;
  int Gra(const double *x, double *gra) const;
  int Hes(const double *x, std::vector<Eigen::Triplet<double>> *hes) const;
  void LocalSolve(const double *x);
  void Project();
  size_t aux_dim() const { return 3*edge_.size(2); }
  const double* get_aux_var() const { return d_.begin(); }
  const Eigen::SparseMatrix<double>& get_df_mat() const { return S_; }
  void build_jts_pattern();
  Eigen::SparseMatrix<double>& get_jts(const double *x);
public:
  const mati_t &edge_;
  matd_t len_;
  matd_t d_;
private:
  const size_t dim_;
  double w_;
  std::unordered_map<std::pair<size_t, size_t>, size_t, pair_hash> ijp_;
  Eigen::SparseMatrix<double> S_, JtS_;
};

class line_bending_potential : public Functional<double>
{
public:
  line_bending_potential(const mati_t &edge, const matd_t &nods, const double w);
  size_t Nx() const;
  int Val(const double *x, double *val) const;
  int Gra(const double *x, double *gra) const;
  int Hes(const double *x, std::vector<Eigen::Triplet<double>> *hes) const;
private:
  const size_t dim_;
  double w_;
  const mati_t &edge_;
  matd_t len_;
};

class surf_bending_potential : public Functional<double>
{
public:
  surf_bending_potential(const mati_t &diams, const matd_t &nods, const double w);
  size_t Nx() const;
  int Val(const double *x, double *val) const;
  int Gra(const double *x, double *gra) const;
  int Hes(const double *x, std::vector<Eigen::Triplet<double>> *hes) const;
  void ResetWeight(const double w) { w_ = w; }
private:
  const size_t dim_;
  double w_;
  const mati_t &diams_;
  matd_t len_, area_, angle_;
};

class ext_force_energy : public Functional<double>
{
public:
  ext_force_energy(const matd_t &nods, const double w);
  size_t Nx() const;
  int Val(const double *x, double *val) const;
  int Gra(const double *x, double *gra) const;
  int Hes(const double *x, std::vector<Eigen::Triplet<double>> *hes) const { return __LINE__; }
  int ApplyForce(const size_t id, const double *f);
  int RemoveForce(const size_t id);
private:
  const size_t dim_;
  double w_;
  matd_t force_;
};

class isometric_bending : public Functional<double>
{
public:
  isometric_bending(const mati_t &diams, const matd_t &nods, const double w);
  size_t Nx() const;
  int Val(const double *x, double *val) const;
  int Gra(const double *x, double *gra) const;
  int Hes(const double *x, std::vector<Eigen::Triplet<double>> *hes) const;
private:
  const size_t dim_;
  const mati_t &diams_;
  double w_;
  matd_t cotv_, area_;
};

class tet_arap_energy : public Functional<double>
{
public:
  tet_arap_energy(const mati_t &tets, const matd_t &nods, const double w);
  size_t Nx() const;
  int Val(const double *x, double *val) const;
  int Gra(const double *x, double *gra) const;
  int Hes(const double *x, std::vector<Eigen::Triplet<double>> *hes) const;
  void LocalSolve(const double *x);
  void CalcLieAlgebraCoord(Eigen::VectorXd &vec) const;
  void UpdateRotation(const Eigen::VectorXd &vec);
  size_t aux_dim() const { return 9*tets_.size(2); }
  const double* get_aux_var() const { return R_.begin(); }
public:
  matd_t R_;
private:
  const size_t dim_;
  double w_;
  const mati_t &tets_;
  matd_t vol_;
  matd_t D_;
};

class bw98_stretch_energy : public Functional<double>
{
public:
  bw98_stretch_energy(const mati_t &tris, const matd_t &nods, const double w);
  size_t Nx() const;
  int Val(const double *x, double *val) const;
  int Gra(const double *x, double *gra) const;
  int Hes(const double *x, std::vector<Eigen::Triplet<double>> *hes) const;
private:
  const size_t dim_;
  double w_;
  const mati_t &tris_;
  matd_t area_;
  matd_t invUV_;
};

class bw98_shear_energy : public Functional<double>
{
public:
  bw98_shear_energy(const mati_t &tris, const matd_t &nods, const double w);
  size_t Nx() const;
  int Val(const double *x, double *val) const;
  int Gra(const double *x, double *gra) const;
  int Hes(const double *x, std::vector<Eigen::Triplet<double>> *hes) const;
private:
  const size_t dim_;
  double w_;
  const mati_t &tris_;
  matd_t area_;
  matd_t invUV_;
};

class fem_stretch_energy : public Functional<double>
{
public:
  fem_stretch_energy(const mati_t &tris, const matd_t &nods, const double w);
  size_t Nx() const;
  int Val(const double *x, double *val) const;
  int Gra(const double *x, double *gra) const;
  int Hes(const double *x, std::vector<Eigen::Triplet<double>> *hes) const;
private:
  const size_t dim_;
  double w_;
  const mati_t &tris_;
  matd_t area_, K_;
  matd_t Dm_;
};

class low_pass_filter_energy : public Functional<double>
{
public:
  low_pass_filter_energy(const mati_t &tris, const matd_t &nods, const size_t patch_num, const double w);
  size_t Nx() const;
  int Val(const double *x, double *val) const;
  int Gra(const double *x, double *gra) const;
  int Hes(const double *x, std::vector<Eigen::Triplet<double>> *hes) const;
  void Update(const double *ref);
private:
  const size_t dim_;
  double w_;
  const size_t patch_num_;
  const mati_t &tris_;
  const double *ref_;
  std::vector<std::vector<std::pair<size_t, double>>> pat_;
};

class cosserat_stretch_energy : public Functional<double>
{
public:
  cosserat_stretch_energy(const mati_t &rod, const matd_t &nods,
                          const double Es, const double r, const double w=1.0);
  size_t Nx() const;
  int Val(const double *xq, double *val) const;
  int Gra(const double *xq, double *gra) const;
  int Hes(const double *xq, std::vector<Eigen::Triplet<double>> *hes) const;
private:
  const mati_t &rod_;
  const double w_;
  const size_t r_size_, q_size_;
  const size_t elem_num_;
  const double Es_, r_;
  Eigen::VectorXd len_;
};

class cosserat_bend_energy : public Functional<double>
{
public:
  cosserat_bend_energy(const mati_t &rod, const matd_t &nods,
                       const double E, const double G, const double r, const double w=1.0);
  size_t Nx() const;
  int Val(const double *xq, double *val) const;
  int Gra(const double *xq, double *gra) const;
  int Hes(const double *xq, std::vector<Eigen::Triplet<double>> *hes) const;
private:
  const mati_t &rod_;
  const double w_;
  const size_t r_size_, q_size_;
  const size_t elem_num_;
  const double E_, G_, r_;
  Eigen::VectorXd len_;
};

class cosserat_couple_energy : public Functional<double>
{
public:
  cosserat_couple_energy(const mati_t &rod, const matd_t &nods,
                         const double kappa, const double w=1.0);
  size_t Nx() const;
  int Val(const double *xq, double *val) const;
  int Gra(const double *xq, double *gra) const;
  int Hes(const double *xq, std::vector<Eigen::Triplet<double>> *hes) const;
private:
  const mati_t &rod_;
  const double w_;
  const size_t r_size_, q_size_;
  const size_t elem_num_;
  const double kappa_;
  Eigen::VectorXd len_;
};

}

#endif
