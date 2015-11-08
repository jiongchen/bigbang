#ifndef MODAL_ANALYSIS_H
#define MODAL_ANALYSIS_H

#include <unordered_set>
#include <Eigen/Sparse>

namespace bigbang {

using spmatd_t=Eigen::SparseMatrix<double>;

/// solve the generalized eigen value problem
/// $Ax=\lambda Bx$, where $A$ is symmetric
/// and $B$ is symmetric positive definite
class gevp_solver
{
public:
  enum EigenvalueType {
    ALGEBRAIC_LARGEST,   // Largest as real numbers
    ALGEBRAIC_SMALLEST,  // Smallest as real numbers
    ALGEBRAIC_BOTH_END,  // Both end as real numbers
    MAGNITUDE_LARGEST,   // Largest as absolute values
    MAGNITUDE_SMALLEST,  // Smallest as absolute values
  };
  int solve(const spmatd_t &A, const spmatd_t &B, const size_t num, const EigenvalueType mode,
            Eigen::MatrixXd &U, Eigen::VectorXd &lambda);
};

/// linear modal analysis: $Kx = \lambda Mx$
/// subject to some fixed vertices
/// modal derivatives extend LMA to capture
/// more nonlinear deformation basis
class basis_builder
{
public:
  basis_builder(const spmatd_t &M, const spmatd_t &K, const std::unordered_set<size_t> &fixDOF);
  int compute(const size_t num, Eigen::MatrixXd &U, Eigen::VectorXd &lambda) const;
  int extend(); // modal derivatives
private:
  const size_t dim_;
  spmatd_t C_, Linv_;
  std::vector<size_t> g2l_;
};

}

#endif
