//#include "fast_proj.h"

//#include "geom_util.h"
//#include "energy.h"
//#include "mass_matrix.h"

//using namespace std;
//using namespace zjucad::matrix;
//using namespace Eigen;

//namespace bigbang {

//cloth_solver::cloth_solver(const mati_t &tris, const matd_t &nods, cloth_args &args)
//  : dim_(nods.size()), tris_(tris), nods_(nods), args_(args) {
//  get_edge_elem(tris_, edges_);
//  get_diam_elem(tris_, diams_);
//}

//int cloth_solver::initialize() {
//  calc_mass_matrix(tris_, nods_, args_.density, 3, &M_, false);
//  vel_.setZero(dim_);
//  fext_.setZero(dim_);

//  ebf_.resize(3);
//  ebf_[0] = make_shared<surf_bending_potential>;
//  ebf_[1] = make_shared<gravitational_potential>;

//  cbf_.resize(2);
//  cbf_[0] = make_shared<>;

//  try {
//    energy_ = make_shared<energy_t<double>>(ebf_);
//    constraint_ = make_shared<constraint_t<double>>(cbf_);
//  } catch ( exception &e ) {
//    cerr << "[exception] " << e.what() << endl;
//    exit(EXIT_FAILURE);
//  }
//  return 0;
//}


//}
