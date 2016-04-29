#include <iostream>
#include <fstream>
#include <jtflib/mesh/io.h>
#include <jtflib/mesh/util.h>
#include <boost/filesystem.hpp>
#include <zjucad/matrix/io.h>

#include "src/config.h"
#include "src/json.h"
#include "src/energy.h"
#include "src/geom_util.h"
#include "src/mass_matrix.h"
#include "src/vtk.h"
#include "src/util.h"

using namespace std;
using namespace bigbang;
using namespace Eigen;
using namespace zjucad::matrix;

static vector<shared_ptr<Functional<double>>> g_buffer;
static size_t g_node_size, g_frm_size;
static SparseMatrix<double> g_M;
static double g_h;

static void advance(VectorXd &xn, VectorXd &vn) {
  const size_t dim = xn.size();
  SimplicialCholesky<SparseMatrix<double>> solver;
  VectorXd xstar = xn;
  for (size_t iter = 0; iter < 50; ++iter) {
    double value = 0; {
      if ( iter % 10 == 0 ) {
        for (auto &energy : g_buffer)
          if ( energy.get() )
            energy->Val(xstar.data(), &value);
        cout << "\t@potential: " << value << endl;
      }
    }
    VectorXd g = VectorXd::Zero(dim); {
      for (auto &energy : g_buffer)
        if ( energy.get() )
          energy->Gra(xstar.data(), g.data());
      g *= -1;
    }
    SparseMatrix<double> H(dim, dim); {
      vector<Triplet<double>> trips;
      for (auto &energy : g_buffer)
        if ( energy.get() )
          energy->Hes(xstar.data(), &trips);
      H.setFromTriplets(trips.begin(), trips.end());
    }
    SparseMatrix<double> LHS = g_M+g_h*g_h*H;
    VectorXd rhs = g_h*g_h*g-g_M*(xstar-xn-g_h*vn);
    solver.compute(LHS);
    ASSERT(solver.info() == Success);
    VectorXd dx = solver.solve(rhs);
    ASSERT(solver.info() == Success);
    xstar += dx;
    for (size_t k = 0; k < g_frm_size/4; ++k)
      xstar.segment<4>(g_node_size+4*k).normalize();
  }
  vn = (xstar-xn)/g_h;
  xn = xstar;
}

class MassCalculator
{
public:
  MassCalculator(const mati_t &tris, const mati_t &rod, const matd_t &nods,
                 const double rho, const double r)
    : r_size_(nods.size()), q_size_(4*(rod.size()-1)), I_(3), B_(3) {
    for (size_t i = 0; i < tris.size(2); ++i) {
      const double mas = rho*jtf::mesh::cal_face_area(nods(colon(), tris(colon(), i)));
      add_diag_block<double, 3>(tris(0, i), tris(0, i), mas/3, &tripsA_);
      add_diag_block<double, 3>(tris(1, i), tris(1, i), mas/3, &tripsA_);
      add_diag_block<double, 3>(tris(2, i), tris(2, i), mas/3, &tripsA_);
    }
    lenq_ = VectorXd::Zero(rod.size()-2);
    for (size_t i = 0; i < lenq_.size(); ++i) {
      lenq_(i) = 0.5*(norm(nods(colon(), rod[i])-nods(colon(), rod[i+1]))+
          norm(nods(colon(), rod[i+1])-nods(colon(), rod[i+2])));
    }
    I_[0] = I_[1] = rho*M_PI*r*r/4.0;
    I_[2] = 2*I_[0];
    B_[0] << 0, 0, 0, 1,
        0, 0, 1, 0,
        0, -1, 0, 0,
        -1, 0, 0, 0;
    B_[1] << 0, 0, -1, 0,
        0, 0, 0, 1,
        1, 0, 0, 0,
        0, -1, 0, 0;
    B_[2] << 0, 1, 0, 0,
        -1, 0, 0, 0,
        0, 0, 0, 1,
        0, 0, -1, 0;
  }
  void mass_matrix(const VectorXd &x, SparseMatrix<double> &M) {
    tripsB_.clear();
    for (size_t i = 0; i < lenq_.size(); ++i) {
      for (size_t j = 0; j < 3; ++j) {
        Vector4d bj = B_[j]*(x.segment<4>(r_size_+4*i)+x.segment<4>(r_size_+4*(i+1)));
        Matrix4d dM = 0.25*lenq_(i)*I_[j]*bj*bj.transpose();
        insert_block<double>(r_size_+4*i, r_size_+4*i, dM.data(), 4, 4, &tripsB_);
        insert_block<double>(r_size_+4*i, r_size_+4*i+4, dM.data(), 4, 4, &tripsB_);
        insert_block<double>(r_size_+4*i+4, r_size_+4*i, dM.data(), 4, 4, &tripsB_);
        insert_block<double>(r_size_+4*i+4, r_size_+4*i+4, dM.data(), 4, 4, &tripsB_);
      }
    }
    vector<Triplet<double>> trips(tripsA_);
    trips.insert(std::end(trips), std::begin(tripsB_), std::end(tripsB_));
    M.resize(x.size(), x.size());
    M.setFromTriplets(trips.begin(), trips.end());
  }
private:
  const size_t r_size_, q_size_;
  VectorXd lenq_;
  vector<double> I_;
  vector<Matrix4d> B_;
  vector<Triplet<double>> tripsA_, tripsB_;
};

int main(int argc, char *argv[])
{
  if ( argc != 2 ) {
    cerr << "# usage: ./test_hybrid_stitch config.json\n";
    return __LINE__;
  }
  Json::Reader reader;
  Json::Value json;
  ifstream ifs(argv[1]);
  if ( ifs.fail() ) {
    cerr << "[Error] can't open " << argv[1] << endl;
    return __LINE__;
  }
  if ( !reader.parse(ifs, json) ) {
    cerr << "[Error] " << reader.getFormattedErrorMessages() << endl;
    return __LINE__;
  }
  ifs.close();
  boost::filesystem::create_directories(json["outdir"].asString());

  // write config file first
  ofstream ofs(json["outdir"].asString()+"/config.json");
  Json::StyledWriter styledWriter;
  ofs << styledWriter.write(json) << endl;
  ofs.close();

  Json::Value &clothJson = json["cloth"];
  Json::Value &rodJson = json["stitch"];

  mati_t tris, edges, diams; matd_t nods; {
    jtf::mesh::load_obj(clothJson["mesh"].asString().c_str(), tris, nods);
    get_edge_elem(tris, edges);
    get_diam_elem(tris, diams);
  }
  mati_t rod; matd_t frm; {
    rod.resize(rodJson["chain"].size());
    for (int i = 0; i < rodJson["chain"].size(); ++i)
      rod[i] = rodJson["chain"][i].asUInt();
    frm.resize(4, rod.size()-1);
    for (size_t i = 0; i < frm.size(2); ++i) {
      Matrix3d basis;
      basis.col(2) = -Vector3d::UnitX();
      basis.col(0) = Vector3d::UnitY();
      basis.col(1) = basis.col(2).cross(basis.col(0));
      Map<Vector4d>(&frm(0, i)) = Quaterniond(basis).coeffs();
    }
  }

  g_buffer.resize(7);
  g_buffer[0] = make_shared<spring_potential>(edges, nods, clothJson["stretch"].asDouble());
  g_buffer[1] = make_shared<isometric_bending>(diams, nods, clothJson["bend"].asDouble());
  g_buffer[2] = make_shared<gravitational_potential>(tris, nods, json["density"].asDouble(), 1.0);
  g_buffer[3] = make_shared<positional_potential>(nods, clothJson["position"].asDouble());
  g_buffer[4] = make_shared<cosserat_stretch_energy>(rod, nods, rodJson["stretch"].asDouble(), rodJson["radius"].asDouble());
  g_buffer[5] = make_shared<cosserat_bend_energy>(rod, nods, rodJson["young"].asDouble(), rodJson["shear"].asDouble(), rodJson["radius"].asDouble());
  g_buffer[6] = make_shared<cosserat_couple_energy>(rod, nods, rodJson["spring"].asDouble());

  // fix handle nodes
  for (int i = 0; i < clothJson["handles"].size(); ++i) {
    const size_t id = clothJson["handles"][i].asUInt();
    dynamic_pointer_cast<positional_potential>(g_buffer[3])->Pin(id, &nods(0, id));
  }

  g_h = json["timestep"].asDouble();
  g_node_size = nods.size();
  g_frm_size = frm.size();

  MassCalculator mc(tris, rod, nods, json["density"].asDouble(), rodJson["radius"].asDouble());
  ASSERT(g_node_size+g_frm_size == g_buffer.back()->Nx());
  VectorXd vn = VectorXd::Zero(g_node_size+g_frm_size), xn(g_node_size+g_frm_size);
  std::copy(nods.begin(), nods.end(), xn.data());
  std::copy(frm.begin(), frm.end(), xn.data()+g_node_size);
  for (size_t frm = 0; frm < json["frames"].asUInt(); ++frm) {
    cout << "[Info] frame " << frm << endl;
    {
      char outfile[256];
      sprintf(outfile, "%s/frame_%zu.vtk", json["outdir"].asString().c_str(), frm);
      ofstream ofs(outfile);
      tri2vtk(ofs, xn.data(), nods.size(2), &tris[0], tris.size(2));
      ofs.close();
    }
    mc.mass_matrix(xn, g_M);
    advance(xn, vn);
  }

  cout << "[Info] done\n";
  return 0;
}
