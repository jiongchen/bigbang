#include <iostream>
#include <fstream>
#include <boost/filesystem.hpp>
#include <jtflib/mesh/io.h>

#include "src/json.h"
#include "src/energy.h"
#include "src/geom_util.h"
#include "src/vtk.h"
#include "src/energy.h"
#include "src/optimizer.h"

using namespace std;
using namespace bigbang;

static int parse_stitch(const Json::Value &json, mati_t &edge) {
  if ( json.isNull() || !json.isArray() )
    return __LINE__;
  const size_t edge_num = json.size()-1;
  edge.resize(2, edge_num);
  for (int i = 0; i < edge_num; ++i) {
    edge(0, i) = json[i].asUInt();
    edge(1, i) = json[i+1].asUInt();
  }
  return 0;
}

int main(int argc, char *argv[])
{
  if ( argc != 2 ) {
    cerr << "# usage: ./prog config.json\n";
    return __LINE__;
  }
  Json::Reader reader;
  Json::Value json;
  ifstream ifs(argv[1]);
  if ( ifs.fail() ) {
    cerr << "[Error] can't open " << argv[1] << endl;
    return __LINE__;
  }
  bool flag = reader.parse(ifs, json);
  if ( !flag ) {
    cerr << "[Error] " << reader.getFormattedErrorMessages() << endl;
    return __LINE__;
  }
  ifs.close();
  boost::filesystem::create_directories(json["outdir"].asString());

  mati_t tris; matd_t nods;
  jtf::mesh::load_obj(json["mesh"].asString().c_str(), tris, nods);

  mati_t edges, diams, stitch;
  get_edge_elem(tris, edges);
  get_diam_elem(tris, diams);
  int stitch_is_valid = parse_stitch(json["stitch"], stitch);

  // parse parameters and assemble energies
  double rho = json["density"].asDouble(),
      h = json["timestep"].asDouble(),
      ws = json["ws"].asDouble(),
      wb = json["wb"].asDouble(),
      wg = json["wg"].asDouble(),
      wp = json["wp"].asDouble(),
      wl = json["wl"].asDouble();
  int gn = json["gravity_direction"].asInt();
  vector<shared_ptr<Functional<double>>> ebf(6);
  shared_ptr<Functional<double>> energy; {
    ebf[0] = make_shared<momentum_potential_imp_euler>(tris, nods, rho, h, 1.0);
    ebf[1] = make_shared<spring_potential>(edges, nods, ws);
    ebf[2] = make_shared<isometric_bending>(diams, nods, wb);
    ebf[3] = make_shared<gravitational_potential>(tris, nods, rho, wg, gn);
    ebf[4] = make_shared<positional_potential>(nods, wp);
    if ( !stitch_is_valid ) ebf[5] = make_shared<line_bending_potential>(stitch, nods, wl);
    try {
      energy = make_shared<energy_t<double>>(ebf);
    } catch ( exception &e ) {
      cerr << e.what() << endl;
      exit(EXIT_FAILURE);
    }
  }

  // set anistropic stitch
  const double stiffness_scale = json["stiffness_scale"].asDouble(), rest_length_scale = json["rest_length_scale"].asDouble();
  auto ms = dynamic_pointer_cast<spring_potential>(ebf[1]);
  for (size_t i = 0; i < stitch.size(2); ++i) {
    ms->ResetEdgeMaterial(stitch(0, i), stitch(1, i), stiffness_scale, rest_length_scale);
  }

  // parse handles
  auto pc = dynamic_pointer_cast<positional_potential>(ebf[4]);
  for (int i = 0; i < json["handles"].size(); ++i) {
    const size_t idx = json["handles"][i]["id"].asInt();
    const double pos[3] = {json["handles"][i]["pos"][0].asDouble(), json["handles"][i]["pos"][1].asDouble(), json["handles"][i]["pos"][2].asDouble()};
    pc->Pin(idx, pos);
  }

  char outfile[256];
  const size_t frames = json["frames"].asInt();
  opt_args optarg = {json["max_iter"].asUInt(), json["tolerance"].asDouble(), json["line_search"].asBool()};
  // simulate
  for (size_t i = 0; i < frames; ++i) {
    cout << "[Info] frame " << i << endl;
    sprintf(outfile, "%s/frame_%zu.vtk", json["outdir"].asString().c_str(), i);
    ofstream os(outfile);
    tri2vtk(os, &nods[0], nods.size(2), &tris[0], tris.size(2));
    os.close();

    lbfgs_solve(&nods[0], nods.size(), energy, optarg);
    dynamic_pointer_cast<momentum_potential>(ebf[0])->Update(&nods[0]);
  }

  cout << "[Info] done\n";
  return 0;
}
