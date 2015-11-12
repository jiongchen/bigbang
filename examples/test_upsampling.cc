#include <iostream>
#include <fstream>

#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>

using namespace std;
namespace po=boost::program_options;

int offline_coarse_sim(po::variables_map &vm) {
  cout << "[info] coarse offline sim\n";
  return 0;
}

int offline_fine_track_sim(po::variables_map &vm) {
  cout << "[info] fine offline sim\n";
  return 0;
}

int upsampling(po::variables_map &vm) {
  cout << "[info] upsampling\n";
  return 0;
}

int online_sim(po::variables_map &vm) {
  cout << "[info] online sim\n";
  return 0;
}

int main(int argc, char *argv[])
{
  po::options_description desc("available options");
  desc.add_options()
      ("help,j", "produce help message")
      ("program,p", po::value<string>(), "program to run")
      ;
  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);
  if ( vm.count("help") ) {
    cout << desc << endl;
    return __LINE__;
  }

  const string prog_name = vm["program"].as<string>();
  if ( prog_name == "offline_coarse_sim" )
    return offline_coarse_sim(vm);
  if ( prog_name == "offline_fine_track_sim" )
    return offline_fine_track_sim(vm);
  if ( prog_name == "upsampling" )
    return upsampling(vm);
  if ( prog_name == "online_sim" )
    return online_sim(vm);

//  subprog:
//    run coarse level simulation, get pc;
//  subprog:
//    run fine level simulation, calculate barycentric interp operator, every frame tracks updated, get pf;
//  subprog:
//    construct upsampling operator according to pc and pf;
//  subprog:
//    online simulation, pf = U*pc;

  return 0;
}
