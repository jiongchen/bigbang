#include <iostream>
#include <fstream>
#include <zjucad/matrix/itr_matrix.h>
#include <jtflib/mesh/io.h>

using namespace std;
using matrixst = zjucad::matrix::matrix<size_t>;
using matrixd = zjucad::matrix::matrix<double>;

template<typename T1, typename T2>
int orient_tet_raw(const double * node_array, T2 node_num,
                   T1 * tet_array, T2 tet_num)
{
  zjucad::matrix::itr_matrix<const double *> node(3, node_num, node_array);
  zjucad::matrix::itr_matrix<T1 *> tet(4, tet_num, tet_array);

  for(T2 ti = 0; ti < tet.size(2); ++ti) {
    matrixd ele(3, 3);
    for(T2 ni = 0; ni < 3; ++ni)
      ele(zjucad::matrix::colon(), ni) =
          node(zjucad::matrix::colon(), tet(ni+1, ti))
          - node(zjucad::matrix::colon(), tet(0, ti));

    if(zjucad::matrix::dot(
         zjucad::matrix::cross(ele(zjucad::matrix::colon(), 0),
                               ele(zjucad::matrix::colon(), 1)),
         ele(zjucad::matrix::colon(), 2)) < 0) {
      std::swap(tet(1, ti), tet(2, ti));
    }
  }
  return 0;
}

template <typename T>
int orient_tet(const matrixd &node,
               zjucad::matrix::matrix<T> &tet)
{
  return orient_tet_raw(&node[0], node.size(2), &tet[0], tet.size(2));
}


int vol2tet(const char * filename,
	    matrixst & tet,
	    matrixd & node)
{
  ifstream ifs(filename);
  if(ifs.fail()){
    cerr << "# [error] can not open vol file." << endl;
    return __LINE__;
  }

  string line;
  while(!ifs.eof()){
    getline(ifs, line);
    if(line  == "dimension") {
      size_t d = 0 ;
      ifs >> d;
      if(d != 3){
        cerr << "# [error] dimesnsion is " << d << ", not 3." << endl;
        return __LINE__;
      }
    }

    if(line == "geomtype"){
      size_t geo_t = 0;
      ifs >> geo_t;
      if(geo_t != 11){
        cerr << "# [error] geomtye is " << geo_t << ", not 11." << endl;
        return __LINE__;
      }
    }

    if(line[0] == '#'){
      ifs >> line;
      if(line == "volumeelements"){
        size_t tet_num = 0;
        ifs >> tet_num;
        tet.resize(4, tet_num);
        size_t trash;
        for(size_t ti = 0; ti < tet_num; ++ti){
          ifs >> trash >> trash;
          for(size_t di = 0; di < 4; ++di)
            ifs >> tet(di,ti);
        }
        tet -= 1;
      }

      if(line == "points"){
        size_t point_num;
        ifs >> point_num;
        node.resize(3, point_num);
        for(size_t pi = 0; pi < point_num; ++pi)
          for(size_t di = 0; di < 3; ++di)
            ifs >> node(di,pi);
      }
    }
  }
  return 0;
}

int main(int argc, char * argv[])
{
  if(argc != 3){
    cerr << "# [usage] vol2tet vol tet." << endl;
    return __LINE__;
  }
  matrixst tet;
  matrixd node;
  vol2tet(argv[1], tet, node);
  orient_tet(node, tet);
  if(jtf::mesh::tet_mesh_write_to_zjumat(argv[2], &node, &tet))
    return __LINE__;
  cerr << "# [info] success." << endl;
  return 0;
}
