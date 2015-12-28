#include <fstream>
#include <iostream>

using namespace std;

int main(int argc, char *argv[])
{
  ifstream ifs(argv[1], ios::binary);
  size_t dimx;
  ifs.read((char *)&dimx, sizeof(size_t));
  cout << dimx/3 << endl;
  size_t steps;
  ifs.read((char *)&steps, sizeof(size_t));
  cout << steps << endl;
  double x, y, z;
  ifs.read((char *)&x, sizeof(double));
  ifs.read((char *)&y, sizeof(double));
  ifs.read((char *)&z, sizeof(double));
  cout << x << " " << y << " " << z << endl;
  return 0;
}
