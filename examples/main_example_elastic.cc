#include <iostream>

#include "src/example_elastic.h"

using namespace std;
using namespace bigbang;

int main(int argc, char *argv[])
{
  strain_calc_test();
  numeric_diff_test();
  return 0;
}
