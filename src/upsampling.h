#ifndef UPSAMPLING_H
#define UPSAMPLING_H

#include <zjucad/matrix/matrix.h>

namespace bigbang {

class upsampler
{
public:
  upsampler(const mati_t &tric, const std::vector<matd_t> &pc,
            const mati_t &trif, const std::vector<matd_t> &pf);
};

}

#endif
