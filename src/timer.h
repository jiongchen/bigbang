#ifndef HIGH_RESOLUTION_TIMER_H
#define HIGH_RESOLUTION_TIMER_H

#include <chrono>

namespace bigbang {

class high_resolution_timer
{
public:
  typedef std::chrono::high_resolution_clock clock_type;
  void start() {
    t0_ = clk_.now();
  }
  void stop() {
    t1_ = clk_.now();
  }
  uint64_t duration() const {
    return std::chrono::duration_cast<std::chrono::milliseconds>(t1_-t0_).count();
  }
private:
  std::chrono::high_resolution_clock clk_;
  clock_type::time_point t0_, t1_;
};

}
#endif
