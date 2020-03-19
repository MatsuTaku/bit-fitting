#ifndef BIT_FITTING_INCLUDE_CALC_HPP_
#define BIT_FITTING_INCLUDE_CALC_HPP_

namespace bit_fitting::calc {

size_t greater_eq_pow2(size_t x) {
  return x == 0 ? 1 : 1ull << (64 - bo::clz_u64(x-1));
}

size_t log_n(size_t x) {
  assert(bo::popcnt_u64(x) == 1);
  return bo::ctz_u64(x);
}

}

#endif //BIT_FITTING_INCLUDE_CALC_HPP_
