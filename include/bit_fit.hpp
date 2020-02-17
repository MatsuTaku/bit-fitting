#ifndef BIT_FIT_HPP_
#define BIT_FIT_HPP_

#include <vector>

#include "sim_ds/BitVector.hpp"

#include "fft.hpp"

namespace bit_fitting {

template <class BitFitter>
class bit_fit {
  using _bit_fitter = BitFitter;
  _bit_fitter bit_fitter_;
 public:
  bit_fit() = default;

  template <typename BitSequence, typename T>
  size_t find(const BitSequence& field, const std::vector<T>& pattern, size_t initial_pos=0) const {
    return bit_fitter_(field, pattern, initial_pos);
  }
};


struct brute_force_bit_fit {
  template <typename BitSequence, typename T>
  size_t operator()(const BitSequence& field, const std::vector<T>& pattern, size_t initial_pos) const {
    const size_t F = field.size();
    for (size_t i = initial_pos; i < F; i++) {
      size_t k = 0;
      while (k < pattern.size() and (i + pattern[k] >= field.size() or field[i + pattern[k]])) {
        k++;
      }
      if (k == pattern.size()) {
        return i;
      }
    }
    return F;
  }
};


struct bit_parallel_bit_fit {
  template <typename BitSequence, typename T>
  size_t operator()(const BitSequence& field, const std::vector<T>& pattern, size_t initial_pos) const {
    const size_t F = field.size();
    auto get_word = [&](size_t b) {
      if (b >= (F-1)/64+1)
        return (uint64_t)-1ull;
      return *(field.data() + b);
    };
    auto get_target_word = [&](size_t p) {
      uint64_t word = get_word(p/64) >> (p%64);
      if (p%64 != 0)
        word |= get_word(p/64 + 1) << (64-p%64);
      return word;
    };
    for (size_t b = 0; b < (F-1)/64+1; b++) {
      auto front = b * 64;
      uint64_t mask = -1ull;
      for (auto p : pattern) {
        uint64_t word = get_target_word(front + p);
        mask &= word;
        if (mask == 0)
          break;
      }
      if (mask != 0) {
        return front + bo::ctz_u64(mask);
      }
    }
    return F;
  }
};


struct fft_bit_fit {
  template <typename BitSequence, typename T>
  size_t operator()(const BitSequence& field, const std::vector<T>& pattern, size_t initial_pos) const {
    const size_t F = field.size();
    const size_t P = *max_element(pattern.begin(), pattern.end())+1;
    const size_t poly_size = calc::upper_pow2(F+P-1);
    std::vector<complex_t> field_poly(poly_size, 0);
    for (size_t i = 0; i < F; i++)
      field_poly[i] = field[i] ? 0 : 1;

    Polynomial pattern_poly_rev(poly_size, 0);
    for (auto p : pattern)
      pattern_poly_rev[(poly_size-p)%poly_size] = 1;

    multiply_polynomial_inplace(field_poly.begin(), field_poly.end(), pattern_poly_rev.begin(), pattern_poly_rev.end());

    for (size_t i = initial_pos; i < F; i++) {
      if (size_t(field_poly[i].real()+0.1) == 0)
        return i;
    }
    return F;
  }
};

}

#endif //BIT_FIT_HPP_
