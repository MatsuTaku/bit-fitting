#ifndef BIT_FIT_HPP_
#define BIT_FIT_HPP_

#include <vector>
#include <exception>

#include "sim_ds/BitVector.hpp"

#include "bit_field.hpp"
#include "empty_link.hpp"
#include "fft.hpp"

namespace bit_fitting {

template <class BitFitter>
class bit_fit {
 public:
  using bit_fitter = BitFitter;
  using field_type = typename BitFitter::field_type;
 private:
  bit_fitter bit_fitter_;

 public:
  bit_fit() = default;

  template <typename T>
  size_t find(const field_type& field, const std::vector<T>& pattern, size_t initial_pos=0) const {
    return bit_fitter_(field, pattern, initial_pos);
  }

  field_type field(default_bit_field* default_bf) {
	if constexpr (std::is_same_v<field_type, default_bit_field>)
	  return *default_bf;
	else
	  return field_type(default_bf);
  }

};


template <typename FieldType = default_bit_field>
struct bit_fit_traits {
 public:
  using field_type = FieldType;
};


struct brute_force_bit_fit {
  using field_type = default_bit_field;
  using traits = bit_fit_traits<field_type>;

  template <typename T>
  size_t operator()(const field_type& field, const std::vector<T>& pattern, size_t initial_pos) const {
    const size_t F = field.size();
    for (size_t i = initial_pos; i < F; i++) {
      size_t k = 0;
	  size_t index = i + pattern[k];
      while (k < pattern.size() and (index >= field.size() or field[index])) {
		index = i + pattern[++k];
      }
      if (k == pattern.size()) {
        return i;
      }
    }
    return F;
  }
};


struct empty_link_bit_fit {
  using field_type = empty_linked_bit_vector<true>;
  using traits = bit_fit_traits<field_type>;

  template <typename T>
  size_t operator()(const field_type& field, const std::vector<T>& pattern, size_t initial_pos) const {
    const size_t F = field.size();
    auto front_it = field.el_begin();
    do {
      auto front = front_it.index() - pattern[0];
      size_t k = 1;
      size_t index = front + pattern[k];
      while (k < pattern.size() and (index >= field.size() or field[index])) {
        index = front + pattern[++k];
      }
      if (k == pattern.size()) {
        return front;
      }
    } while (++front_it != field.el_end());
    return F;
  }
};


struct bit_parallel_bit_fit {
  using field_type = default_bit_field;
  using traits = bit_fit_traits<field_type>;

  template <typename T>
  size_t operator()(const field_type& field, const std::vector<T>& pattern, size_t initial_pos) const {
    const size_t F = field.size();

    auto get_word = [&](size_t b) {
      if (b >= (F-1)/64+1)
        return (uint64_t)-1ull;
      return *(field.data() + b);
    };
    auto get_target_word = [&](size_t p) {
      if (p%64 == 0)
        return get_word(p/64) >> (p%64);
      else
        return (get_word(p/64) >> (p%64)) | (get_word(p/64+1) << (64-p%64));
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
  using field_type = default_bit_field;
  using traits = bit_fit_traits<field_type>;

  template <typename T>
  size_t operator()(const field_type& field, const std::vector<T>& pattern, size_t initial_pos) const {
    const size_t F = field.size();
    const size_t P = *max_element(pattern.begin(), pattern.end())+1;

    auto start = clock();

    const size_t poly_size = calc::upper_pow2(F+P-1);

    polynomial_vector field_poly(poly_size, 0);
    for (size_t i = 0; i < F; i++)
      field_poly[i] = field[i] ? 0 : 1;

    polynomial_vector pattern_poly_rev(poly_size, 0);
    for (auto p : pattern)
      pattern_poly_rev[(poly_size-p)%poly_size] = 1;

    std::cout << "  initialize arrays: " << double(clock() - start)/1000000 << "s" << std::endl;
    start = clock();

    multiply_polynomial_inplace(field_poly.begin(), field_poly.end(), pattern_poly_rev.begin(), pattern_poly_rev.end());

    std::cout << "  multiply polynomial: " << double(clock() - start)/1000000 << "s" << std::endl;
    start = clock();

    for (size_t i = initial_pos; i < F; i++) {
      if (size_t(field_poly[i].real()+0.1) == 0) {
        std::cout << "  find min answer: " << double(clock() - start)/1000000 << "s" << std::endl;

        return i;
      }
    }

    return F;
  }

};

}

#endif //BIT_FIT_HPP_
