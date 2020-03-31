#ifndef BIT_FIT_HPP_
#define BIT_FIT_HPP_

#include <vector>
#include <exception>

#include "sim_ds/BitVector.hpp"

#include "bit_field.hpp"
#include "empty_link.hpp"
#include "fft.hpp"
#include "ntt.hpp"
#include "calc.hpp"

namespace bit_fitting {


template <typename BitFitter>
struct bit_fit_traits {
 public:
  using field_type = typename BitFitter::field_type;
};

template <class BitFitter>
class bit_fit {
 public:
  using bit_fitter = BitFitter;
  using traits = bit_fit_traits<bit_fitter>;
  using field_type = typename traits::field_type;
 private:
  bit_fitter bit_fitter_;

 public:
  bit_fit() = default;

  template <typename T>
  size_t find(const field_type& field, const std::vector<T>& pattern, size_t initial_pos=0) const {
    return bit_fitter_(field, pattern, initial_pos);
  }

  field_type field(default_bit_field* default_bf) const {
	if constexpr (std::is_same_v<field_type, default_bit_field>)
	  return *default_bf;
	else
	  return field_type(default_bf);
  }

};


struct brute_force_bit_fit {
  using field_type = default_bit_field;
  using traits = bit_fit_traits<field_type>;

  template <typename T>
  size_t operator()(const field_type& field, const std::vector<T>& pattern, size_t initial_pos) const {
    const size_t F = field.size();
    for (size_t i = initial_pos; i < F; i++) {
      size_t k = 0;
      for (; k < pattern.size(); ++k) {
        auto index = i + pattern[k];
        if (not (index >= field.size() or field[index]))
          break;
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
    if (front_it.index() == field_type::kDisabledFront)
      return F;
    while ((long long)front_it.index() - (long long)pattern[0] < (long long)initial_pos)
        ++front_it;
    do {
      auto front = (long long)front_it.index() - (long long)pattern[0];
      size_t k = 1;
      for (; k < pattern.size(); ++k) {
        auto index = front + pattern[k];
        if (not (index >= field.size() or field.is_empty_at(index)))
          break;
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

    auto get_word = [&](size_t b) -> uint64_t {
      if (b >= (F-1)/64+1)
        return (uint64_t)-1ull;
      return *(field.data() + b);
    };
    auto get_target_word = [&](size_t p) -> uint64_t {
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


template <typename Vec>
struct convolutable_vector_behavior : Vec {
  convolutable_vector_behavior() = default;
  explicit convolutable_vector_behavior(default_bit_field* bf) {
    Vec::resize(bf->size());
    std::transform(bf->begin(), bf->end(), Vec::begin(), [&](bool v) {return v?0:1;});
  }
};

template <class Transformer, typename Value, class Eval>
struct convolution_bit_fit {
  using field_type = default_bit_field;
  using traits = bit_fit_traits<field_type>;
  using transformer_type = Transformer;
  using polynomial_type = typename transformer_type::polynomial_type;

  Eval eval;

  template <typename T>
  size_t operator()(const field_type& field, const std::vector<T>& pattern, size_t initial_pos) const {
    T max_p = 0, min_p = std::numeric_limits<T>::max();
    for (auto p : pattern) {
      max_p = std::max(max_p, p);
      min_p = std::min(min_p, p);
    }
    const size_t m = max_p-min_p+1;
    transformer_type transformer(calc::greater_eq_pow2(2 * m - 1));
    const size_t poly_size = transformer.n();
    const size_t shifted_initial_pos = initial_pos + min_p;

    polynomial_type pattern_poly_rev_trans(poly_size);
    for (auto p : pattern)
      pattern_poly_rev_trans[(poly_size-(p-min_p))%poly_size] = 1;

    transformer.inplace_transform(pattern_poly_rev_trans);
    auto convolute_conflicts = [&](int block, polynomial_type& poly) {
      if (block*poly_size >= field.size()) {
        fill(poly.begin(), poly.end(), 0);
        return;
      }
      std::transform(field.begin()+block*poly_size, field.begin()+(block+1)*poly_size, poly.begin(), [](bool v) {return v?0:1;});
      transformer.inplace_transform(poly.begin(), poly.end());
      for (size_t i = 0; i < poly_size; i++)
        *(poly.begin()+i) *= pattern_poly_rev_trans[i];
      transformer.inplace_inverse_transform(poly.begin(), poly.end());
    };

    { // closure of b
      std::array<polynomial_type, 2> convoluted = {polynomial_type(poly_size), polynomial_type(poly_size)};
      size_t b = shifted_initial_pos/m;
      convolute_conflicts(b, convoluted[b%2]);
      const size_t num_blocks = (field.size()-1)/m+1;
      for (; b < num_blocks; b++) {
        const size_t offset = b*m;
        const size_t idc = b%2;
        const size_t idn = (b+1)%2;
        convolute_conflicts(b+1, convoluted[idn]);
        size_t pos = b == shifted_initial_pos/m ? shifted_initial_pos%m : 0;
        for (; pos < m; pos++) {
          auto cnt_conflict = eval(convoluted[idc][pos]) + eval(convoluted[idn][poly_size-(int)m+pos]);
          if (cnt_conflict == 0) {
            return offset + pos - min_p;
          }
        }
      }
    }

    return field.size();
  }

};

struct eval_complex_to_integral {
  long long operator()(complex_t c) const { return (long long)(c.real()+0.125); }
};
using convolution_fft_bit_fit = convolution_bit_fit<Fft, complex_t, eval_complex_to_integral>;


template <typename T>
struct eval_modint {
  long long operator()(T x) const { return x.val(); }
};
using convolution_ntt_bit_fit = convolution_bit_fit<Ntt<>, Ntt<>::modint_type, eval_modint<Ntt<>::modint_type>>;

}

#endif //BIT_FIT_HPP_
