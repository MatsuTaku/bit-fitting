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


struct convolution_fft_bit_fit {
  using field_type = default_bit_field;
  using traits = bit_fit_traits<field_type>;
  using transformer_type = Fft;
//  using convolution_class = convolution<transformer_type>;
  using polynomial_type = transformer_type::polynomial_type;

  long long integer(complex_t c) const { return (long long)(c.real()+0.125); }

  template <typename T>
  size_t operator()(const field_type& field, const std::vector<T>& pattern, size_t initial_pos) const {
    const size_t m = *max_element(pattern.begin(), pattern.end())+1;
    transformer_type transformer(calc::greater_eq_pow2(2 * m - 1));
    const size_t poly_size = transformer.n();

    auto set_field_block = [&](size_t block, auto& vec) {
      size_t offset = block*m;
      size_t i = 0;
      if (field.size()-offset >= m) {
        for (; i < m; i++)
          vec[i] = field[offset+i] ? 0 : 1;
      } else {
        for (; offset+i < field.size(); i++)
          vec[i] = field[offset+i] ? 0 : 1;
        for (; i < m; i++)
          vec[i] = 0;
      }
    };
    polynomial_type pattern_poly_rev_trans(poly_size);
    for (auto p : pattern)
      pattern_poly_rev_trans[(poly_size-p)%poly_size] = 1;

    transformer.inplace_transform(pattern_poly_rev_trans);
    auto convolute_conflicts = [&](auto field_block, auto& dst) {
      transformer.transform(field_block, dst);
      for (size_t i = 0; i < poly_size; i++)
        dst[i] *= pattern_poly_rev_trans[i];
      transformer.inplace_inverse_transform(dst);
    };

    const size_t num_blocks = (field.size()-1)/m+1;
    polynomial_type field_poly[2] = {polynomial_type(poly_size), polynomial_type(poly_size)};
    polynomial_type convoluted[2];

    { // closure of b
      size_t b = initial_pos/m;
      set_field_block(b, field_poly[b%2]);
      convolute_conflicts(field_poly[b%2], convoluted[b%2]);
      for (; b < num_blocks; b++) {
        const size_t offset = b*m;
        const size_t idc = b%2;
        const size_t idn = (b+1)%2;
        set_field_block(b+1, field_poly[idn]);
        convolute_conflicts(field_poly[idn], convoluted[idn]);
        size_t pos = b == initial_pos/m ? initial_pos%m : 0;
        for (; pos < m; pos++) {
          auto cnt_conflict = integer(convoluted[idc][pos]) + integer(convoluted[idn][poly_size-(int)m+pos]);
          if (cnt_conflict == 0) {
            return offset + pos;
          }
        }
      }
    }

    return field.size();
  }

};


struct convolution_ntt_bit_fit {
  using field_type = default_bit_field;
  using traits = bit_fit_traits<field_type>;
  using transformer_type = Ntt<>;
//  using convolution_class = convolution<transformer_type>;
  using polynomial_type = transformer_type::polynomial_type;

  template <typename T>
  size_t operator()(const field_type& field, const std::vector<T>& pattern, size_t initial_pos) const {
	const size_t m = *max_element(pattern.begin(), pattern.end())+1;
	transformer_type transformer(calc::greater_eq_pow2(2 * m - 1));
	const size_t poly_size = transformer.n();

	auto set_field_block = [&](size_t block, auto& vec) {
	  size_t offset = block*m;
	  size_t i = 0;
	  if (field.size()-offset >= m) {
		for (; i < m; i++)
		  vec[i] = field[offset+i] ? 0 : 1;
	  } else {
		for (; offset+i < field.size(); i++)
          vec[i] = field[offset+i] ? 0 : 1;
		for (; i < m; i++)
		  vec[i] = 0;
	  }
	};
	polynomial_type pattern_poly_rev_trans(poly_size);
	for (auto p : pattern)
	  pattern_poly_rev_trans[(poly_size-p)%poly_size] = 1;

	transformer.inplace_transform(pattern_poly_rev_trans);
	auto convolute_conflicts = [&](auto field_block, auto& dst) {
	  transformer.transform(field_block, dst);
	  for (size_t i = 0; i < poly_size; i++)
        dst[i] *= pattern_poly_rev_trans[i];
	  transformer.inplace_inverse_transform(dst);
	};

	const size_t num_blocks = (field.size()-1)/m+1;
	std::array<polynomial_type, 2> field_poly = {polynomial_type(poly_size), polynomial_type(poly_size)};
    std::array<polynomial_type, 2> convoluted;

	{ // closure of b
	  size_t b = initial_pos/m;
	  set_field_block(b, field_poly[b%2]);
	  convolute_conflicts(field_poly[b%2], convoluted[b%2]);
	  for (; b < num_blocks; b++) {
		const size_t offset = b*m;
		const size_t idc = b%2;
		const size_t idn = (b+1)%2;
		set_field_block(b+1, field_poly[idn]);
		convolute_conflicts(field_poly[idn], convoluted[idn]);
		size_t pos = b == initial_pos/m ? initial_pos%m : 0;
		for (; pos < m; pos++) {
		  auto cnt_conflict = convoluted[idc][pos].val() + convoluted[idn][poly_size-(int)m+pos].val();
          if (cnt_conflict == 0) {
            return offset + pos;
		  }
		}
	  }
	}

	return field.size();
  }

};

}

#endif //BIT_FIT_HPP_
