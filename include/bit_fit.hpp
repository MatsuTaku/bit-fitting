#ifndef BIT_FIT_HPP_
#define BIT_FIT_HPP_

#include <vector>
#include <exception>

#include "sim_ds/BitVector.hpp"

#include "bit_field.hpp"
#include "empty_link.hpp"
#include "convolution.hpp"
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
    if (front_it.index() == field_type::kDisabledFront)
      return F;
    while ((long long)front_it.index() - (long long)pattern[0] < (long long)initial_pos)
        ++front_it;
    do {
      auto front = (long long)front_it.index() - (long long)pattern[0];
      size_t k = 1;
      auto index = front + pattern[k];
      while (k < pattern.size() and (index >= F or field.is_empty_at(index))) {
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
  using convolution_class = convolution<Fft>;
  using polynomial_type = convolution_class::polynomial_type;

  long long integer(complex_t c) const { return (long long)(c.real()+0.125); }

  template <typename T>
  size_t operator()(const field_type& field, const std::vector<T>& pattern, size_t initial_pos) const {
	const size_t m = *max_element(pattern.begin(), pattern.end())+1;
	convolution_class convolutor(2*m-1);
	const size_t poly_size = convolutor.n();
	polynomial_type pattern_poly_rev(poly_size, 0);
	for (auto p : pattern)
	  pattern_poly_rev[(poly_size-p)%poly_size] = 1;

	const size_t num_blocks = (field.size()-1)/m+1;
	polynomial_type field_poly[2] = {polynomial_type(poly_size, 0), polynomial_type(poly_size, 0)};
	polynomial_type convoluted[2];
	{
	  size_t b = initial_pos/m;
	  for (size_t i = 0; i < m; i++)
		field_poly[b%2][i] = field[i]?0:1;
	  convoluted[b%2] = convolutor(field_poly[b%2], pattern_poly_rev);
	  for (; b < num_blocks; b++) {
		const size_t offset = b*m;
		const size_t idc = b%2;
		const size_t idn = (b+1)%2;
		field_poly[idn].assign(poly_size, 0);
		for (size_t i = 0; i < m and offset+m+i < field.size(); i++)
		  field_poly[idn][i] = field[offset+m+i]?0:1;
		convoluted[idn] = convolutor(field_poly[idn], pattern_poly_rev);
		size_t pos = b == initial_pos/m ? initial_pos%m : 0;
		for (; pos < m; pos++) {
		  auto cnt_conflict = integer(convoluted[idc][pos]) + integer(convoluted[idn][poly_size-(int)m+pos]);
		  if (cnt_conflict == 0) {
//			std::cout << pos << std::endl;
//			for (auto c : convoluted[idc])
//			  std::cout << integer(c) << " ";
//			for (auto c : convoluted[idn])
//			  std::cout << integer(c) << " ";
//			std::cout << std::endl;
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
  using convolution_class = convolution<Ntt<>>;
  using polynomial_type = convolution_class::polynomial_type;

  template <typename T>
  size_t operator()(const field_type& field, const std::vector<T>& pattern, size_t initial_pos) const {
	const size_t m = *max_element(pattern.begin(), pattern.end())+1;
	convolution_class convolutor(2*m-1);
	const size_t poly_size = convolutor.n();
	polynomial_type pattern_poly_rev(poly_size, 0);
	for (auto p : pattern)
	  pattern_poly_rev[(poly_size-p)%poly_size] = 1;

	const size_t num_blocks = (field.size()-1)/m+1;
	polynomial_type field_poly[2] = {polynomial_type(poly_size, 0), polynomial_type(poly_size, 0)};
	polynomial_type convoluted[2];
	{
	  size_t b = initial_pos/m;
	  for (size_t i = 0; i < m; i++)
		field_poly[b%2][i] = field[i]?0:1;
	  convoluted[b%2] = convolutor(field_poly[b%2], pattern_poly_rev);
	  for (; b < num_blocks; b++) {
		const size_t offset = b*m;
		const size_t idc = b%2;
		const size_t idn = (b+1)%2;
		field_poly[idn].assign(poly_size, 0);
		for (size_t i = 0; i < m and offset+m+i < field.size(); i++)
		  field_poly[idn][i] = field[offset+m+i]?0:1;
		convoluted[idn] = convolutor(field_poly[idn], pattern_poly_rev);
		size_t pos = b == initial_pos/m ? initial_pos%m : 0;
		for (; pos < m; pos++) {
		  auto cnt_conflict = convoluted[idc][pos].val() + convoluted[idn][poly_size-(int)m+pos].val();
		  if (cnt_conflict == 0) {
//			std::cout << pos << std::endl;
//			for (auto c : convoluted[idc])
//			  std::cout << c << " ";
//			for (auto c : convoluted[idn])
//			  std::cout << c << " ";
//			std::cout << std::endl;
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
