#ifndef BIT_FITTING_INCLUDE_NTT_HPP_
#define BIT_FITTING_INCLUDE_NTT_HPP_

#include <array>
#include <vector>
#include <bitset>

#include "bo.hpp"

#include "modint.hpp"

namespace bit_fitting {

struct ntt_prime_tuple {
  unsigned long long prime;
  unsigned primitive_root;
  unsigned ctz;
  constexpr ntt_prime_tuple(unsigned long long p, unsigned r, unsigned c) : prime(p), primitive_root(r), ctz(c) {}
};

constexpr std::array<ntt_prime_tuple, 18> kNttPrimes = {
	ntt_prime_tuple{1053818881, 7, 20}, // 2^20 * 3 * 5 * 67 + 1
	ntt_prime_tuple{1051721729, 6, 20}, // 2^20 * 17 * 59 + 1
	ntt_prime_tuple{1045430273, 3, 20}, // 2^20 * 997 + 1
	ntt_prime_tuple{1007681537, 3, 20}, // 2^20 * 31^2 + 1
	ntt_prime_tuple{976224257, 3, 20},  // 2^20 * 7^2 * 19 + 1
	ntt_prime_tuple{1012924417, 5, 21}, // 2^21 * 3 * 7 * 23 + 1
	ntt_prime_tuple{1004535809, 3, 21}, // 2^21 * 479 + 1
	ntt_prime_tuple{975175681, 17, 21}, // 2^21 * 3 * 5 * 31 + 1
	ntt_prime_tuple{962592769, 7, 21},  // 2^21 * 3^3 * 17 + 1
	ntt_prime_tuple{950009857, 7, 21},  // 2^21 * 4 * 151 + 1
	ntt_prime_tuple{924844033, 5, 21},  // 2^21 * 3^2 * 7^2 + 1
	ntt_prime_tuple{985661441, 3, 22},  // 2^22 * 5 * 47 + 1
	ntt_prime_tuple{943718401, 7, 22},  // 2^22 * 3^2 * 5^2 + 1
	ntt_prime_tuple{935329793, 3, 22},  // 2^22 * 223 + 1
	ntt_prime_tuple{998244353, 3, 23},  // 2^23 * 7 * 17 + 1
	ntt_prime_tuple{1224736769, 3, 24}, // 2^24 * 73 + 1
	ntt_prime_tuple{167772161, 3, 25},  // 2^25 * 5 + 1
	ntt_prime_tuple{469762049, 3, 26},  // 2^26 * 7 + 1
};

ntt_prime_tuple get_ntt_prime_id(size_t size) {
  if (size == 0)
    return kNttPrimes[0];
  auto ctz = 64-bo::clz_u64(size-1);
  return *std::lower_bound(kNttPrimes.begin(), kNttPrimes.end(), ctz,
  	[](auto it, auto val) {return it.ctz < val;});
}


template <unsigned long long Mod=469762049, unsigned long long PrimitiveRoot=3>
class Ntt {
 public:
  using modint_type = modint<unsigned long long, Mod>;
  using polynomial_type = std::vector<modint_type>;

 private:
  size_t n_;
  size_t log_n_;
  std::vector<modint_type> zeta_;
  std::vector<modint_type> izeta_;

 public:
  explicit Ntt(size_t n) : n_(n) {
	assert((n&(-n)) == n);
	log_n_ = 64-bo::clz_u64(n-1);
	assert(bo::ctz_u64(Mod-1) >= log_n_);
	modint_type w = pow(modint_type{PrimitiveRoot}, (Mod-1)>>log_n_);
	assert(pow(w, n) == 1);

	zeta_.resize(log_n_);
	izeta_.resize(log_n_);
	auto u = modint_type{1}/w;
	auto v = w;
	for (int i = log_n_-1; i >= 0; i--) {
	  zeta_[i] = u;
	  u *= u;
	  izeta_[i] = v;
	  v *= v;
	}
  }

  size_t n() const { return n_; }

  template <typename InputIterator, typename OutputIterator>
  void transform(InputIterator in_begin, InputIterator in_end, OutputIterator out_begin, OutputIterator out_end) const {
    std::copy(in_begin, in_end, out_begin);
    _transform(out_begin, out_end);
  }

  void transform(const polynomial_type& f, polynomial_type& tf) const {
	tf.assign(n_, 0);
	transform(f.begin(), f.end(), tf.begin(), tf.end());
  }

  template <typename Iterator>
  void inplace_transform(Iterator begin, Iterator end) const {
    _transform(begin, end);
  }

  void inplace_transform(polynomial_type& f) const {
    inplace_transform(f.begin(), f.end());
  }

  template <typename InputIterator, typename OutputIterator>
  void inverse_transform(InputIterator in_begin, InputIterator in_end, OutputIterator out_begin, OutputIterator out_end) const {
    std::copy(in_begin, in_end, out_begin);
    _inverse_transform(out_begin, out_end);
  }

  void inverse_transform(const polynomial_type& f, polynomial_type& tf) const {
    tf.assign(n_, 0);
    inverse_transform(f.begin(), f.end(), tf.begin(), tf.end());
  }

  template <typename Iterator>
  void inplace_inverse_transform(Iterator begin, Iterator end) const {
    _inverse_transform(begin, end);
  }

  void inplace_inverse_transform(polynomial_type& f) const {
    inplace_inverse_transform(f.begin(), f.end());
  }

  uint64_t bitreverse(uint64_t x, size_t log_n) const {
    return bo::bitreverse_u64(x) >> (64-log_n);
  }

 private:
  template <bool Forward, typename It>
  void _transform_cooly_tukey(It begin, It end) const {
    // Iterative bitreverse
    size_t n = end-begin;
    size_t log_n = bo::ctz_u64(n);
    for (size_t i = 0; i < n; i++) {
      auto j = bitreverse(i, log_n);
      if (i < j)
        std::swap(*(begin+i), *(begin+j));
    }
    // FFT: Cooly-Tukey
    for (size_t log_m = 0; log_m < log_n; log_m++) {
      auto m = 1ull<<log_m;
      auto zeta = Forward ? zeta_[log_m] : izeta_[log_m];
      for (size_t chunk = 0; chunk < n; chunk += 2*m) {
        modint_type pow_zeta = 1;
        for (size_t i = 0; i < m; i++) {
          auto a = *(begin + chunk + i + 0);
          auto b = *(begin + chunk + i + m) * pow_zeta;
		  *(begin + chunk + i + 0) = a + b;
		  *(begin + chunk + i + m) = a - b;
          pow_zeta *= zeta;
        }
      }
    }
  }

  template <typename Iter>
  void _transform(Iter begin, Iter end) const {
    _transform_cooly_tukey<true>(begin, end);
  }

  template <typename Iter>
  void _inverse_transform(Iter begin, Iter end) const {
	_transform_cooly_tukey<false>(begin, end);
	modint_type div_n = modint_type{1}/n();
	for (auto it = begin; it != end; ++it)
	  *it *= div_n;
  }

};

}

#endif //BIT_FITTING_INCLUDE_NTT_HPP_
