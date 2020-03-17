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

constexpr ntt_prime_tuple get_ntt_prime_id(size_t size) {
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
  size_t size_;
  size_t log_n_;
  std::vector<modint_type> zeta_;
  std::vector<modint_type> izeta_;

 public:
  Ntt(size_t size) : size_(size) {
	assert((size&(-size)) == size);
	log_n_ = 64-bo::clz_u64(size-1);
	assert(bo::ctz_u64(Mod-1) >= log_n_);
	modint_type w = pow(modint_type{PrimitiveRoot}, (Mod-1)>>log_n_);
	assert(pow(w, size) == 1);

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

  polynomial_type transform(const polynomial_type& f) const {
	auto ff = f;
	ff.resize(size_, 0);
	_transform(ff);
	return ff;
  }

  void inplace_transform(polynomial_type& f) const {
    _transform(f);
  }

  polynomial_type inverse_transform(const polynomial_type& f) const {
	auto ff = f;
	ff.resize(size_, 0);
	_inverse_transform(ff);
	return ff;
  }

  void inplace_inverse_transform(polynomial_type& f) const {
	_inverse_transform(f);
  }

  uint64_t bitreverse(uint64_t x) const {
	x = (x<<32) | (x>>32);
	x = ((x&0x0000FFFF0000FFFF)<<16) | ((x>>16)&0x0000FFFF0000FFFF);
	x = ((x&0x00FF00FF00FF00FF)<<8) | ((x>>8)&0x00FF00FF00FF00FF);
	x = ((x&0x0F0F0F0F0F0F0F0F)<<4) | ((x>>4)&0x0F0F0F0F0F0F0F0F);
	x = ((x&0x3333333333333333)<<2) | ((x>>2)&0x3333333333333333);
	x = ((x&0x5555555555555555)<<1) | ((x>>1)&0x5555555555555555);
	return x >> (64-log_n_);
  }

 private:
  template <bool Forward>
  void _transform_impl(polynomial_type& f) const {
    // Iterative bitreverse
    for (size_t i = 0; i < size_; i++) {
      auto j = bitreverse(i);
      if (i >= j)
        continue;
      std::swap(f[i], f[j]);
    }
    // Cooly-Tukey Algorithm
    for (size_t log_m = 0; log_m < log_n_; log_m++) {
      auto m = 1ull<<log_m;
      for (size_t chank = 0; chank < size_; chank += 2*m) {
        modint_type w = 1;
        for (size_t i = 0; i < m; i++) {
          auto a = f[chank + i + 0];
          auto b = f[chank + i + m] * w;
          f[chank + i + 0] = a + b;
          f[chank + i + m] = a - b;
          w *= Forward ? zeta_[log_m] : izeta_[log_m];
        }
      }
    }
  }

  void _transform(polynomial_type& f) const {
    _transform_impl<true>(f);
  }

  void _inverse_transform(polynomial_type& f) const {
	_transform_impl<false>(f);
	for (auto& x : f)
	  x /= size_;
  }

};

}

#endif //BIT_FITTING_INCLUDE_NTT_HPP_