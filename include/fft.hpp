#include <vector>
#include <iterator>
#include <complex>
#include <cmath>
#include <algorithm>
#include <iomanip>

#include "bo.hpp"

#ifndef BIT_FITTING_INCLUDE_FFT_HPP_
#define BIT_FITTING_INCLUDE_FFT_HPP_

namespace bit_fitting {


namespace calc {

size_t upper_pow2(size_t x) {
  return x == 0 ? 0 : 1ull << (64 - bo::clz_u64(x-1));
}

size_t log_n(size_t x) {
  assert(bo::popcnt_u64(x) == 1);
  return bo::ctz_u64(x);
}

}

using complex_t = std::complex<double>;

/* FFT(Fast Fourier Transform) - Stockham algorithm */
template <bool Odd, long long Sign, typename It>
void _fft(size_t len, size_t stride, It x_begin, It y_begin) {
  if (len == 2) {
    auto z_begin = Odd ? y_begin : x_begin;
    for (size_t q = 0; q < stride; q++) {
      const auto a = *(x_begin + q + 0);
      const auto b = *(x_begin + q + stride);
      *(z_begin + q + 0) = a + b;
      *(z_begin + q + stride) = a - b;
    }

  } else { // len >= 4
    const size_t m = len/2;
    const double theta0 = M_PI*2/len;
    for (size_t p = 0; p < m; p++) {
      const complex_t wp = {cos(p*theta0), sin(p*theta0)*Sign};
      for (size_t q = 0; q < stride; q++) {
        const complex_t a = *(x_begin + q + stride*(p + 0));
        const complex_t b = *(x_begin + q + stride*(p + m));
        *(y_begin + q + stride*(2*p + 0)) = a + b;
        *(y_begin + q + stride*(2*p + 1)) = (a - b) * wp;
      }
    }
    _fft<not Odd, Sign>(len/2, 2*stride, y_begin, x_begin);

  }
}

template <long long Sign, typename It>
void _sixstep_fft(size_t log_n, It x_begin, It y_begin) {
  const size_t N = 1ull << log_n;
  const size_t n = 1ull << (log_n/2); // N = n*n
  for (size_t k = 0; k < n; k++) { // transpose x
    for (size_t p = k+1; p < n; p++) {
      std::swap(*(x_begin + p + k*n), *(x_begin + k + p*n));
    }
  }
  for (size_t p = 0; p < n; p++) // FFT all p-line of x
    _fft<false, Sign>(n, 1, x_begin + p*n, y_begin + p*n);
  for (size_t p = 0; p < n; p++) { // multiply twiddle factor and transpose x
    const double theta0 = M_PI*2*p/N;
    for (size_t k = p; k < n; k++) {
      const double theta = k*theta0;
      const complex_t wkp = {cos(theta), sin(theta)*Sign};
      if (k == p) {
        *(x_begin + p + p*n) *= wkp;
      } else {
        const complex_t a = *(x_begin + k + p*n) * wkp;
        const complex_t b = *(x_begin + p + k*n) * wkp;
        *(x_begin + k + p*n) = b;
        *(x_begin + p + k*n) = a;
      }
    }
  }
  for (size_t k = 0; k < n; k++) // FFT all k-line of x
    _fft<false, Sign>(n, 1, x_begin + k*n, y_begin + k*n);
  for (size_t k = 0; k < n; k++) { // transpose x
    for (size_t p = k+1; p < n; p++) {
      std::swap(*(x_begin + p + k*n), *(x_begin + k + p*n));
    }
  }
}

template <long long Sign, typename It>
void _eightstep_fft(size_t log_n, It x_begin, It y_begin) {
  const size_t n = 1ull << log_n;
  const size_t m = n/2;
  const double theta0 = M_PI*2/n;
  for (size_t p = 0; p < m; p++) {
    const double theta = p*theta0;
    const complex_t wp = {cos(theta), sin(theta)*Sign};
    const complex_t a = *(x_begin + p + 0);
    const complex_t b = *(x_begin + p + m);
    *(y_begin + p + 0) = a + b;
    *(y_begin + p + m) = (a - b) * wp;
  }
  _sixstep_fft<Sign>(log_n - 1, y_begin + 0, x_begin + 0);
  _sixstep_fft<Sign>(log_n - 1, y_begin + m, x_begin + m);
  for (size_t p = 0; p < m; p++) {
    *(x_begin + 2*p + 0) = *(y_begin + p + 0);
    *(x_begin + 2*p + 1) = *(y_begin + p + m);
  }
}

template <long long Sign=-1, typename It>
void fft(It vec_begin, It vec_end, It aux_begin, It aux_end) {
  auto len = vec_end - vec_begin;
  assert(bo::popcnt_u64(len) == 1);
  assert(aux_end - aux_begin == len);
  auto log_n = calc::log_n(len);
  if (len <= 1) {}
  else if (len == 2)
    _fft<false, Sign>(len, 1, vec_begin, aux_begin);
  else if ((log_n & 1) == 0)
    _sixstep_fft<Sign>(log_n, vec_begin, aux_begin);
  else
    _eightstep_fft<Sign>(log_n, vec_begin, aux_begin);
}

template <typename It>
void fft(It vec_begin, It vec_end) {
  auto len = vec_end - vec_begin;
  assert(bo::popcnt_u64(len) == 1);
  std::vector<complex_t> aux(len, 0);
  fft(vec_begin, vec_end, aux.begin(), aux.end());
}

void fft(std::vector<complex_t>& vec) {
  auto len = calc::upper_pow2(vec.size());
  vec.resize(len, 0);
  fft(vec.begin(), vec.end());
}

template <typename It>
void inverse_fft(It vec_begin, It vec_end, It aux_begin, It aux_end) {
  auto len = vec_end - vec_begin;
  assert(bo::popcnt_u64(len) == 1);
  assert(aux_end - aux_begin == len);
  fft<1>(vec_begin, vec_end, aux_begin, aux_end);
  for (auto it = vec_begin; it != vec_end; ++it)
    *it /= len;
}

template <typename It>
void inverse_fft(It vec_begin, It vec_end) {
  auto len = vec_end - vec_begin;
  assert(bo::popcnt_u64(len) == 1);
  std::vector<complex_t> aux(len, 0);
  inverse_fft(vec_begin, vec_end, aux.begin(), aux.end());
}

void inverse_fft(std::vector<complex_t>& vec) {
  auto len = calc::upper_pow2(vec.size());
  vec.resize(len, 0);
  inverse_fft(vec.begin(), vec.end());
}


template <typename It>
void multiply_polynomial(It g_begin, It g_end, It h_begin, It h_end, It f_begin, It f_end) {
  fft(g_begin, g_end);
  fft(h_begin, h_end);
  auto len = g_end - g_begin;
  for (size_t i = 0 ; i < len; i++) {
    *(f_begin+i) = *(g_begin+i) * *(h_begin+i);
  }
  inverse_fft(f_begin, f_end);
}

void multiply_polynomial(std::vector<complex_t>& g, std::vector<complex_t>& h, std::vector<complex_t>& f) {
  auto len = calc::upper_pow2(g.size()+h.size()-1);
  g.resize(len, 0);
  h.resize(len, 0);
  f.resize(len, 0);
  multiply_polynomial(g.begin(), g.end(), h.begin(), h.end(), f.begin(), f.end());
}


template <typename It>
void multiply_polynomial_inplace(It g_begin, It g_end, It h_begin, It h_end) {
  fft(g_begin, g_end);
  fft(h_begin, h_end);
  auto len = g_end - g_begin;
  for (auto git = g_begin, hit = h_begin; git != g_end; ++git, ++hit)
    *git *= *hit;
  inverse_fft(g_begin, g_end);
}

void multiply_polynomial_inplace(std::vector<complex_t>& g, std::vector<complex_t>& h) {
  auto len = calc::upper_pow2(g.size()+h.size()-1);
  g.resize(len, 0);
  h.resize(len, 0);
  std::cout << "len: " << len << std::endl;
  multiply_polynomial_inplace(g.begin(), g.end(), h.begin(), h.end());
}


class Polynomial : public std::vector<complex_t> {
  using _base = std::vector<complex_t>;
 public:
  using base_type = _base;

  Polynomial() = default;

  Polynomial(base_type&& vec) : _base(std::move(vec)) {}

  Polynomial(size_t size) : _base(size, 0) {}
  Polynomial(size_t size, complex_t initial_val) : _base(size, initial_val) {}

  Polynomial operator*(const Polynomial& x) const {
    base_type g = (_base&)*this;
    base_type h = (_base&)x;
    multiply_polynomial_inplace(g, h);
    return Polynomial(std::move(g));
  }

  Polynomial& operator*=(Polynomial& x) {
    multiply_polynomial_inplace((_base&)*this, (_base&)x);
    return *this;
  }

  Polynomial& operator*=(const Polynomial& x) {
    auto h = x;
    return *this *= x;
  }
};

}

#endif //BIT_FITTING_INCLUDE_FFT_HPP_
