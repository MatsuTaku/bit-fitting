#include <vector>
#include <iterator>
#include <complex>
#include <cmath>
#include <algorithm>
#include <iomanip>

#include "bo.hpp"
#include "boost/align/aligned_allocator.hpp"
#include "fftw3.h"

#include "calc.hpp"

#ifndef BIT_FITTING_INCLUDE_FFT_HPP_
#define BIT_FITTING_INCLUDE_FFT_HPP_

namespace bit_fitting {

/* FFT(Fast Fourier Transform) - Stockham algorithm */

//using complex_t = std::complex<double>;

class complex_t {
 private:
  double re_, im_;
 public:
  complex_t() : re_(0), im_(0) {}
  complex_t(double r) : re_(r), im_(0) {}
  complex_t(double r, double i) : re_(r), im_(i) {}

  double real() const { return re_; }
  double imag() const { return im_; }

  const double* data() const { return &re_; }
  double* data() { return &re_; }

  complex_t operator+(const complex_t& x) const {
    return {re_ + x.re_, im_ + x.im_};
  }
  complex_t& operator+=(const complex_t& x) {
	return *this = *this + x;
  }
  complex_t operator-() const {
	return {-re_, -im_};
  }
  complex_t operator-(const complex_t& x) const {
	return {re_ - x.re_, im_ - x.im_};
  }
  complex_t& operator-=(const complex_t& x) {
	return *this += -x;
  }
  complex_t operator*(const complex_t& x) const {
	return {re_*x.re_ - im_*x.im_, re_*x.im_ + im_*x.re_};
  }
  complex_t& operator*=(const complex_t& x) {
	return *this = *this * x;
  }
  complex_t operator*(double x) const {
	return {re_*x, im_*x};
  }
  complex_t& operator*=(double x) {
	return *this = *this * x;
  }
  friend complex_t operator*(double x, const complex_t& y) {
	return y * x;
  }
  complex_t operator/(double x) const {
	return {re_/x, im_/x};
  }
  complex_t& operator/=(double x) {
	return *this = *this / x;
  }
};

using complex_vector = std::vector<complex_t, boost::alignment::aligned_allocator<complex_t, 32>>;


namespace fft {

#ifdef __AVX__
__m256d mulpz2(const __m256d ab, const __m256d xy) {
  const __m256d aa = _mm256_unpacklo_pd(ab, ab);
  const __m256d bb = _mm256_unpackhi_pd(ab, ab);
  const __m256d yx = _mm256_shuffle_pd(xy, xy, 0b0101);
  return _mm256_addsub_pd(_mm256_mul_pd(aa, xy), _mm256_mul_pd(bb, yx));
}
#endif

// MARK: FFT Implementations using six-eight-step Stockham Algorithm

template <bool Odd, unsigned long long Stride, bool Forward, typename It>
void _fft_standard(size_t len, It x_begin, It y_begin) {
  if (len == 2) {
    auto z_begin = Odd ? y_begin : x_begin;
    for (size_t q = 0; q < Stride; q++) {
      const auto a = *(x_begin + q + 0);
      const auto b = *(x_begin + q + Stride);
      *(z_begin + q + 0) = a + b;
      *(z_begin + q + Stride) = a - b;
    }

  } else if (len >= 4) {
    const size_t m = len/2;
    const double theta0 = M_PI*2/len;
    for (size_t p = 0; p < m; p++) {
      const complex_t wp = {cos(p*theta0), Forward?-sin(p*theta0):sin(p*theta0)};
      for (size_t q = 0; q < Stride; q++) {
        const complex_t a = *(x_begin + q + Stride*(p + 0));
        const complex_t b = *(x_begin + q + Stride*(p + m));
        *(y_begin + q + Stride*(2*p + 0)) = a + b;
        *(y_begin + q + Stride*(2*p + 1)) = (a - b) * wp;
      }
    }
	_fft_standard<!Odd, 2*Stride, Forward>(len / 2, y_begin, x_begin);

  }
}

#ifdef __AVX__
template <bool Odd, unsigned long long Stride, bool Forward, typename It>
void _fft_avx(size_t len, It x_begin, It y_begin) {
  if (len == 2) {
    auto z_begin = Odd ? y_begin : x_begin;
    if constexpr (Stride == 1) {
      const __m128d a = _mm_load_pd(x_begin->data() + 2*0);
      const __m128d b = _mm_load_pd(x_begin->data() + 2*1);
      _mm_store_pd(z_begin->data() + 2*0, _mm_add_pd(a, b));
      _mm_store_pd(z_begin->data() + 2*1, _mm_sub_pd(a, b));
    } else {
      for (size_t q = 0; q < Stride; q += 2) {
        const __m256d a = _mm256_load_pd((x_begin+q + 0)->data());
        const __m256d b = _mm256_load_pd((x_begin+q + Stride)->data());
        _mm256_store_pd((z_begin+q + 0)->data(), _mm256_add_pd(a, b));
        _mm256_store_pd((z_begin+q + Stride)->data(), _mm256_sub_pd(a, b));
      }
    }

  } else if (len >= 4) {
	const size_t m = len/2;
	const double theta0 = M_PI*2/len;
    if constexpr (Stride == 1) {
      for (size_t p = 0; p < m; p++) {
        const complex_t wp = {cos(p*theta0), Forward?-sin(p*theta0):sin(p*theta0)};
        const complex_t a = *(x_begin + p + 0);
        const complex_t b = *(x_begin + p + m);
        *(y_begin + 2*p + 0) = a + b;
        *(y_begin + 2*p + 1) = (a - b) * wp;
      }
    } else {
      for (size_t p = 0; p < m; p++) {
        const double cs = cos(p*theta0);
        const double sn = sin(p*theta0);
        const __m256d wp = _mm256_setr_pd(cs, Forward?-sn:sn, cs, Forward?-sn:sn);
        for (size_t q = 0; q < Stride; q += 2) {
          const __m256d a = _mm256_load_pd((x_begin + q + Stride*(p + 0))->data());
		  const __m256d b = _mm256_load_pd((x_begin + q + Stride*(p + m))->data());
		  _mm256_store_pd((y_begin + q + Stride*(2*p + 0))->data(), _mm256_add_pd(a, b));
		  _mm256_store_pd((y_begin + q + Stride*(2*p + 1))->data(), mulpz2(wp, _mm256_sub_pd(a, b)));
        }
      }
    }
    _fft_avx<!Odd, 2*Stride, Forward>(len/2, y_begin, x_begin);

  }
}

#endif

template <bool Odd, bool Forward, typename It>
void _fft_multi_column(size_t len, It x_begin, It y_begin) {
#ifdef __AVX__
  _fft_avx<Odd, 1, Forward>(len, x_begin, y_begin);
#else
  _fft_standard<Odd, 1, Forward>(len, x_begin, y_begin);
#endif
}

template <bool Forward, typename It>
void _sixstep_fft(size_t log_n, It x_begin, It y_begin) {
  const size_t N = 1ull << log_n;
  const size_t n = 1ull << (log_n/2); // N = n*n
  for (size_t k = 0; k < n; k++) { // transpose x
    for (size_t p = k+1; p < n; p++) {
      std::swap(*(x_begin + p + k*n), *(x_begin + k + p*n));
    }
  }
  for (size_t p = 0; p < n; p++) // FFT all p-line of x
    _fft_multi_column<false, Forward>(n, x_begin + p * n, y_begin + p * n);
  for (size_t p = 0; p < n; p++) { // multiply twiddle factor and transpose x
    const double theta0 = M_PI*2*p/N;
	{ // k = p
	  const double theta = p*theta0;
	  const complex_t wkp = {cos(theta), Forward?-sin(theta):sin(theta)};
	  *(x_begin + p + p*n) *= wkp;
	}
    for (size_t k = p+1; k < n; k++) {
      const double theta = k*theta0;
      const complex_t wkp = {cos(theta), Forward?-sin(theta):sin(theta)};
	  complex_t& a = *(x_begin + k + p*n);
	  a *= wkp;
	  complex_t& b = *(x_begin + p + k*n);
	  b *= wkp;
	  std::swap(a, b);
    }
  }
  for (size_t k = 0; k < n; k++) // FFT all k-line of x
    _fft_multi_column<false, Forward>(n, x_begin + k * n, y_begin + k * n);
  for (size_t k = 0; k < n; k++) { // transpose x
    for (size_t p = k+1; p < n; p++) {
      std::swap(*(x_begin + p + k*n), *(x_begin + k + p*n));
    }
  }
}

template <bool Forward, typename It>
void _eightstep_fft(size_t log_n, It x_begin, It y_begin) {
  const size_t n = 1ull << log_n;
  const size_t m = n/2;
  const double theta0 = M_PI*2/n;
  // step 1
  for (size_t p = 0; p < m; p++) {
	const double theta = p*theta0;
	const complex_t wp = {cos(theta), Forward?-sin(theta):sin(theta)};
	const complex_t a = *(x_begin + p + 0);
	const complex_t b = *(x_begin + p + m);
	*(y_begin + p + 0) = a + b;
	*(y_begin + p + m) = (a - b) * wp;
  }
  // step 2 to 7
  _sixstep_fft<Forward>(log_n - 1, y_begin + 0, x_begin + 0);
  _sixstep_fft<Forward>(log_n - 1, y_begin + m, x_begin + m);
  // step 8
#ifdef __AVX__
  for (size_t p = 0; p < m; p+=2) {
	const __m256d ab = _mm256_load_pd((y_begin + p + 0)->data());
	const __m256d xy = _mm256_load_pd((y_begin + p + m)->data());
	_mm256_store_pd((x_begin + 2*(2*p+0))->data(), _mm256_permute2f128_pd(ab, xy, 0b00100000)); // ax
	_mm256_store_pd((x_begin + 2*(2*p+1))->data(), _mm256_permute2f128_pd(ab, xy, 0b00110001)); // by
  }
#else
  for (size_t p = 0; p < m; p++) {
	*(x_begin + 2*p + 0) = *(y_begin + p + 0);
	*(x_begin + 2*p + 1) = *(y_begin + p + m);
  }
#endif
}


template <bool Forward, typename It>
void _fft_stockham(It vec_begin, It vec_end, It aux_begin, It aux_end) {
  auto len = vec_end - vec_begin;
  assert(bo::popcnt_u64(len) == 1);
  assert(aux_end - aux_begin == len);

  auto log_n = calc::log_n(len);
  if (len <= 1) {}
  else if (len == 2)
    _fft_multi_column<false, Forward>(len, vec_begin, aux_begin);
  else if ((log_n & 1) == 0)
	_sixstep_fft<Forward>(log_n, vec_begin, aux_begin);
  else
	_eightstep_fft<Forward>(log_n, vec_begin, aux_begin);
}

template <bool Forward, typename It>
void _fft(It vec_begin, It vec_end, It aux_begin, It aux_end) {
  _fft_stockham<Forward>(vec_begin, vec_end, aux_begin, aux_end);
}

template <bool Forward, typename It>
void _fft(It vec_begin, It vec_end) {
  auto len = vec_end - vec_begin;
  assert(bo::popcnt_u64(len) == 1);
#ifdef CUSTOM_FFT
  complex_vector aux(len, {0, 0});
  _fft<Forward>(vec_begin, vec_end, aux.begin(), aux.end());
#else
  fftw_plan plan = fftw_plan_dft_1d(len, reinterpret_cast<fftw_complex*>(&*vec_begin), reinterpret_cast<fftw_complex*>(&*vec_begin), Forward?FFTW_FORWARD:FFTW_BACKWARD, FFTW_ESTIMATE);
  fftw_execute(plan);
  fftw_destroy_plan(plan);
#endif
}

template <typename It>
void fft(It vec_begin, It vec_end, It aux_begin, It aux_end) {
  _fft<true>(vec_begin, vec_end, aux_begin, aux_end);
}

template <typename It>
void fft(It vec_begin, It vec_end) {
  _fft<true>(vec_begin, vec_end);
}

void fft(complex_vector& vec) {
  auto len = calc::greater_eq_pow2(vec.size());
  vec.resize(len, {0,0});
  fft(vec.begin(), vec.end());
}

template <typename It>
void divide_all(It vec_begin, It vec_end, double n) {
#ifdef __AVX__
  const auto nn = _mm256_set1_pd(n);
  for (auto it = vec_begin; it != vec_end; it+=2) {
    const auto ab = _mm256_load_pd(reinterpret_cast<double*>(&*it));
    _mm256_store_pd(reinterpret_cast<double*>(&*it), _mm256_div_pd(ab, nn));
  }
#else
  for (auto it = vec_begin; it != vec_end; ++it)
	*it /= n;
#endif
}

template <typename Iit, typename Oit>
void multiply_for_each(Iit g_begin, Iit g_end, Iit h_begin, Iit h_end, Oit f_begin) {
  auto len = g_end - g_begin;
  assert(len == h_end-h_begin);
#ifdef __AVX__
  for (size_t i = 0; i < len; i+=2) {
	const auto ab = _mm256_load_pd(reinterpret_cast<const double*>(&*(g_begin+i)));
	const auto cd = _mm256_load_pd(reinterpret_cast<const double*>(&*(h_begin+i)));
	_mm256_store_pd(reinterpret_cast<double*>(&*(f_begin+i)), mulpz2(ab, cd));
  }
#else
  for (size_t i = 0 ; i < len; i++)
    *(f_begin+i) = *(g_begin+i) * *(h_begin+i);
#endif
}

template <typename It>
void ifft(It vec_begin, It vec_end, It aux_begin, It aux_end) {
  _fft<false>(vec_begin, vec_end, aux_begin, aux_end);
  divide_all(vec_begin, vec_end, vec_end-vec_begin);
}

template <typename It>
void ifft(It vec_begin, It vec_end) {
  _fft<false>(vec_begin, vec_end);
  divide_all(vec_begin, vec_end, vec_end-vec_begin);
}

void ifft(complex_vector& vec) {
  auto len = calc::greater_eq_pow2(vec.size());
  vec.resize(len, {0,0});
  ifft(vec.begin(), vec.end());
}

template <typename It>
void multiply_polynomial(It g_begin, It g_end, It h_begin, It h_end, It f_begin, It f_end) {
  fft(g_begin, g_end);
  fft(h_begin, h_end);
  multiply_for_each(g_begin, g_end, h_begin, h_end, f_begin);
  ifft(f_begin, f_end);
}

void multiply_polynomial(complex_vector& g, complex_vector& h, complex_vector& f) {
  auto len = calc::greater_eq_pow2(g.size()+h.size()-1);
  g.resize(len, {0,0});
  h.resize(len, {0,0});
  f.resize(len, {0,0});
  multiply_polynomial(g.begin(), g.end(), h.begin(), h.end(), f.begin(), f.end());
}


void multiply_polynomial_inplace(complex_vector& g, complex_vector& h) {
  auto len = calc::greater_eq_pow2(g.size()+h.size()-1);
  g.resize(len, {0,0});
  h.resize(len, {0,0});
  multiply_polynomial(g.begin(), g.end(), h.begin(), h.end(), g.begin(), g.end());
}

} // namespace fft


class Fft {
 public:
  using polynomial_type = complex_vector;
 private:
  size_t n_;

 public:
  Fft(size_t n) : n_(n) {}

  size_t n() const { return n_; }

  void transform(const polynomial_type& f, polynomial_type& tf) const {
	tf.assign(n_, 0);
	std::copy(f.begin(), f.end(), tf.begin());
	fft::fft(tf.begin(), tf.end());
  }

  template <typename InputIterator, typename OutputIterator>
  void transform(InputIterator begin, InputIterator end, OutputIterator out_begin, OutputIterator out_end) const {
    std::copy(begin, end, out_begin);
    fft::fft(out_begin, out_end);
  }

  void inplace_transform(polynomial_type& f) const {
	fft::fft(f.begin(), f.end());
  }

  template <typename Iterator>
  void inplace_transform(Iterator begin, Iterator end) const {
    fft::fft(begin, end);
  }

  void inverse_transform(const polynomial_type& f, polynomial_type& tf) const {
    tf.assign(n_, 0);
    std::copy(f.begin(), f.end(), tf.begin());
	fft::ifft(tf.begin(), tf.end());
  }

  template <typename InputIterator, typename OutputIterator>
  void inverse_transform(InputIterator in_begin, InputIterator in_end, OutputIterator out_begin, OutputIterator out_end) const {
    std::copy(in_begin, in_end, out_begin);
    fft::ifft(out_begin, out_end);
  }

  void inplace_inverse_transform(polynomial_type& f) const {
	fft::ifft(f.begin(), f.end());
  }

  template <typename Iterator>
  void inplace_inverse_transform(Iterator begin, Iterator end) const {
    fft::ifft(begin, end);
  }

};

}

#endif //BIT_FITTING_INCLUDE_FFT_HPP_
