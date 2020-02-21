#include <vector>
#include <iterator>
#include <complex>
#include <cmath>
#include <algorithm>
#include <iomanip>

#include "bo.hpp"
#include "boost/align/aligned_allocator.hpp"

#ifndef BIT_FITTING_INCLUDE_FFT_HPP_
#define BIT_FITTING_INCLUDE_FFT_HPP_

namespace bit_fitting {

/* FFT(Fast Fourier Transform) - Stockham algorithm */

namespace calc {

size_t upper_pow2(size_t x) {
  return x == 0 ? 0 : 1ull << (64 - bo::clz_u64(x-1));
}

size_t log_n(size_t x) {
  assert(bo::popcnt_u64(x) == 1);
  return bo::ctz_u64(x);
}

}

//using complex_t = std::complex<double>;

struct complex_t {
  double Re, Im;
  complex_t() : Re(0), Im(0) {}
  complex_t(double x) : Re(x), Im(0) {}
  complex_t(double x, double y) : Re(x), Im(y) {}

  double real() const { return Re; }
  double img() const { return Im; }

  complex_t operator+(const complex_t& x) const {
    return {Re + x.Re, Im + x.Im};
  }
  complex_t& operator+=(const complex_t& x) {
    Re += x.Re;
    Im += x.Im;
    return *this;
  }
  complex_t operator-() const {
    return {-Re, -Im};
  }
  complex_t operator-(const complex_t& x) const {
    return {Re - x.Re, Im - x.Im};
  }
  complex_t& operator-=(const complex_t& x) {
    return *this += -x;
  }
  complex_t operator*(const complex_t& x) const {
    return {Re*x.Re - Im*x.Im, Re*x.Im + Im*x.Re};
  }
  complex_t& operator*=(const complex_t& x) {
    return *this = *this * x;
  }
  complex_t operator*(double x) const {
    return {Re*x, Im*x};
  }
  complex_t& operator*=(double x) {
    Re *= x;
    Im *= x;
    return *this;
  }
  friend complex_t operator*(double x, const complex_t& y) {
    return y * x;
  }
  complex_t& operator/=(double x) {
    Re /= x;
    Im /= x;
    return *this;
  }
};


using polynomial_vector = std::vector<complex_t, boost::alignment::aligned_allocator<complex_t, 32>>;


// MARK: FFT Implementations using six-eight-step Stockham Algorithm

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

  } else if (len >= 4) {
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
    _fft<!Odd, Sign>(len/2, 2*stride, y_begin, x_begin);

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


// MARK: FFT Implementation using AVX intrinsics
#ifdef __AVX__

__m256d mulpz2(const __m256d ab, const __m256d xy) {
  const __m256d aa = _mm256_unpacklo_pd(ab, ab);
  const __m256d bb = _mm256_unpackhi_pd(ab, ab);
  const __m256d yx = _mm256_shuffle_pd(xy, xy, 0b0101);
  return _mm256_addsub_pd(_mm256_mul_pd(aa, xy), _mm256_mul_pd(bb, yx));
}

template <bool Odd, long long Sign, typename It>
void _fft_avx(size_t len, size_t stride, It x_begin, It y_begin) {
  const size_t m = len/2;
  const double theta0 = M_PI*2/len;

  if (len == 2) {
    auto z_begin = Odd ? y_begin : x_begin;
    if (stride == 1) {
      const __m128d a = _mm_load_pd(&x_begin->Re + 2*0);
      const __m128d b = _mm_load_pd(&x_begin->Re + 2*1);
      _mm_store_pd(&z_begin->Re + 2*0, _mm_add_pd(a, b));
      _mm_store_pd(&z_begin->Re + 2*1, _mm_sub_pd(a, b));
    } else {
      for (size_t q = 0; q < stride; q += 2) {
        auto xd = &(x_begin+q)->Re;
        auto zd = &(z_begin+q)->Re;
        const __m256d a = _mm256_load_pd(xd + 2*0);
        const __m256d b = _mm256_load_pd(xd + 2*stride);
        _mm256_store_pd(zd + 2*0, _mm256_add_pd(a, b));
        _mm256_store_pd(zd + 2*stride, _mm256_sub_pd(a, b));
      }
    }

  } else if (len >= 4) {
    if (stride == 1) {
      for (size_t p = 0; p < m; p++) {
        const complex_t wp = {cos(p*theta0), sin(p*theta0)*Sign};
        const complex_t a = *(x_begin + p + 0);
        const complex_t b = *(x_begin + p + m);
        *(y_begin + 2*p + 0) = a + b;
        *(y_begin + 2*p + 1) = (a - b) * wp;
      }
    } else {
      for (size_t p = 0; p < m; p++) {
        const double cs = cos(p*theta0);
        const double sn = sin(p*theta0);
        const __m256d wp = _mm256_setr_pd(cs, sn*Sign, cs, sn*Sign);
        for (size_t q = 0; q < stride; q += 2) {
          auto xd = &(x_begin + q)->Re;
          auto yd = &(y_begin + q)->Re;
          const __m256d a = _mm256_load_pd(xd + 2*stride*(p + 0));
          const __m256d b = _mm256_load_pd(xd + 2*stride*(p + m));
          _mm256_store_pd(yd + 2*stride*(2*p + 0), _mm256_add_pd(a, b));
          _mm256_store_pd(yd + 2*stride*(2*p + 1), mulpz2(wp, _mm256_add_pd(a, b)));
        }
      }
    }
    _fft<!Odd, Sign>(len/2, 2*stride, y_begin, x_begin);

  }
}

template <long long Sign, typename It>
void fft_avx(It vec_begin, It vec_end, It aux_begin, It aux_end) {
  auto len = vec_end - vec_begin;
  assert(bo::popcnt_u64(len) == 1);
  assert(aux_end - aux_begin == len);
  _fft_avx<false, Sign>(len, 1, vec_begin, aux_begin);
}

#endif


template <long long Sign=-1, typename It>
void fft(It vec_begin, It vec_end, It aux_begin, It aux_end) {
  auto len = vec_end - vec_begin;
  assert(bo::popcnt_u64(len) == 1);
  assert(aux_end - aux_begin == len);

#ifdef __AVX__
  _fft_avx<false, Sign>(len, 1, vec_begin, aux_begin);
#else
  auto log_n = calc::log_n(len);
  if (len <= 1) {}
  else if (len == 2)
    _fft<false, Sign>(len, 1, vec_begin, aux_begin);
  else if ((log_n & 1) == 0)
    _sixstep_fft<Sign>(log_n, vec_begin, aux_begin);
  else
    _eightstep_fft<Sign>(log_n, vec_begin, aux_begin);
#endif
}

template <typename It>
void fft(It vec_begin, It vec_end) {
  auto len = vec_end - vec_begin;
  assert(bo::popcnt_u64(len) == 1);
  polynomial_vector aux(len, 0);
  fft(vec_begin, vec_end, aux.begin(), aux.end());
}

void fft(polynomial_vector& vec) {
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
  polynomial_vector aux(len, 0);
  inverse_fft(vec_begin, vec_end, aux.begin(), aux.end());
}

void inverse_fft(polynomial_vector& vec) {
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

void multiply_polynomial(polynomial_vector& g, polynomial_vector& h, polynomial_vector& f) {
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

void multiply_polynomial_inplace(polynomial_vector& g, polynomial_vector& h) {
  auto len = calc::upper_pow2(g.size()+h.size()-1);
  g.resize(len, 0);
  h.resize(len, 0);
  multiply_polynomial_inplace(g.begin(), g.end(), h.begin(), h.end());
}


class Polynomial : public polynomial_vector {
  using _base = polynomial_vector;
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


class BinaryObservingDft {
 private:
  sim_ds::BitVector* bv_;
  polynomial_vector dft_;
  size_t dft_size_ = 0;

 public:
  BinaryObservingDft(sim_ds::BitVector* bv) : bv_(bv) {
    dft_size_ = calc::upper_pow2(bv->size());
    dft_.resize(dft_size_, 0);
  }

  void resize(size_t size) {
    if (size > dft_size_) {
      dft_size_ = size;
      bv_->resize(dft_size_, 1);
      for (size_t i = 0; i < size; i++) {
        dft_[i] = (int)!(*bv_)[i];
      }
      fft(dft_);
    }
  }

  void update(size_t i, bool bit) {
    if (bit == (*bv_)[i])
      return;
    double diff = (int)bit - (int)(*bv_)[i];
    (*bv_)[i] = bit;
    double theta0 = M_PI*2/dft_size_;
    for (size_t p = 0; p < dft_size_; p++) {
      complex_t wp = {cos((i*p%dft_size_)*theta0), -sin((i*p%dft_size_)*theta0)};
      dft_[p] += wp * diff;
    }
  }

  const polynomial_vector& dft() const {
    return dft_;
  }
  polynomial_vector& dft() {
    return dft_;
  }

  void fit_pattern(size_t front, const std::vector<size_t>& pattern) {
    if (pattern.size() < calc::log_n(dft_size_)) {
      for (auto p : pattern) {
        update(front + p, 0);
      }
    } else {

    }
  }

};

}

#endif //BIT_FITTING_INCLUDE_FFT_HPP_
