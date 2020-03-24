#ifndef BIT_FITTING_INCLUDE_CONVOLUTION_HPP_
#define BIT_FITTING_INCLUDE_CONVOLUTION_HPP_

#include "bo.hpp"

#include "fft.hpp"
#include "ntt.hpp"

namespace bit_fitting {

template <typename Transformer>
class convolution {
 public:
  using transformer = Transformer;
  using polynomial_type = typename transformer::polynomial_type;
 private:
  Transformer transformer_;

 public:
  convolution(size_t size) : transformer_(1ull<<(64-bo::clz_u64(size-1))) {}

  size_t n() const { return transformer_.n(); }

  void operator()(const polynomial_type& g, const polynomial_type& h, polynomial_type& f) const {
	polynomial_type th;
    transformer_.transform(g, f);
	transformer_.transform(h, th);
	for (size_t i = 0; i < f.size(); i++)
      f[i] *= th[i];
	transformer_.inplace_inverse_transform(f);
    for (auto& v : f)
      v /= n();
  }

};

}

#endif //BIT_FITTING_INCLUDE_CONVOLUTION_HPP_
