#ifndef BIT_FITTING_INCLUDE_CONVOLUTION_HPP_
#define BIT_FITTING_INCLUDE_CONVOLUTION_HPP_

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

  polynomial_type operator()(const polynomial_type& g, const polynomial_type& h) const {
    auto tg = transformer_.transform(g);
	auto th = transformer_.transform(h);
	for (size_t i = 0; i < tg.size(); i++)
	  tg[i] *= th[i];
	transformer_.inplace_transform(tg);
	return tg;
  }

};

}

#endif //BIT_FITTING_INCLUDE_CONVOLUTION_HPP_
