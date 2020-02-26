#ifndef BIT_FITTING_INCLUDE_BIT_FIELD_HPP_
#define BIT_FITTING_INCLUDE_BIT_FIELD_HPP_

namespace bit_fitting {

class default_bit_field : public sim_ds::BitVector {
  using _base = sim_ds::BitVector;
 public:
  default_bit_field() = default;
  default_bit_field(size_t size) : _base(size) {}
  default_bit_field(size_t size, bool initial_bit) : _base(size, initial_bit) {}

  template <class Container>
  void multiple_set(const Container& indexes, bool bit) {
	for (auto i : indexes) {
	  operator[](i) = bit;
	}
  }

};

}

#endif //BIT_FITTING_INCLUDE_BIT_FIELD_HPP_
