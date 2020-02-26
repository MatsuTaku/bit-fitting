#ifndef BIT_FITTING_INCLUDE_EMPTY_LINK_HPP_
#define BIT_FITTING_INCLUDE_EMPTY_LINK_HPP_

#include <vector>
#include <iterator>

#include <sim_ds/BitVector.hpp>

#include "bit_field.hpp"

namespace bit_fitting {

template <bool EmptyBit>
class empty_linked_bit_vector {
  struct Unit {
	long long succ = 0;
	long long pred = 0;
  };

 private:
  default_bit_field* bf_;
  std::vector<Unit> elv_;
  size_t front_;

 public:
  static constexpr size_t kDisabledFront = -1ull;

  class const_iterator : public std::iterator<std::bidirectional_iterator_tag, Unit, ptrdiff_t, const Unit*, const Unit&> {
   public:
    using _base = std::iterator<std::bidirectional_iterator_tag, Unit, ptrdiff_t, const Unit*, const Unit&>;
    using pointer = typename _base::pointer;
   private:
    pointer front_ptr_;
    size_t index_;
   public:
	const_iterator(pointer front_ptr, size_t index) : front_ptr_(front_ptr), index_(index) {}
	const_iterator& operator++() {
	  index_ = operator*().succ;
	  return *this;
	}
	const_iterator& operator--() {
	  index_ = operator*().pred;
	  return *this;
	}
	bool operator!=(const_iterator x) const {
	  return front_ptr_ != x.front_ptr_ or index_ != x.index_;
	}
	Unit operator*() const {
	  return *(front_ptr_ + index_);
	}
	size_t index() const {
	  return index_;
	}
  };

 public:
  empty_linked_bit_vector(default_bit_field* bf) : bf_(bf), front_(kDisabledFront) {
    elv_.resize(bf_->size());
    size_t prev = -1;
    for (size_t i = 0; i < bf_->size(); i++) {
      if ((*bf_)[i] == EmptyBit) {
        if (prev == -1) {
          front_ = i;
        } else {
          elv_[prev].succ = i;
          elv_[i].pred = prev;
        }
		prev = i;
      }
    }
    if (prev != -1) {
	  elv_[prev].succ = front_;
	  elv_[front_].pred = prev;
    }
  }

  auto begin() const { return bf_->begin(); }
  auto end() const { return bf_->end(); }

  size_t size() const { return bf_->size(); }

  bool operator[](size_t index) const { return (*bf_)[index]; }
  auto operator[](size_t index) { return (*bf_)[index]; }

  const_iterator el_begin() const { return const_iterator(elv_.data(), front_); }
  const_iterator el_end() const { return el_begin(); }

  template <class Container>
  void multiple_set(const Container& indexes, bool bit) {
    for (auto i : indexes) {
      operator[](i) = bit;
      if (bit == EmptyBit) {
        push(i);
      } else {
        pop(i);
      }
    }
  }

  void push(size_t index) {
    auto& unit = elv_[index];
    if (front_ == kDisabledFront) {
      front_ = index;
      unit.succ = index;
      unit.pred = index;
    } else {
      auto& front = elv_[front_];
      auto& tail = elv_[front.pred];
      unit.succ = front_;
      unit.pred = front.pred;
      tail.succ = index;
      front.pred = index;
    }
  }

  void pop(size_t index) {
    auto& unit = elv_[index];
    auto succ_i = unit.succ;
    if (index == front_) {
      front_ = front_ == succ_i ? kDisabledFront : succ_i;
    }
    auto& succ = elv_[succ_i];
    auto& pred = elv_[unit.pred];
    succ.pred = unit.pred;
    pred.succ = succ_i;
    unit.succ = 0;
    unit.pred = 0;
  }

};

}

#endif //BIT_FITTING_INCLUDE_EMPTY_LINK_HPP_
