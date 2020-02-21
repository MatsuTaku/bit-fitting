#ifndef BIT_FITTING_INCLUDE_EMPTY_LINK_HPP_
#define BIT_FITTING_INCLUDE_EMPTY_LINK_HPP_

#include <vector>
#include <iterator>

namespace bit_fitting {

struct empty_linked_list_unit {
  long long succ = 0;
  long long pred = 0;
};

template <typename Container,
    typename Iter = typename Container::iterator,
    typename BidirectionalUnit = empty_linked_list_unit,
    typename Traits = std::iterator_traits<Iter>>
class EmptyLinkedListInline {
 public:
  using container_iterator = Iter;
  using container_iterator_traits = Traits;
  using unit = BidirectionalUnit;

  static constexpr size_t kDisabledFront = -1ull;

  class iterator : std::iterator<std::bidirectional_iterator_tag, Iter> {
   public:
    using pointer = Iter;
   private:
    pointer unit_iter_;
   public:
    iterator(pointer iter) : unit_iter_(iter) {}
    iterator& operator++() {
      unit_iter_ = begin_ + unit_iter_->succ;
      return *this;
    }
    iterator& operator--() {
      unit_iter_ = begin_ + unit_iter_->pred;
      return *this;
    }
    bool operator!=(iterator x) const {
      return unit_iter_ != x.unit_iter_;
    }
    size_t operator*() const {
      return unit_iter_ - begin_;
    }
  };

 private:
  Iter begin_;
  Iter end_;
  size_t front_;

 public:
  EmptyLinkedListInline(Iter begin, Iter end) : begin_(begin), end_(end), front_(kDisabledFront) {}

  void update_iterator(Iter new_begin, Iter new_end) {
    begin_ = new_begin;
    end_ = new_end;
  }

  iterator begin() const {
    return iterator(begin_ + front_);
  }
  iterator end() const {
    return begin();
  }
  iterator begin() {
    return iterator(begin_ + front_);
  }
  iterator end() {
    return begin();
  }

  void push(size_t index) {
    auto unit = begin_ + index;
    if (front_ == kDisabledFront) {
      front_ = index;
      unit->succ = index;
      unit->pred = index;
    } else {
      auto front = begin_ + front_;
      auto tail = begin_ + front->pred;
      unit->succ = front_;
      unit->pred = front->pred;
      tail->succ = index;
      front->pred = index;
    }
  }

  void pop(size_t index) {
    auto unit = begin_ + index;
    auto succ_i = unit->succ;
    if (index == front_) {
      front_ = front_ == succ_i ? kDisabledFront : succ_i;
    }
    auto succ = begin_ + succ_i;
    auto pred = begin_ + unit->pred;
    succ->pred = unit->pred;
    pred->succ = succ_i;
    unit->succ = 0;
    unit->pred = 0;
  }

};

class EmptyLinkedList {
 public:
  using container_type = std::vector<empty_linked_list_unit>;
  using linked_list_type = EmptyLinkedListInline<container_type>;
  using iterator = linked_list_type::iterator;
 private:
  container_type container_;
  linked_list_type linked_list_;

 public:
  EmptyLinkedList() : linked_list_(container_.begin(), container_.end()) {}
  EmptyLinkedList(size_t size) : container_(size), linked_list_(container_.begin(), container_.end()) {}

  void resize(size_t new_size) {
    container_.resize(new_size);
    linked_list_.update_iterator(container_.begin(), container_.end());
  }

  iterator begin() const { return linked_list_.begin(); }
  iterator end() const { return linked_list_.end(); }
  iterator begin() { return linked_list_.begin(); }
  iterator end() { return linked_list_.end(); }

  void push(size_t index) {
    linked_list_.push(index);
  }
  void pop(size_t index) {
    linked_list_.pop(index);
  }

};

}

#endif //BIT_FITTING_INCLUDE_EMPTY_LINK_HPP_
