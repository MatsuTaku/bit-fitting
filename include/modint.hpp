#ifndef BIT_FITTING_INCLUDE_MODINT_HPP_
#define BIT_FITTING_INCLUDE_MODINT_HPP_

namespace bit_fitting {

template <typename T, T Mod>
class modint {
 private:
  T val_;

 public:
  constexpr modint() : val_(0) {}
  constexpr modint(T x) : val_(x%Mod) {}

  constexpr T val() const { return val_; }

  constexpr bool operator==(modint x) const { return val() == x.val(); }
  constexpr bool operator!=(modint x) const { return !(*this == x); }

  constexpr modint operator+(modint x) const { return val() + x.val(); }
  constexpr modint& operator+=(modint x) { return *this = *this + x; }
  constexpr modint operator-(modint x) const { return val()+Mod - x.val(); }
  constexpr modint& operator-=(modint x) { return *this = *this - x; }
  constexpr modint operator*(modint x) const { return val() * x.val(); }
  constexpr modint& operator*=(modint x) { return *this = *this * x; }
  friend constexpr modint pow(modint x, modint p) {
    modint t = 1;
    modint u = x;
    auto pv = p.val();
    while (pv) {
      if (pv&1)
        t *= u;
      u *= u;
      pv >>=1;
    }
    return t;
  }
  constexpr modint operator/(modint x) const { return *this * pow(x, Mod-2); }
  constexpr modint& operator/=(modint x) { return *this = *this / x; }

  constexpr bool operator<(modint x) const { return val() < x.val(); }
  constexpr bool operator<=(modint x) const { return val() <= x.val(); }
  constexpr bool operator>(modint x) const { return val() > x.val(); }
  constexpr bool operator>=(modint x) const { return val() >= x.val(); }

  std::istream& operator>>(std::istream& is) const { is >> val_; *this = val_; return is; }
  friend std::ostream& operator<<(std::ostream& os, const modint& x) { return os << x.val(); }

};

}

#endif //BIT_FITTING_INCLUDE_MODINT_HPP_
