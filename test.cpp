#include <vector>
#include <iostream>
#include <bitset>

#include "bit_fit.hpp"

namespace {

void show_pattern(const std::vector<size_t> pattern) {
  for (int i = 0, j = 0; i <= pattern.back(); i++) {
    if (i < pattern[j]) {
      std::cout << 0;
    } else {
      std::cout << 1;
      j++;
    }
  }
  std::cout << std::endl;
}

void test(std::vector<bool> field, std::vector<size_t> pattern) {
  std::cout << "Test" << std::endl;

  auto run_algorithm = [&](std::string alg_name, auto bit_fit) {
    std::cout << alg_name << std::endl;
    auto pos = bit_fit.find(field, pattern);

    for (auto bit : field) {
      std::cout << bit;
    }
    std::cout << std::endl;
    for (int i = 0; i < pos; i++) std::cout << " ";
    show_pattern(pattern);
  };
  run_algorithm("Brute force", bit_fitting::bit_fit<bit_fitting::brute_force_bit_fit>());
  run_algorithm("FFT", bit_fitting::bit_fit<bit_fitting::fft_bit_fit>());
}

}

int main() {
  test({1,1,1,0,1,0,1,1,1,0,0,1,1,1,0,1,1,1,1,1,1,0,1,1,0,1},
       {1,3,6,8,10});

}
