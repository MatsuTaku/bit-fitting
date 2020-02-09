#include <vector>
#include <iostream>
#include <bitset>
#include <unordered_set>

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

void test(std::vector<bool> field, std::vector<size_t> pattern, size_t ans) {
  std::cout << "ANSWER = " << ans << std::endl;

  auto run_algorithm = [&](std::string alg_name, auto bit_fit) {
    auto start = std::clock();
    auto pos = bit_fit.find(field, pattern);
    auto time = std::clock()-start;
    std::cout << alg_name << ":\t" << pos << "\t in: " << (double)time/1000000 << "sec" << std::endl;
  };
  run_algorithm("Brute force", bit_fitting::bit_fit<bit_fitting::brute_force_bit_fit>());
  run_algorithm("FFT", bit_fitting::bit_fit<bit_fitting::fft_bit_fit>());
}

void benchmark(size_t field_size, size_t alphabet_size, size_t inv_exist_rate) {
  size_t F = field_size;
  size_t P = alphabet_size;
  std::vector<size_t> p;
  for (size_t i = 0; i < P; i++) {
    if (rand()%inv_exist_rate == 0)
      p.push_back(i);
  }
  size_t ans = rand()%(F-P);
  std::vector<bool> f(F, 1);
  std::unordered_set<size_t> p_set;
  for (auto pos : p)
    p_set.insert(ans + pos);
  for (size_t i = 0; i < F-P+1; i++) {
    if (i == ans)
      continue;
    size_t target = i + p[rand()%p.size()];
    while (p_set.count(target) == 1)
      target = i + p[rand()%p.size()];
    f[target] = 0;
  }
  std::cout << "N:\t" << field_size << "M:\t" << alphabet_size << "rate: 1/" << inv_exist_rate << std::endl;
  test(f, p, ans);
  std::cout << "---------" << std::endl;
}

}

int main() {
  std::cout << "Test various bit-fit algorithm" << std::endl;
  for (size_t log_m = 8; log_m < 24; log_m++) {
    benchmark(1<<24, 1<<log_m, 4);
  }

//  test({1,1,1,0,1,0,1,1,1,0,0,1,1,1,0,1,1,1,1,1,1,0,1,1,0,1},
//       {1,3,6,8,10}, 5);

  return 0;
}
