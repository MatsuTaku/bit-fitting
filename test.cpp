#include <vector>
#include <iostream>
#include <bitset>
#include <unordered_set>

#include "sim_ds/BitVector.hpp"

#include "bit_fit.hpp"

namespace {

void show_pattern(const std::vector<size_t>& pattern) {
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

std::vector<size_t> get_random_pattern(size_t alphabet_size, size_t occurence_rate_inv) {
  std::vector<size_t> p;
  while (p.empty()) {
    for (size_t i = 0; i < alphabet_size; i++) {
      if (arc4random()%occurence_rate_inv == 0)
        p.push_back(i);
    }
  }
  return p;
}

void test_single_find_for_each(const sim_ds::BitVector& field, const std::vector<size_t>& pattern, size_t ans) {
  std::cout << "ANSWER = " << ans << std::endl;

  auto run_algorithm = [&](std::string alg_name, auto bit_fit) {
    auto start = std::clock();
    auto pos = bit_fit.find(field, pattern);
    auto time = std::clock()-start;
    std::cout << alg_name << ":\t" << pos << "\t in: " << (double)time/1000000 << "sec" << std::endl;
  };
  run_algorithm("Brute force", bit_fitting::bit_fit<bit_fitting::brute_force_bit_fit>());
  run_algorithm("Bit parallel", bit_fitting::bit_fit<bit_fitting::bit_parallel_bit_fit>());
  run_algorithm("FFT", bit_fitting::bit_fit<bit_fitting::fft_bit_fit>());
}

void benchmark_single_find(size_t field_size, size_t alphabet_size, size_t inv_exist_rate) {
  size_t F = field_size;
  size_t P = alphabet_size;
  const auto p = get_random_pattern(P, inv_exist_rate);
  size_t ans = arc4random()%F;
  sim_ds::BitVector f(F, 1);
  std::unordered_set<size_t> p_set;
  for (auto pos : p)
    p_set.insert(ans + pos);
  for (size_t i = 0; i < F-P+1; i++) {
    if (i == ans)
      continue;
    size_t target = i + p[arc4random()%p.size()];
    while (p_set.count(target) == 1)
      target = i + p[arc4random()%p.size()];
    f[target] = 0;
  }
  std::cout << "---------" << std::endl;
  std::cout << "N: " << field_size << "\tM: " << alphabet_size << "\trate: 1/" << inv_exist_rate << std::endl;
  test_single_find_for_each(f, p, ans);
}

template<typename BitFitter>
void benchmark_continuous_construction(size_t field_size, size_t alphabet_size, size_t occurence_rate_inv, BitFitter fitter) {
  const size_t F = field_size;
  const size_t P = alphabet_size;
  sim_ds::BitVector field(F, 1);
  const size_t num_patterns = F*occurence_rate_inv/P/2;
  for (size_t n = 0; n < num_patterns; n++) {
    auto pattern = get_random_pattern(P, occurence_rate_inv);
    auto front = fitter.find(field, pattern);
    for (auto pos : pattern) {
      if (front + pos >= F)
        break;
      assert(field[front + pos] == 1);
      field[front + pos] = 0;
    }
  }
}

}

int main() {
  std::cout << "Test various bit-fit algorithm" << std::endl;
  size_t log_n = 16;
  for (size_t log_m = 1; log_m < log_n; log_m++) {
//    benchmark_single_find(1 << log_n, 1 << log_m, 4);
    auto test_construct = [&](const std::string& fitter_name, auto fitter) {
      std::cout << fitter_name << ": ";
      auto start = std::clock();
      benchmark_continuous_construction(1<<log_n, 1<<log_m, 4, fitter);
      auto time = std::clock() - start;
      std::cout << "in " << (double)time/1000000 << "sec" << std::endl;
    };
    std::cout << "---------- N: " << (1<<log_n) << ", M: " << (1<<log_m) << " ----------" << std::endl;
    test_construct("Brute force", bit_fitting::bit_fit<bit_fitting::brute_force_bit_fit>());
    test_construct("Bit parallel", bit_fitting::bit_fit<bit_fitting::bit_parallel_bit_fit>());
    test_construct("FFT", bit_fitting::bit_fit<bit_fitting::fft_bit_fit>());
  }

//  test_single_find_for_each({1,1,1,0,1,0,1,1,1,0,0,1,1,1,0,1,1,1,1,1,1,0,1,1,0,1},
//       {1,3,6,8,10}, 5);

  return 0;
}
