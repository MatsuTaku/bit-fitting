#include <iostream>
#include <random>
#include <vector>
#include <unordered_set>

#include "sim_ds/BitVector.hpp"

#include "bit_fit.hpp"

namespace {

constexpr size_t kNumTests = 10;
constexpr size_t kNumAlgorithms = 4;
using bit_fitter_tuple = std::tuple<
    bit_fitting::bit_fit<bit_fitting::brute_force_bit_fit>,
	bit_fitting::bit_fit<bit_fitting::empty_link_bit_fit>,
	bit_fitting::bit_fit<bit_fitting::bit_parallel_bit_fit>,
	bit_fitting::bit_fit<bit_fitting::fft_bit_fit>
    >;
bit_fitter_tuple bit_fitters;

std::random_device rd;
std::mt19937_64 eng(rd());
std::uniform_int_distribution<unsigned long long> distr;

uint64_t random_ll() {
  return distr(eng);
}

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

template<class Process>
double time_us_in(Process process) {
  auto start = clock();
  process();
  return clock() - start;
}

std::vector<size_t> create_randmom_pieces(size_t alphabet_size, size_t inv_exist_rate) {
  std::vector<size_t> p;
  while (p.empty()) {
	for (size_t i = 0; i < alphabet_size; i++) {
	  if (random_ll()%inv_exist_rate == 0)
		p.push_back(i);
	}
  }
  return p;
}

bit_fitting::default_bit_field create_field_fit_at_with(size_t field_size, size_t x, const std::vector<size_t>& pieces) {
  auto F = field_size;
  auto P = pieces.back();
  bit_fitting::default_bit_field field(F, true);
  std::unordered_set<size_t> p_set;
  for (auto pos : pieces)
	p_set.insert(x + pos);
  for (size_t i = 0; i < F-P; i++) {
	if (i == x)
	  continue;
    bool should_guard = true;
    for (auto p : pieces) {
      should_guard &= field[i + p];
    }
    if (not should_guard)
      continue;
	size_t target = i + pieces[random_ll()%pieces.size()];
	while (p_set.count(target) == 1)
	  target = i + pieces[random_ll()%pieces.size()];
	field[target] = false;
  }
  return field;
}

template <class Action, size_t Num = 0>
void for_each_fitter(Action action) {
  action(Num, std::get<Num>(bit_fitters));
  if constexpr (Num+1 < kNumAlgorithms) {
    for_each_fitter<Action, Num+1>(action);
  }
}

void benchmark_all(size_t field_size, size_t alphabet_size, size_t inv_exist_rate) {
  std::array<double, kNumAlgorithms> time_sum = {};
  { // warm up
	auto pattern = create_randmom_pieces(alphabet_size, inv_exist_rate);
	auto ans = random_ll()%field_size;
	auto field = create_field_fit_at_with(field_size, ans, pattern);
	volatile double t = 0;
	{
	  auto f = std::get<0>(bit_fitters).field(&field);
	  t = time_us_in([&]{ std::cout << std::get<0>(bit_fitters).find(f, pattern) << std::endl; });
	}
	{
	  auto f = std::get<1>(bit_fitters).field(&field);
	  t = time_us_in([&]{ std::cout << std::get<1>(bit_fitters).find(f, pattern) << std::endl; });
	}
	{
	  auto f = std::get<2>(bit_fitters).field(&field);
	  t = time_us_in([&]{ std::cout << std::get<2>(bit_fitters).find(f, pattern) << std::endl; });
	}
	{
	  auto f = std::get<3>(bit_fitters).field(&field);
	  t = time_us_in([&]{ std::cout << std::get<3>(bit_fitters).find(f, pattern) << std::endl; });
	}
  }
  for (int i = 0; i < kNumTests; i++) {
	auto pattern = create_randmom_pieces(alphabet_size, inv_exist_rate);
	auto ans = random_ll()%field_size;
	auto field = create_field_fit_at_with(field_size, ans, pattern);
	{
	  auto f = std::get<0>(bit_fitters).field(&field);
	  time_sum[0] += time_us_in([&]{ std::cout << std::get<0>(bit_fitters).find(f, pattern) << std::endl; });
	}
	{
	  auto f = std::get<1>(bit_fitters).field(&field);
	  time_sum[1] += time_us_in([&]{ std::cout << std::get<1>(bit_fitters).find(f, pattern) << std::endl; });
	}
	{
	  auto f = std::get<2>(bit_fitters).field(&field);
	  time_sum[2] += time_us_in([&]{ std::cout << std::get<2>(bit_fitters).find(f, pattern) << std::endl; });
	}
	{
	  auto f = std::get<3>(bit_fitters).field(&field);
	  time_sum[3] += time_us_in([&]{ std::cout << std::get<3>(bit_fitters).find(f, pattern) << std::endl; });
	}
  }

  std::cout << "---------" << std::endl
  << "N: " << field_size << "\tM: " << alphabet_size << "\trate: 1/" << inv_exist_rate << std::endl
  << std::fixed
  << "Brute force: " << "\t in " << time_sum[0]/kNumTests/1000000 << " sec" << std::endl
  << "Empty link: " << "\t in " << time_sum[1]/kNumTests/1000000 << " sec" << std::endl
  << "Bit parallel: " << "\t in " << time_sum[2]/kNumTests/1000000 << " sec" << std::endl
  << "FFT: " << "\t in " << time_sum[3]/kNumTests/1000000 << " sec" << std::endl;
}

}

int main() {
  std::cout << "Test various bit-fit algorithm" << std::endl;
  size_t log_n = 24;
  for (size_t log_m = 4; log_m < log_n; log_m+=2) {
	benchmark_all(1 << log_n, 1 << log_m, 4);
  }

//  test({1,1,1,0,1,0,1,1,1,0,0,1,1,1,0,1,1,1,1,1,1,0,1,1,0,1},
//       {1,3,6,8,10}, 5);

  return 0;
}
