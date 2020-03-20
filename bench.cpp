#include <iostream>
#include <random>
#include <vector>
#include <set>
#include <unordered_set>

#include "bit_fit.hpp"

namespace {

constexpr bool kShowBase = false;

constexpr size_t kNumTests = 10;
constexpr size_t kNumAlgorithms = 5;
const std::array<std::string, kNumAlgorithms> algorithm_names = {
	"Brute-force",
	"Empty-link",
	"Bit-parallel",
	"Convolution-FFT",
	"Convolution-NTT",
};
using bit_fitter_tuple = std::tuple<
	bit_fitting::bit_fit<bit_fitting::brute_force_bit_fit>,
	bit_fitting::bit_fit<bit_fitting::empty_link_bit_fit>,
	bit_fitting::bit_fit<bit_fitting::bit_parallel_bit_fit>,
	bit_fitting::bit_fit<bit_fitting::convolution_fft_bit_fit>,
	bit_fitting::bit_fit<bit_fitting::convolution_ntt_bit_fit>
>;
bit_fitter_tuple bit_fitters;

const size_t log_n = 24;
//const size_t occurence_rate_inv = 4;
const size_t log_alphabets = 20;

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

std::vector<size_t> create_randmom_pieces(size_t alphabet_size, size_t cnt_occures) {
  std::set<size_t> ps;
  for (size_t i = 0; i < cnt_occures; i++)
    ps.insert(i);
  for (size_t i = 0; i < cnt_occures; i++) {
    size_t t = random_ll()%alphabet_size;
    if (ps.count(t) == 0) {
      ps.erase(i);
      ps.insert(t);
    }
  }
  return std::vector(ps.begin(), ps.end());
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
	  if (not should_guard)
		break;
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

std::array<double, kNumAlgorithms> benchmark_all(size_t field_size, size_t alphabet_size, size_t cnt_occures) {
  std::array<double, kNumAlgorithms> time_sum = {};
  volatile long long base = 0;
  { // warm up
	const auto pattern = create_randmom_pieces(alphabet_size, cnt_occures);
	const auto ans = random_ll()%field_size;
	auto field = create_field_fit_at_with(field_size, ans, pattern);
	volatile double t = 0;
	{
	  auto f = std::get<0>(bit_fitters).field(&field);
	  t = time_us_in([&]{ base = std::get<0>(bit_fitters).find(f, pattern); });
	}
	{
	  auto f = std::get<1>(bit_fitters).field(&field);
	  t = time_us_in([&]{ base = std::get<1>(bit_fitters).find(f, pattern); });
	}
	{
	  auto f = std::get<2>(bit_fitters).field(&field);
	  t = time_us_in([&]{ base = std::get<2>(bit_fitters).find(f, pattern); });
	}
	{
	  auto f = std::get<3>(bit_fitters).field(&field);
	  t = time_us_in([&]{ base = std::get<3>(bit_fitters).find(f, pattern); });
	}
	{
	  auto f = std::get<4>(bit_fitters).field(&field);
	  t = time_us_in([&]{ base = std::get<4>(bit_fitters).find(f, pattern); });
	}
  }
  for (int i = 0; i < kNumTests; i++) {
	auto pattern = create_randmom_pieces(alphabet_size, cnt_occures);
	auto ans = random_ll()%field_size;
	auto field = create_field_fit_at_with(field_size, ans, pattern);
	{
	  auto f = std::get<0>(bit_fitters).field(&field);
	  time_sum[0] += time_us_in([&]{
	    base = std::get<0>(bit_fitters).find(f, pattern);
	    if (kShowBase)
	    	std::cerr<<base<<std::endl;
	  });
	}
	{
	  auto f = std::get<1>(bit_fitters).field(&field);
	  time_sum[1] += time_us_in([&]{
	    base = std::get<1>(bit_fitters).find(f, pattern);
		if (kShowBase)
		  std::cerr<<base<<std::endl;
	  });
	}
	{
	  auto f = std::get<2>(bit_fitters).field(&field);
	  time_sum[2] += time_us_in([&] {
	    base = std::get<2>(bit_fitters).find(f, pattern);
		if (kShowBase)
		  std::cerr<<base<<std::endl;
	  });
	}
	{
	  auto f = std::get<3>(bit_fitters).field(&field);
	  time_sum[3] += time_us_in([&]{
	    base = std::get<3>(bit_fitters).find(f, pattern);
		if (kShowBase)
		  std::cerr<<base<<std::endl;
	  });
	}
	{
	  auto f = std::get<4>(bit_fitters).field(&field);
	  time_sum[4] += time_us_in([&]{
	  	base = std::get<4>(bit_fitters).find(f, pattern);
		if (kShowBase)
		  std::cerr<<base<<std::endl;
	  });
	}
  }

  for (auto& sum : time_sum) sum /= kNumTests;
  return time_sum;
}

}

int main() {
  std::cout << "Test various bit-fit algorithm" << std::endl;
  std::cout << "N: " << (1<<log_n) << std::endl;
//  std::cout << "Occurence rate: " << "1/" << occurence_rate_inv << std::endl;
  std::cout << "\\Sigma: " << (1<<log_alphabets) << std::endl;
  std::cout << "---------------------------------------------------------------------" << std::endl;
  std::cout << "       \t";
  for (auto name : algorithm_names)
    std::cout << name << '\t' ;
  std::cout << std::endl;
  for (size_t log_m = 2; log_m < log_alphabets; log_m+=1) {
	std::cout<< (1<<log_m) << '\t' << std::flush;
	auto times = benchmark_all(1 << log_n, 1 << log_alphabets, 1<<log_m);
	for (auto time : times) {
	  std::cout<< time/1000000 << '\t' ;
	}
	std::cout << std::endl;
  }

//  test({1,1,1,0,1,0,1,1,1,0,0,1,1,1,0,1,1,1,1,1,1,0,1,1,0,1},
//       {1,3,6,8,10}, 5);

  return 0;
}
