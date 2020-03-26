#include <iostream>
#include <random>
#include <vector>
#include <set>
#include <unordered_set>

#include "bit_fit.hpp"
#include "convolution.hpp"

namespace {

constexpr bool kShowBase = true;

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

bit_fitting::default_bit_field create_sparse_field_fit_at_with(size_t field_size, size_t x, const std::vector<size_t>& pieces) {
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

bit_fitting::default_bit_field create_dense_field_fit_at_with(size_t field_size, size_t x, const std::vector<size_t>& pieces) {
  auto F = field_size;
  auto P = pieces.back();
  bit_fitting::default_bit_field field(F, true);
  std::unordered_set<size_t> p_set;
  for (auto pos : pieces)
	p_set.insert(x + pos);
  for (size_t i = 0; i < F-P; i++) {
	if (i == x)
	  continue;
	auto p = pieces[random_ll()%pieces.size()];
	while (p_set.count(i+p) > 0)
	  p = pieces[random_ll()%pieces.size()];
	field[i+p] = false;
  }
  return field;
}

bit_fitting::default_bit_field create_field_rated(size_t field_size, size_t inv_occurence_rate) {
  auto F = field_size;
  bit_fitting::default_bit_field field(F, true);
  for (int i = 0; i < F/inv_occurence_rate; i++)
    field[i] = false;
  for (int i = 0; i < F; i++) {
    auto t = random_ll()%F;
    auto tmp = field[i];
    field[i] = field[t];
    field[t] = tmp;
  }
  return field;
}

std::array<double, kNumAlgorithms> benchmark_all(size_t field_size, size_t alphabet_size, size_t cnt_occures, int sd_type) {
  std::array<double, kNumAlgorithms> time_sum = {};
  volatile long long base = 0;
  { // warm up
	const auto pattern = create_randmom_pieces(alphabet_size, cnt_occures);
	const auto ans = random_ll()%field_size;
	auto field = create_sparse_field_fit_at_with(field_size, ans, pattern);
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
	auto field = (sd_type==0?
				  create_sparse_field_fit_at_with(field_size, ans, pattern):
				  create_field_rated(field_size, sd_type));
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

void check_fft() {
  using transformer = bit_fitting::Fft;
  transformer::polynomial_type T = {1,1,1,1,0,0,0,0};
  transformer::polynomial_type P = {1,0,0,0,0,1,1,1};
  transformer::polynomial_type C; bit_fitting::convolution<bit_fitting::Fft>(8)(T, P, C);
  for (auto v : C) std::cerr<<(long long)(v.real()+0.125)<<" ";
  std::cerr<<std::endl;
}

}

int main(int argc, char* argv[]) {
//  check_fft();

  if (argc < 2) {
    std::cerr<<"Usage: "<<argv[0]<<" [rate: 0(sparse), 1-(rate)]"<<std::endl;
    exit(EXIT_FAILURE);
  }
  int sd_type = argv[1][0]-'0';

  std::cout << "Test various bit-fit algorithm" << std::endl;
  std::cout << "N: " << (1<<log_n) << std::endl;
  std::cout << "\\Sigma: " << (1<<log_alphabets) << std::endl;
  if (sd_type == 0)
    std::cout << "Field type: Sparse field" << std::endl;
  else
    std::cout << "Field type: occurence rate is 1/" << sd_type << std::endl;
  std::cout << "---------------------------------------------------------------------" << std::endl;
  std::cout << "       \t";
  for (auto& name : algorithm_names)
    std::cout << name << '\t' ;
  std::cout << std::endl;
  for (size_t log_m = 2; log_m < log_alphabets; log_m+=1) {
	std::cout<< (1<<log_m) << '\t' << std::flush;
	auto times = benchmark_all(1 << log_n, 1 << log_alphabets, 1<<log_m, sd_type);
	for (auto time : times) {
	  std::cout<< time/1000000 << '\t' ;
	}
	std::cout << std::endl;
  }

  return 0;
}
