#include <iostream>
#include <chrono>
#include <random>
#include <vector>
#include <set>
#include <unordered_set>

#include "bit_fit.hpp"
#include "convolution.hpp"

namespace {

constexpr bool kShowBase = false;

constexpr size_t kNumTests = 10;
constexpr size_t log_n = 24;
constexpr size_t log_alphabets = 20;


using bit_fitter_tuple = std::tuple<
	bit_fitting::brute_force_bit_fit,
	bit_fitting::empty_link_bit_fit,
	bit_fitting::bit_parallel_bit_fit,
	bit_fitting::convolution_fft_bit_fit,
	bit_fitting::convolution_ntt_bit_fit
>;
constexpr size_t kNumAlgorithms = std::tuple_size_v<bit_fitter_tuple>;
const std::array<std::string, kNumAlgorithms> algorithm_names = {
    "Brute-force",
    "Empty-link",
    "Bit-parallel",
    "Convolution-FFT",
    "Convolution-NTT",
};


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
  auto start = std::chrono::high_resolution_clock::now();
  process();
  auto duration = std::chrono::high_resolution_clock::now() - start;
  return std::chrono::duration<double, std::micro>(duration).count();
}

void create_randmom_pieces(size_t alphabet_size, size_t cnt_occures, std::vector<size_t>& dst) {
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
  dst.clear();
  dst.reserve(ps.size());
  for (auto p : ps)
    dst.push_back(p);
}

void create_sparse_field_fit_at_with(size_t field_size, size_t x, const std::vector<size_t>& pieces, bit_fitting::default_bit_field& dst) {
  auto F = field_size;
  auto P = pieces.back();
  dst.assign(F, true);
  std::unordered_set<size_t> p_set;
  for (auto pos : pieces)
	p_set.insert(x + pos);
  for (size_t i = 0; i < F-P; i++) {
	if (i == x)
	  continue;
	bool should_guard = true;
	for (auto p : pieces) {
	  should_guard &= dst[i + p];
	  if (not should_guard)
		break;
	}
	if (not should_guard)
	  continue;
	size_t target = i + pieces[random_ll()%pieces.size()];
	while (p_set.count(target) == 1)
	  target = i + pieces[random_ll()%pieces.size()];
    dst[target] = false;
  }
}

void create_dense_field_fit_at_with(size_t field_size, size_t x, const std::vector<size_t>& pieces, bit_fitting::default_bit_field& dst) {
  auto F = field_size;
  auto P = pieces.back();
  dst.assign(F, true);
  std::unordered_set<size_t> p_set;
  for (auto pos : pieces)
	p_set.insert(x + pos);
  for (size_t i = 0; i < F-P; i++) {
	if (i == x)
	  continue;
	auto p = pieces[random_ll()%pieces.size()];
	while (p_set.count(i+p) > 0)
	  p = pieces[random_ll()%pieces.size()];
    dst[i+p] = false;
  }
}

void create_field_rated(size_t field_size, size_t inv_occurence_rate, bit_fitting::default_bit_field& dst) {
  auto F = field_size;
  dst.assign(F, true);
  for (int i = 0; i < F/inv_occurence_rate; i++)
    dst[i] = false;
  for (int i = 0; i < F; i++) {
    auto t = random_ll()%F;
    auto tmp = dst[i];
    dst[i] = dst[t];
    dst[t] = tmp;
  }
}

template <size_t Id, typename BitFitterTuple>
struct ForEachBitFit : ForEachBitFit<Id-1, BitFitterTuple> {
  using _prev = ForEachBitFit<Id-1, BitFitterTuple>;
  using bit_fit = bit_fitting::bit_fit<std::tuple_element_t<Id, BitFitterTuple>>;
  bit_fit bit_fit_;

  template <class Action>
  void operator()(Action action) const {
    _prev::operator()(action);
    action(Id, bit_fit_);
  }
};

template <typename BitFitterTuple>
struct ForEachBitFit<0, BitFitterTuple> {
  using bit_fit = bit_fitting::bit_fit<std::tuple_element_t<0, BitFitterTuple>>;
  bit_fit bit_fit_;

  template <class Action>
  void operator()(Action action) const {
    action(0, bit_fit_);
  }
};

std::array<double, kNumAlgorithms> benchmark_all(size_t field_size, size_t alphabet_size, size_t cnt_occures, int sd_type) {
  std::array<double, kNumAlgorithms> time_sum = {};
  volatile long long base = 0;
  std::vector<size_t> pattern;
  bit_fitting::default_bit_field field;
  ForEachBitFit<kNumAlgorithms-1, bit_fitter_tuple> for_each_bit_fit;
  { // warm up
	create_randmom_pieces(alphabet_size, cnt_occures, pattern);
	const auto ans = random_ll()%field_size;
    if (sd_type==0)
      create_sparse_field_fit_at_with(field_size, ans, pattern, field);
    else
      create_field_rated(field_size, sd_type, field);
	volatile double t = 0;
    for_each_bit_fit([&](auto id, auto& bit_fit) {
      auto f = bit_fit.field(&field);
      t = time_us_in([&]{ base = bit_fit.find(f, pattern); });
	});
  }
  for (int i = 0; i < kNumTests; i++) {
	create_randmom_pieces(alphabet_size, cnt_occures, pattern);
	auto ans = random_ll()%field_size;
	if (sd_type==0)
	  create_sparse_field_fit_at_with(field_size, ans, pattern, field);
	else
	  create_field_rated(field_size, sd_type, field);
	for_each_bit_fit([&](auto id, auto& bit_fit) {
      auto f = bit_fit.field(&field);
      base = bit_fit.find(f, pattern); // warm up
      time_sum[id] += time_us_in([&]{
        base = bit_fit.find(f, pattern);
        if (kShowBase)
          std::cerr<<base<<std::endl;
      });
	});
  }

  for (auto& sum : time_sum) sum /= kNumTests;
  return time_sum;
}

void check_fft() {
  using transformer = bit_fitting::Fft;
  transformer::polynomial_type T = {1,1,1,1,0,0,0,0};
  transformer::polynomial_type P = {1,0,0,0,0,1,1,1};
  transformer::polynomial_type C; bit_fitting::convolution<transformer>(8)(T, P, C);
  for (auto v : C) std::cerr<<(long long)(v.real()+0.125)<<" ";
  std::cerr<<std::endl;
}

}

int main(int argc, char* argv[]) {
//  check_fft();

  if (argc < 2) {
    std::cerr<<"Usage: "<<argv[0]<<" (rate: 0(sparse), 1~(rate))"<<std::endl;
    exit(EXIT_FAILURE);
  }
  int sd_type = std::stoi(argv[1]);

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

  std::cout << std::fixed;
  for (size_t log_m = 2; log_m <= log_alphabets; log_m+=1) {
	std::cout<< (1<<log_m) << '\t' << std::flush;
	auto times = benchmark_all(1 << log_n, 1 << log_alphabets, 1<<log_m, sd_type);
	for (auto time : times) {
	  std::cout<< time/1000000 << '\t' ;
	}
	std::cout << std::endl;
  }

  return 0;
}
