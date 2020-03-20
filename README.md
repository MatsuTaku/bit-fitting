# bit-fitting

## Benchmark usage
```bash
# Build program
mkdir build
cd build
cmake ..
make
cd ..

# Setup to plot benchmark resuit
python -m venv env
source env/bin/activate
pip install -r requirements.txt

# Run benchmark
build/bench | tee >(python plot_bench.py)
```
