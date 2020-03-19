import sys
import numpy
import matplotlib.pyplot as plt

process_id = 0
alg_names = []
alg_datas = []
for line in sys.stdin:
    if process_id == 0:
        if line[0] == '-':
            process_id = 1
    elif process_id == 1:
        alg_names = line.strip().split('\t')
        alg_datas = [[] for _ in range(len(alg_names)+1)]
        process_id = 2
    elif process_id == 2:
        datas = line.strip().split('\t')
        for (i, data) in enumerate(datas):
            alg_datas[i].append(int(data))


fig, ax = plt.subplots()
for i in range(len(alg_names)):
    ax.plot(alg_datas[0], alg_datas[i+1], label=alg_names[i])
ax.set_xlabel('alphabet size')
ax.set_ylabel('time [us/query]')
ax.set_xscale('log')
ax.set_yscale('log')
ax.legend()

fig.show()
fig.savefig('plot_bench.png')
