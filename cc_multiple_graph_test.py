import os
import csr
import time
import numpy as np
import matplotlib.pyplot as plt
from clustering_coefficient import uniform_wedge, uniform_edge, uniform_vertex

cwd = os.getcwd()
graph = "out.com-amazon"
G = csr.CSR(tsv_file=f"{cwd}\\tsv_graphs\\{graph}")

write_to_txt = True
sample_sizes = [300, 500, 1000]
iterations_per_sample_size = 10
triangle_count = 667129

x = np.arange(iterations_per_sample_size)
uw_results = np.zeros((len(sample_sizes), iterations_per_sample_size))
ue_results = np.zeros((len(sample_sizes), iterations_per_sample_size))
uv_results = np.zeros((len(sample_sizes), iterations_per_sample_size))

uw_time_results = np.zeros((len(sample_sizes), iterations_per_sample_size))
ue_time_results = np.zeros((len(sample_sizes), iterations_per_sample_size))
uv_time_results = np.zeros((len(sample_sizes), iterations_per_sample_size))

fig, ax = plt.subplots(nrows=len(sample_sizes), ncols=2)

for s, sample_size in enumerate(sample_sizes):
    print(f"Sample size:\t{sample_size}")
    for i in range(iterations_per_sample_size):
        start = time.process_time()
        uw = uniform_wedge(G, sample_size)
        uw_time = time.process_time() - start
        print(f"Uniform wedge time:\t{uw_time}")
        uw_results[s, i] = uw
        uw_time_results[s, i] = uw_time

        start = time.process_time()
        ue = uniform_edge(G, sample_size)
        ue_time = time.process_time() - start
        print(f"Uniform edge time:\t{ue_time}")
        ue_results[s, i] = ue
        ue_time_results[s, i] = ue_time

        start = time.process_time()
        uv = uniform_vertex(G, sample_size)
        uv_time = time.process_time() - start
        print(f"Uniform vertex time:\t{uv_time}")
        uv_results[s, i] = uv
        uv_time_results[s, i] = uv_time

    ax[s, 0].plot(x, uw_results[s], '-o', color='tab:blue')
    ax[s, 0].plot(x, ue_results[s], '-o', color='tab:green')
    ax[s, 0].plot(x, uv_results[s], '-o', color='tab:red')
    ax[s, 0].axhline(y=np.mean(uw_results[s]), color='tab:blue', linestyle='--')
    ax[s, 0].axhline(y=np.mean(ue_results[s]), color='tab:green', linestyle='--')
    ax[s, 0].axhline(y=np.mean(uv_results[s]), color='tab:red', linestyle='--')
    ax[s, 0].axhline(y=triangle_count, color='k', linestyle='--')

    ax[s, 1].plot(x, uw_time_results[s], '-o', color='tab:blue')
    ax[s, 1].plot(x, ue_time_results[s], '-o', color='tab:green')
    ax[s, 1].plot(x, uv_time_results[s], '-o', color='tab:red')
    ax[s, 1].axhline(y=np.mean(uw_time_results[s]), color='tab:blue', linestyle='--')
    ax[s, 1].axhline(y=np.mean(ue_time_results[s]), color='tab:green', linestyle='--')
    ax[s, 1].axhline(y=np.mean(uv_time_results[s]), color='tab:red', linestyle='--')

handles, labels = ax[0, 0].get_legend_handles_labels()
labels = ["Uniform Wedge", "Uniform Edge", "Uniform Vertex",
          "Uniform Wedge Mean", "Uniform Edge Mean", "Uniform Vertex Mean", "Real value"]
fig.legend(handles, labels, loc='upper center')

if write_to_txt:
    with open(f"{graph}_results.txt", "w", encoding="utf-8") as txt:
        txt.write(f"Real triangle count:\t{triangle_count}\n")
        txt.write(f"Iterations:\t{iterations_per_sample_size}\n")
        for s, sample_size in enumerate(sample_sizes):
            txt.write(f"Sample size:\t{sample_size}\n")
            txt.write(f"\tUniform Wedge Mean:\t{np.mean(uw_results[s])}\n")
            txt.write(f"\tUniform Wedge Time Mean:\t{np.mean(uw_time_results[s])}s\n")
            txt.write(f"\tUniform Edge Mean:\t{np.mean(ue_results[s])}\n")
            txt.write(f"\tUniform Edge Time Mean:\t{np.mean(ue_time_results[s])}s\n")
            txt.write(f"\tUniform Vertex Mean:\t{np.mean(uv_results[s])}\n")
            txt.write(f"\tUniform Vertex Time Mean:\t{np.mean(uv_time_results[s])}s\n")

plt.show()
