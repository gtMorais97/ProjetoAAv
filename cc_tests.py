import os
import csr
import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from clustering_coefficient import uniform_wedge, uniform_edge, uniform_vertex

matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})

TSV_FILE = "out.com-amazon"
TRIANGLE_COUNT = 667129
SAMPLE_SIZE = 300
ITERATIONS = 15
WRITE_TO_FILE = True

G = csr.CSR(tsv_file=f"{os.getcwd()}\\tsv_graphs\\{TSV_FILE}")

x = np.arange(ITERATIONS)
uw_results = np.zeros(ITERATIONS)
ue_results = np.zeros(ITERATIONS)
uv_results = np.zeros(ITERATIONS)

uw_time_results = np.zeros(ITERATIONS)
ue_time_results = np.zeros(ITERATIONS)
uv_time_results = np.zeros(ITERATIONS)

uw_mape = 0
ue_mape = 0
uv_mape = 0

for i in range(ITERATIONS):
    print(i)
    start = time.process_time()
    uw = uniform_wedge(G, SAMPLE_SIZE)
    uw_time = time.process_time() - start
    uw_results[i] = uw
    uw_time_results[i] = uw_time

    start = time.process_time()
    ue = uniform_edge(G, SAMPLE_SIZE)
    ue_time = time.process_time() - start
    ue_results[i] = ue
    ue_time_results[i] = ue_time

    start = time.process_time()
    uv = uniform_vertex(G, SAMPLE_SIZE)
    uv_time = time.process_time() - start
    uv_results[i] = uv
    uv_time_results[i] = uv_time

    uw_mape += abs(1 - (uw / TRIANGLE_COUNT))
    ue_mape += abs(1 - (ue / TRIANGLE_COUNT))
    uv_mape += abs(1 - (uv / TRIANGLE_COUNT))

plt.plot(x, uw_results, '-o', color="tab:blue")
plt.plot(x, ue_results, '-o', color="tab:green")
plt.plot(x, uv_results, '-o', color="tab:red")
plt.axhline(y=np.mean(uw_results), linestyle='--', color='tab:blue')
plt.axhline(y=np.mean(ue_results), linestyle='--', color='tab:green')
plt.axhline(y=np.mean(uv_results), linestyle='--', color='tab:red')
plt.axhline(y=TRIANGLE_COUNT, linestyle='--', color='k')
plt.legend(["Uniform Wedge", "Uniform Edge", "Uniform Vertex",
            "Uniform Wedge Mean", "Uniform Edge Mean", "Uniform Vertex Mean", "Real"])
plt.title(f"Results over {ITERATIONS} iterations with a sample size of {SAMPLE_SIZE}")
plt.xlabel("Iteration")
plt.ylabel("Triangle count")
plt.savefig(f"{TSV_FILE}_results.pgf")
plt.show()

plt.cla()
plt.clf()

plt.plot(x, uw_time_results, '-o', color="tab:blue")
plt.plot(x, ue_time_results, '-o', color="tab:green")
plt.plot(x, uv_time_results, '-o', color="tab:red")
plt.axhline(y=np.mean(uw_time_results), linestyle='--', color='tab:blue')
plt.axhline(y=np.mean(ue_time_results), linestyle='--', color='tab:green')
plt.axhline(y=np.mean(uv_time_results), linestyle='--', color='tab:red')
plt.legend(["Uniform Wedge", "Uniform Edge", "Uniform Vertex",
            "Uniform Wedge Mean", "Uniform Edge Mean", "Uniform Vertex Mean"])
plt.title(f"Execution time over {ITERATIONS} iterations with a sample size of {SAMPLE_SIZE}")
plt.xlabel("Iteration")
plt.ylabel("Time (seconds)")
plt.savefig(f"{TSV_FILE}_times.pgf")
plt.show()

if WRITE_TO_FILE:
    with open(f"{TSV_FILE}_results.txt", "w", encoding="utf-8") as txt:
        txt.write(f"Triangle count: {TRIANGLE_COUNT}\n")
        txt.write(f"Iterations: {ITERATIONS}\n")
        txt.write(f"Sample size: {SAMPLE_SIZE}\n")
        txt.write(f"Uniform Wedge Mean: {np.mean(uw_results)}\n")
        txt.write(f"Uniform Wedge MAPE: {uw_mape / ITERATIONS}\n")
        txt.write(f"Uniform Wedge Time Mean: {np.mean(uw_time_results)}\n")
        txt.write(f"Uniform Edge Mean: {np.mean(ue_results)}\n")
        txt.write(f"Uniform Edge MAPE: {ue_mape / ITERATIONS}\n")
        txt.write(f"Uniform Edge Time Mean: {np.mean(ue_time_results)}\n")
        txt.write(f"Uniform Vertex Mean: {np.mean(uv_results)}\n")
        txt.write(f"Uniform Vertex MAPE: {uv_mape / ITERATIONS}\n")
        txt.write(f"Uniform Vertex Time Mean: {np.mean(uv_time_results)}\n")
