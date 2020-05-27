import os
import csr
import numpy as np
import matplotlib.pyplot as plt
from clustering_coefficient import uniform_wedge, uniform_edge, uniform_vertex

cwd = os.getcwd()
graph = csr.CSR(tsv_file=f"{cwd}\\tsv_graphs\\out.com-amazon")

iterations = 10
triangle_count = 667129
sample_size = 300

x = np.array([*range(1, iterations + 1)])
wedge = np.array([])
edge = np.array([])
vertex = np.array([])

for i in range(iterations):
    print(f"Iteration\t{i}")
    wedge = np.append(wedge, uniform_wedge(graph, sample_size))
    edge = np.append(edge, uniform_edge(graph, sample_size))
    vertex = np.append(vertex, uniform_vertex(graph, sample_size))

plt.plot(x, wedge, '-o', color='b')
plt.plot(x, edge, '-o', color='r')
plt.plot(x, vertex, '-o', color='g')
plt.axhline(y=np.mean(wedge), color='b', linestyle='--')
plt.axhline(y=np.mean(edge), color='r', linestyle='--')
plt.axhline(y=np.mean(vertex), color='g', linestyle='--')
plt.axhline(y=triangle_count, color='k', linestyle='--')
plt.legend(["Uniform wedge", "Uniform edge", "Uniform vertex",
            "Uniform wedge mean", "Uniform edge mean", "Uniform vertex mean", "Real"])
plt.show()
