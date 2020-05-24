import random
from csr import CSR


def erdos_renyi_graph(n=5, p=0.2):
    n_edges = 0
    edges = {i: [] for i in range(n)}

    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < p:
                edges[i].append(j)
                edges[j].append(i)
                n_edges += 1

    return CSR(n, n_edges, edges)
