import random
import csr
import networkx as nx


def erdos_renyi_graph(n=5, p=0.2):
    n_edges = 0
    edges = {i: [] for i in range(n)}

    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < p:
                edges[i].append(j)
                edges[j].append(i)
                n_edges += 1

    return csr.CSR(n_vertices=n, n_edges=n_edges, edges=edges)


def tsv_to_array(tsv_file):
    print(tsv_file)
    with open(tsv_file, "r", encoding='utf-8') as tsv:
        tsv.readline()
        info = tsv.readline().split(" ")
        n_vertices = int(info[2])
        n_edges = int(info[1])

        edges = {i: [] for i in range(n_vertices)}

        for line in tsv.readlines():
            info = line.split(" ")
            vertex_from = int(info[0]) - 1
            vertex_to = int(info[1]) - 1

            edges[vertex_from].append(vertex_to)
            edges[vertex_to].append(vertex_from)

        return (n_vertices, n_edges, edges)


def csr_to_networkx(csr):
    graph = nx.Graph()
    graph.add_nodes_from(csr.vertices)
    graph.add_edges_from([(i, j)
                          for i in csr.vertices for j in csr.get_neighbors(i)])
    return graph


def betweenness_centrality(csr, normalized=True):
    return nx.betweenness_centrality(csr_to_networkx(csr), normalized=normalized)
