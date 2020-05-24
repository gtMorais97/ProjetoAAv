import networkx as nx


def csr_to_networkx(csr):
    graph = nx.Graph()
    graph.add_nodes_from(csr.vertices)
    graph.add_edges_from([(i, j)
                          for i in csr.vertices for j in csr.get_neighbors(i)])
    return graph


def betweenness_centrality(csr):
    return nx.betweenness_centrality(csr_to_networkx(csr))
