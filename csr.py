import random
from utils import tsv_to_array


class CSR:
    """
    The class representing the CSR representation of a graph.
    It contains the following attributes:
        - n_vertices:   Number of vertices in the graph
        - n_edges:      Number of edges in the graph
        - v:            Vector containing all neighbors of every vertex
        - offset:       Vector containing the offsets relative to the 'v' vector
        - vertices:     Vector containing the indices of every node, e.g., [0..n_vertices - 1]

    The constructor can take three types of inputs:
        - txt_file:                     Text file containing the number of vertices, edges, and
                                        all the edges between the vertices

        - tsv_file:                     TSV representation of the graph, which is then transformed
                                        into an easier format in order to build and instance of the class

        - n_vertices, n_edges, edges:   Number of vertices, edges, and a dictionary containing all the
                                        neighbor vertices for every vertex
    """
    def __init__(self, txt_file=None, tsv_file=None, n_vertices=None, n_edges=None, edges=None):
        if txt_file:
            self.txt_init(txt_file)
        if tsv_file:
            array = tsv_to_array(tsv_file)
            self.array_init(*array)
        else:
            self.array_init(n_vertices, n_edges, edges)

    def txt_init(self, txt_file):
        with open(txt_file, "r", encoding="utf-8") as f:
            self.n_vertices = int(f.readline())
            self.n_edges = int(f.readline())

            self.v = []
            self.offset = [0]
            self.vertices = [*range(self.n_vertices)]

            for line in f.readlines():
                if line.strip():
                    neighbors = line.split(" ")

                    for neighbor in neighbors:
                        self.v.append(int(neighbor))

                    self.offset.append(self.offset[-1] + len(neighbors))
                else:
                    self.offset.append(self.offset[-1])

    def array_init(self, n_vertices, n_edges, edges):
        self.n_vertices = n_vertices
        self.n_edges = n_edges

        self.v = []
        self.offset = [0]
        self.vertices = [*range(self.n_vertices)]
        for neighbors in edges.values():
            if neighbors:
                for neighbor in neighbors:
                    self.v.append(int(neighbor))
                self.offset.append(self.offset[-1] + len(neighbors))
            else:
                self.offset.append(self.offset[-1])

    def get_degree(self, node):
        """
        Returns the degree of the node passes as the argument.
        """
        return self.offset[node + 1] - self.offset[node]

    def get_neighbors(self, node):
        """
        Returns a list containing all the neighbors of the node passed as the argument.
        """
        return [self.v[i] for i in range(self.offset[node], self.offset[node + 1])]


def erdos_renyi_graph(n=5, p=0.2):
    """
    This is a implementation of the Erdos-Renyi model for random graph generation.
    """
    n_edges = 0
    edges = {i: [] for i in range(n)}

    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < p:
                edges[i].append(j)
                edges[j].append(i)
                n_edges += 1

    return CSR(n_vertices=n, n_edges=n_edges, edges=edges)
