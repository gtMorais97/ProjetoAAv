from utils import tsv_to_array


class CSR:
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
        return self.offset[node + 1] - self.offset[node]

    def get_neighbors(self, node):
        return [self.v[i] for i in range(self.offset[node], self.offset[node + 1])]
