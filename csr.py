class CSR:
    def __init__(self, file=None, n_vertices=None, n_edges=None, edges=None):
        if file:
            with open(file, "r") as f:
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
        else:
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
