class CSR:

    def __init__(self, file):
        with open(file, "r") as f:
            self.vertices = int(f.readline())
            self.edges = int(f.readline())

            self.v = []
            self.offset = [0]

            for line in f.readlines():
                if line.strip():
                    neighbours = line.split(" ")

                    for neighbour in neighbours:
                        self.v.append(int(neighbour))

                    self.offset.append(self.offset[-1] + len(neighbours))
                else:
                    self.offset.append(self.offset[-1])

    def get_degree(self, node):
        return self.offset[node + 1] - self.offset[node]

    def get_neighbors(self, node):
        return [self.v[i] for i in range(self.offset[node], self.offset[node + 1])]
