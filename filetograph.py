def csr(file):
    with open(file, "r") as f:
        vertices = int(f.readline())
        edges = int(f.readline())

        v = []
        offset = [0]

        for line in f.readlines():
            neighbours = line.split(" ")

            for neighbour in neighbours:
                v.append(int(neighbour))

            offset.append(offset[-1] + len(neighbours))

        return (v, offset)
