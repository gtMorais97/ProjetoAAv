import os
import csr
import random
import random_graph


def binary_search(r, acc_wedge_count):
    left = 0
    right = len(acc_wedge_count) - 1
    while left <= right:
        mid = int(left + ((right - left) / 2))
        if acc_wedge_count[mid] == r:
            return mid
        elif r < acc_wedge_count[mid]:
            right = mid - 1
        else:
            left = mid + 1
    return None


def search(r, acc_wedge_count):
    for i in range(len(acc_wedge_count)):
        if r <= acc_wedge_count[i]:
            return i


def generate_random_wedge(csr, index):
    neighbors = csr.get_neighbors(index)
    random.shuffle(neighbors)
    return [neighbors.pop(), index, neighbors.pop()]


def c(csr, w):
    return 1 if w[0] in csr.get_neighbors(w[2]) else 0


def uniform_wedge(csr, sample_size):
    total_wedges = 0
    acc_wedge_count = []

    for v in csr.vertices:
        if csr.get_degree(v) > 1:
            total_wedges += int(csr.get_degree(v) *
                                (csr.get_degree(v) - 1) / 2)
        acc_wedge_count.append(total_wedges)

    sum = 0
    for i in range(sample_size):
        r = random.randint(0, total_wedges)
        index = search(r, acc_wedge_count)
        w = generate_random_wedge(csr, index)
        sum += c(csr, w)

    return total_wedges * sum / (3 * sample_size)


def uniform_edge():
    pass


def uniform_vertex():
    pass


if __name__ == "__main__":
    cwd = os.getcwd()
    # graph = csr.CSR(file=f"{cwd}\\graphs\\4.txt")
    graph = random_graph.erdos_renyi_graph(20, .6)
    print(graph.v)
    print(graph.offset)
    print(uniform_wedge(graph, 15))
