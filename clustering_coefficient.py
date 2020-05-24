import os
import csr
import random


def binary_search(r, acc_wedge_count):
    pass


def generate_random_wedge(csr, index):
    pass


def c(csr, w):
    return w[0] in csr.get_neighbors(w[2])


def uniform_wedge(csr, sample_size):
    total_wedges = 0
    acc_wedge_count = {}

    for v in csr.vertices:
        acc_wedge_count[v] = total_wedges
        if csr.offset[v + 1] - csr.offset[v] > 1:
            total_wedges += csr.get_degree[v] * (csr.get_degree[v] - 1) / 2

    sum = 0
    for i in range(sample_size):
        r = random.randint(0, total_wedges)
        index = binary_search(r, acc_wedge_count)
        w = generate_random_wedge(csr, index)
        sum += c(csr, w)

    return total_wedges * sum / (3 * sample_size)


def uniform_edge():
    pass


def uniform_vertex():
    pass


if __name__ == "__main__":
    print("cc")
