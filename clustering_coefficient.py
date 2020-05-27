import os
import csr
import random
import random_graph
import utils


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
            total_wedges += csr.get_degree(v) * (csr.get_degree(v) - 1) / 2
        acc_wedge_count.append(total_wedges)

    sum = 0
    for i in range(sample_size):
        r = random.randint(0, total_wedges)
        index = search(r, acc_wedge_count)
        w = generate_random_wedge(csr, index)
        sum += c(csr, w)

    return (total_wedges * sum) / (3 * sample_size)


def uniform_edge(csr, sample_size):
    s1_estimate = 0

    for i in range(sample_size):
        r = random.randint(0, csr.n_vertices - 1)
        if csr.get_degree(r) == 1:
            s1_estimate += 1

    s1_estimate = csr.n_vertices * s1_estimate / sample_size

    sum = 0
    index = None
    for i in range(sample_size):
        while not index or csr.get_degree(index) <= 1:
            r = random.randint(0, csr.n_edges - 1)
            index = search(r, csr.offset)

        w = generate_random_wedge(csr, index)
        sum += c(csr, w) * (csr.get_degree(index) - 1)

    return ((2 * csr.n_edges - s1_estimate) * sum) / (6 * sample_size)


def uniform_vertex(csr, sample_size):
    s_estimate = 0

    for i in range(sample_size):
        r = random.randint(0, csr.n_vertices - 1)
        if csr.get_degree(r) < 2:
            s_estimate += 1

    s_estimate = csr.n_vertices * s_estimate / sample_size

    sum = 0
    index = None
    for i in range(sample_size):
        while not index or csr.get_degree(index) < 2:
            index = random.randint(0, csr.n_vertices - 1)

        w = generate_random_wedge(csr, index)
        sum += c(csr, w) * ((csr.get_degree(index) - 1) ** 2) / 2

    return ((csr.n_vertices - s_estimate) * sum) / (3 * sample_size)


if __name__ == "__main__":
    cwd = os.getcwd()
    # array = utils.tsv_to_array(f"{cwd}\\tsv_graphs\\test")
    # print(array)
    graph = csr.CSR(tsv_file=f"{cwd}\\tsv_graphs\\out.com-amazon")
    # graph = random_graph.erdos_renyi_graph(20, .6)
    # print(graph.n_vertices)
    # print(graph.n_edges)
    # print(graph.vertices)
    # print(graph.v)
    # print(graph.offset)
    uw_mean = 0
    ue_mean = 0
    uv_mean = 0
    iterations = 5
    for i in range(iterations):
        uw = uniform_wedge(graph, 300)
        ue = uniform_edge(graph, 300)
        uv = uniform_vertex(graph, 300)

        uw_mean += uw
        ue_mean += ue
        uv_mean += uv

        print(f"Iteration:\t{i}")
        print(f"Uniform wedge:\t{uw}")
        print(f"Uniform edge:\t{ue}")
        print(f"Uniform vertex:\t{uv}")
    print(f"Uniform wedge mean:\t{uw_mean / iterations}")
    print(f"Uniform edge mean:\t{ue_mean / iterations}")
    print(f"Uniform vertex mean:\t{uv_mean / iterations}")
