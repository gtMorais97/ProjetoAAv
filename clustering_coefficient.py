import random


def uniform_wedge(csr, sample_size):
    total_wedges = 0
    acc_wedge_count = []

    for v in csr.vertices:
        total_wedges += csr.get_degree(v) * (csr.get_degree(v) - 1) / 2
        acc_wedge_count.append(total_wedges)

    sum = 0
    for i in range(sample_size):
        r = random.randint(0, total_wedges - 1)
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
