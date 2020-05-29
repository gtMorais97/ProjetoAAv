"""
This module contains a implementation of the triangle count approximation algorithms
for large graphs as presented in Siddharth Bhatia's paper "Approximate Triangle Count
and Clustering Coefficient" which can be found in the following link:

https://dl.acm.org/doi/10.1145/3183713.3183715

These algorithms were implemented having a CSR representation of the graphs in mind.
"""

import random


def uniform_wedge(csr, sample_size):
    """
    This is a implementation of the Uniform Wedge algorithm as presented in the
    aforementioned paper.
    """
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
    """
    This is a implementation of the Uniform Edge algorithm as presented in the
    aforementioned paper.
    """
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
    """
    This is a implementation of the Uniform Vertex algorithm as presented in the
    aforementioned paper.
    """
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


def search(r, acc_wedge_count):
    """
    This is a simple linear search method.
    If it is used in the context of the uniform wedge algorithm then it traverses
    the accumulative wedge count array and returns the index of the vertex that is
    the center of a given wedge.
    If it it used in the context of the uniform edge algorithm then it traverses
    the offset vertex of the CSR graph representation and retuns the index of the
    vertex that is connected to a given edge.
    Time complexity:    O(n)
    Space complexity:   O(n)
    """
    for i in range(len(acc_wedge_count)):
        if r <= acc_wedge_count[i]:
            return i


def generate_random_wedge(csr, index):
    """
    This method generates a random wedge from the graph 'csr' with vertex 'index'
    as its center. It starts by collecting all the neighbors of vertex 'index' and
    in an array and then randomizes its order. By taking the first two elements of
    the randomized neighbors array, it generates a random wedge with vertex 'index'
    as its center.
    It returns an array of size three with vertex 'index' as its middle element
    and the two random neighbors as the end points of the array.
    Time complexity:    O(dv), where dv is the degree of vertex 'index'
    Time complexity:    O(dv), where dv is the degree of vertex 'index'
    """
    neighbors = csr.get_neighbors(index)
    random.shuffle(neighbors)
    return [neighbors.pop(), index, neighbors.pop()]


def c(csr, w):
    """
    This methos verifies if a given wedge is closed or not.
    It does that by verifying if the vertices located at the ending points
    of the wedge are each other neighbors, by checking if the vertex w[0] is
    contained in the list of neighbors of the vertex w[2]
    It returns 1 if they are neighbors and 0 otherwise.
    Time complexity:    O(dv), where dv is the degree of the vertex w[2]
    Time complexity:    O(dv), where dv is the degree of the vertex w[2]
    """
    return 1 if w[0] in csr.get_neighbors(w[2]) else 0
