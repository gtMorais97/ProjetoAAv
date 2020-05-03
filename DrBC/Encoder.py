import numpy as np
from numpy import linalg as la
import math

"""Encodes graph G where G's nodes have features represented in X. 
   L is the encoder's number of layers. 
   W and U are lists of weight matrices (W contains W0...W3, U contains U1...U3). """
def encode(G, X, L, W, U):
    V = G[0]
    nNodes = len(X)
    nFeatures = len(X[0])

    """h - embeddings dos nodes em cada camada (0...L)"""
    h = np.zeros((L + 1, nNodes, len(X[0])))
    """hN - embeddings do neighborhood dos nodes nas camadas 2..L"""
    hN = np.zeros((L - 1, nNodes, len(X[0])))

    h[0] = X

    h[1] = ReLU(np.matmul(W[0], h[0]))
    h[1] = h[1] / la.norm(h[1], 2)

    for l in range(2, L):
        for i, v in enumerate(V):
            """AGGREGATE"""
            hN[l, i] = aggregateNeighborhood(G, h[l - 1], v, G.getNeighbors(v), G.getNeighborsIndex(v))
            """COMBINE"""
            h[l, i] = GRUCell(h[l - 1, i], hN[l, i], W, U)

        h[l] = h[l] / la.norm(h[l], 2)

    """z sera o embedding final, obtido atraves da funcao maxpool"""
    z = maxPool(h, nNodes, nFeatures)


"""Para cada elemento do z, escolhe o valor maximo dos embeddings de 1...L"""
def maxPool(h, nNodes, nFeatures):
    z = np.zeros((nNodes, nFeatures))

    lenH = len(h)
    for i, emb in enumerate(z):

        for j0 in range(nFeatures):
            candidates = np.zeros(lenH - 1)
            for j1 in range(1, lenH):
                candidates[j1 - 1] = h[j1, i, j0]

            emb[j0] = max(candidates)

    return z

"""
MAX POOL TEST

X = np.array([[4, 1, 2, 1], [4, 4, 5, 1], [4, 6, 2, 1]])

*Na primeira iterecao, o maxPool escolhe entre (4,1,6), que corresponde a (h[1,0,0],h[2,0,0],h[3,0,0]), 
depois entre (5,1,2), (3,3,6) e por aí em diante*

h = np.array([[[4, 1, 3, 1], [3, 3, 4, 1], [1, 5, 2, 1]],
              [[4, 5, 3, 1], [3, 4, 4, 1], [9, 5, 6, 1]],
              [[1, 1, 3, 1], [3, 6, 4, 1], [1, 2, 7, 1]],
              [[6, 2, 6, 2], [1, 5, 2, 1], [9, 3, 5, 8]]])
print(maxPool(h, len(X), len(X[0])))
"""

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def GRUCell(hLastLayer, hN, W, U):
    h = np.zeros((len(hLastLayer),len(hLastLayer[0])))

    """Vectorized Sigmoid and tanh functions"""
    sigmoid_v = np.vectorize(sigmoid)
    tanh_v = np.vectorize(np.tanh)
    for i,emb in enumerate(h):
        u = sigmoid_v(W[1]*hN[i] + U[0]*hLastLayer[i])
        r = sigmoid_v(W[2]*hN[i] + U[1]*hLastLayer[i])
        f = tanh_v(W[3]*hN[i] + U[2]*hLastLayer[i])

        h[i] = np.multiply(u,f) + np.multiply(1-u, hLastLayer[i])


    return h

"""GRU test

hN = np.array([[4, 1, 3], [3, 3, 4], [1, 5, 2]])

hLastLayer = np.array([[2, 9, 6], [2, 6, 4], [6, 1, 2]])

W = np.array([0,4,5,2])
U = np.array([5,2,3])

print(GRUCell(hLastLayer,hN,W,U))
"""


def aggregateNeighborhood(G, hLastLayer, v, neighbors):
    sum = 0
    for i in neighbors:
        sum += hLastLayer[i] / (math.sqrt(G.getDegree(v) + 1) * math.sqrt(G.getDegree(i) + 1))

    return sum


def ReLU(x):
    return np.maximum(x, 0)
