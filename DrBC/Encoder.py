import numpy as np
from numpy import linalg as la
import math





class Encoder:

    def __init__(self, L, nNodes, nFeatures):
        self.nFeatures = nFeatures
        self.nNodes = nNodes
        self.H = np.zeros((L + 1, nNodes, nFeatures))  # stores the values for all nodes

    """Encodes G's node reprensented by X. 
       L is the encoder's number of layers. 
       W0..3 and U0..2 are 3x3 weight matrices. """

    def encode(self, G, v, X, L, W0, W1, W2, W3, U1, U2, U3):

        """h - embeddings dos nodes em cada camada (0...L)"""
        h = np.zeros((L + 1, self.nFeatures))
        """hN - embeddings do neighborhood dos nodes nas camadas 2..L (tem a mesma dimensão por uma questão de simplicidade,
         os 2 primeiros elementos vão ficar a 0) """
        hN = np.zeros((L + 1, self.nFeatures))

        h[0] = np.transpose(X)
        self.H[0, v] = h[0]

        for node in range(self.nNodes):
            self.H[1, node] = np.transpose(ReLU(np.matmul(W0, np.transpose(self.H[0, node]))))
            if self.H[1, node].any():  # se nao for um vetor de zeros
                self.H[1, node] = self.H[1, node] / la.norm(self.H[1, node], 2)

        h[1] = self.H[1, v]

        for l in range(2, L + 1):
            for node in range(self.nNodes):
                """AGGREGATE"""
                hN[l] = self.aggregateNeighborhood(G, node, G.get_neighbors(node), l)
                """COMBINE"""
                self.H[l, node] = self.GRUCell(self.H[l - 1, node], hN[l], W1, W2, W3, U1, U2, U3)

            self.H[l, v] = self.H[l, v] / la.norm(self.H[l, v], 2)
            h[l] = self.H[l, v]

        """z sera o embedding final, obtido atraves da funcao maxpool"""
        z = maxPool(h[1:], self.nFeatures)
        return z

    """Store the values of h so they can be used as neighbor values later
    def storeH(self, h, v):
        for i, hl in enumerate(h, start=1):
            self.H[i, v] = hl
    """

    """Para cada elemento do z, escolhe o valor maximo dos embeddings de 1...L"""

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

    def GRUCell(self, hLastLayer, hN, W1, W2, W3, U1, U2, U3):
        h = np.zeros(self.nFeatures)

        """Vectorized Sigmoid and tanh functions"""
        sigmoid_v = np.vectorize(sigmoid)
        tanh_v = np.vectorize(np.tanh)

        hNt = np.transpose(hN)
        hLastLayert = np.transpose(hLastLayer)

        u = sigmoid_v(np.matmul(W1, hNt) + np.matmul(U1, hLastLayert))
        r = sigmoid_v(np.matmul(W2, hNt) + np.matmul(U2, hLastLayert))
        f = tanh_v(np.matmul(W3, hNt) + np.matmul(U3, hLastLayert))

        h = np.multiply(u, f) + np.multiply(1 - u, hLastLayert)

        return np.transpose(h)

    """
    GRU test
    
    hN = np.array([[4, 1, 3], [3, 3, 4], [1, 5, 2]])
    
    hLastLayer = np.array([[2, 9, 6], [2, 6, 4], [6, 1, 2]])
    
    W = np.array([0,4,5,2])
    U = np.array([5,2,3])
    
    print(GRUCell(hLastLayer,hN,W,U))
    """

    def aggregateNeighborhood(self, G, v, neighbors, l):
        hN = np.zeros(self.nFeatures)
        for n in neighbors:
            """
            if not self.H[currentLayer - 1, n].any():  # se o neighbor ainda nao foi encoded, temos de o fazer
                Xn = np.array([[G.get_degree(n)], [1], [1]])  # neighbor's feature vector X
                self.encode(G, n, Xn, L, W0, W1, W2, W3, U1, U2, U3)
            """
            hN += self.H[l - 1, n] / (math.sqrt(G.get_degree(v) + 1) * math.sqrt(G.get_degree(n) + 1))

        return hN


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def ReLU(x):
    return np.maximum(x, 0)


def maxPool(h, nFeatures):
    z = np.zeros(nFeatures)

    ht = np.transpose(h)

    for i, candidates in enumerate(ht):
        print("candidates for z", i, ": ", candidates)
        z[i] = max(candidates)

    return z