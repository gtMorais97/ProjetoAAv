import numpy as np
from numpy import linalg as la
import math
from DrBC import EncoderDecoder as ed


class Encoder:

    def __init__(self, L, nNodes, nFeatures):
        self.nFeatures = nFeatures
        self.nNodes = nNodes
        self.H = np.zeros((L + 1, nNodes, nFeatures))  # stores the values for all nodes
        self.HN = np.zeros((L + 1, nNodes, self.nFeatures))
        self.MaxPoolMaxLayer = 1

    """Encodes G's node represented by X. 
       L is the encoder's number of layers. 
       W0..3 and U0..2 are 3x3 weight matrices. """

    def encode(self, G, v, X, L, W0, W1, W2, W3, U1, U2, U3):

        """h - embeddings dos nodes em cada camada (0...L)"""
        h = np.zeros((L + 1, self.nFeatures))
        """hN - embeddings do neighborhood dos nodes nas camadas 2..L (tem a mesma dimensão por uma questao de simplicidade,
         os 2 primeiros elementos vão ficar a 0) """
        hN = np.zeros((L + 1, self.nFeatures))

        h[0] = np.transpose(X)
        self.H[0, v] = h[0]

        for node in range(self.nNodes):
            self.H[1, node] = np.transpose(ed.ReLU(np.matmul(W0, np.transpose(self.H[0, node]))))
            if self.H[1, node].any():  # se nao for um vetor de zeros
                self.H[1, node] = self.H[1, node] / la.norm(self.H[1, node], 2)

        h[1] = self.H[1, v]

        for l in range(2, L + 1):
            for node in range(self.nNodes):
                """AGGREGATE"""
                self.HN[l, node] = self.aggregateNeighborhood(G, node, G.get_neighbors(node), l)
                """COMBINE"""
                self.H[l, node] = self.GRUCell(self.H[l - 1, node], self.HN[l, node], W1, W2, W3, U1, U2, U3)

            self.H[l, v] = self.H[l, v] / la.norm(self.H[l, v], 2)
            h[l] = self.H[l, v]

        """z sera o embedding final, obtido atraves da funcao maxpool"""
        z = self.maxPool(h[1:], self.nFeatures)
        return [z]

    def GRUCell(self, hLastLayer, hN, W1, W2, W3, U1, U2, U3):
        h = np.zeros(self.nFeatures)

        """Vectorized Sigmoid and tanh functions"""
        sigmoid_v = np.vectorize(ed.sigmoid)
        tanh_v = np.vectorize(np.tanh)

        hNt = np.transpose(hN)
        hLastLayert = np.transpose(hLastLayer)

        u = sigmoid_v(np.matmul(W1, hNt) + np.matmul(U1, hLastLayert))
        r = sigmoid_v(np.matmul(W2, hNt) + np.matmul(U2, hLastLayert))
        f = tanh_v(np.matmul(W3, hNt) + np.matmul(U3, np.multiply(r,hLastLayert)))

        h = np.multiply(u, f) + np.multiply(1 - u, hLastLayert)

        return np.transpose(h)

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

    def maxPool(self, h, nFeatures):
        z = np.zeros(nFeatures)

        ht = np.transpose(h)

        for i, candidates in enumerate(ht):
            z[i] = max(candidates)
            index = np.where(candidates == z[i])[0]
            if index+1 > self.MaxPoolMaxLayer:
                self.MaxPoolMaxLayer = index+1

        return z
