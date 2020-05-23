from DrBC import Encoder
import csr
import numpy as np
import os
import math
import time
from random import randrange


class EncoderDecoder:
    def __init__(self, G, v, X):
        self.INPUT_DIMENSION = 3
        self.EMBEDDING_DIMENSION = 3
        self.L = 3  # number of encoder layers -1 (0...L)
        self.HIDDEN_NEURONS = 5  # number of decoder hidden neurons
        self.OUTPUT_DIMENSION = 1
        self.TOTAL_LAYERS = 6
        self.learning_rate = 0.01
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.eps = 10 ** (-8)

        self.SAMPLE_SIZE = 5

        self.G = G
        self.v = v
        self.X = X
        self.nNodes = G.n_vertices

        self.encoder = Encoder.Encoder(self.L, G.n_vertices, self.INPUT_DIMENSION)

        """Encoder weights"""
        W0 = np.random.rand(self.INPUT_DIMENSION, self.EMBEDDING_DIMENSION)
        W1 = np.random.rand(self.EMBEDDING_DIMENSION, self.EMBEDDING_DIMENSION)
        W2 = np.random.rand(self.EMBEDDING_DIMENSION, self.EMBEDDING_DIMENSION)
        W3 = np.random.rand(self.EMBEDDING_DIMENSION, self.EMBEDDING_DIMENSION)

        U1 = np.random.rand(self.EMBEDDING_DIMENSION, self.EMBEDDING_DIMENSION)
        U2 = np.random.rand(self.EMBEDDING_DIMENSION, self.EMBEDDING_DIMENSION)
        U3 = np.random.rand(self.EMBEDDING_DIMENSION, self.EMBEDDING_DIMENSION)

        """Decoder weights"""
        W4 = np.random.rand(self.HIDDEN_NEURONS, self.INPUT_DIMENSION)
        W5 = np.random.rand(self.OUTPUT_DIMENSION, self.HIDDEN_NEURONS)

        B4 = np.transpose(np.ones(self.HIDDEN_NEURONS))
        B5 = np.ones(self.OUTPUT_DIMENSION)

        self.W = [None] * self.TOTAL_LAYERS
        self.W[0] = W0
        self.W[1] = W1
        self.W[2] = W2
        self.W[3] = W3
        self.W[4] = W4
        self.W[5] = W5

        self.U = [None] * self.TOTAL_LAYERS
        self.U[1] = U1
        self.U[2] = U2
        self.U[3] = U3

        self.B = [None] * self.TOTAL_LAYERS
        self.B[4] = B4
        self.B[5] = B5

        self.x = [None] * self.TOTAL_LAYERS
        self.z = [None] * self.TOTAL_LAYERS

    def feedForward(self):
        """Encoder"""
        self.x[3] = self.encoder.encode(self.G, self.v, self.X,
                                        self.L,
                                        self.W[0], self.W[1], self.W[2], self.W[3],
                                        self.U[1], self.U[2], self.U[3])

        """MLP Decoder"""
        self.z[4] = self.B[4] + np.matmul(self.W[4], np.transpose(self.x[3][0]))
        self.x[4] = ReLU(self.z[4])

        self.x[5] = (self.B[5] + np.matmul(self.W[5], self.x[4]))[0]

    def fit(self):
        self.mW = [None] * self.TOTAL_LAYERS  # 1st moment vector
        self.vW = [None] * self.TOTAL_LAYERS  # 2nd moment vector
        self.mU = [None] * self.TOTAL_LAYERS
        self.vU = [None] * self.TOTAL_LAYERS
        self.mB = [None] * self.TOTAL_LAYERS
        self.vB = [None] * self.TOTAL_LAYERS

        self.mW[0] = np.zeros((self.INPUT_DIMENSION, self.EMBEDDING_DIMENSION))
        self.vW[0] = np.zeros((self.INPUT_DIMENSION, self.EMBEDDING_DIMENSION))

        for i in range(1, 4):
            self.mW[i] = np.zeros((self.EMBEDDING_DIMENSION, self.EMBEDDING_DIMENSION))
            self.vW[i] = np.zeros((self.EMBEDDING_DIMENSION, self.EMBEDDING_DIMENSION))

            self.mU[i] = np.zeros((self.EMBEDDING_DIMENSION, self.EMBEDDING_DIMENSION))
            self.vU[i] = np.zeros((self.EMBEDDING_DIMENSION, self.EMBEDDING_DIMENSION))

        self.mW[4] = np.zeros((self.HIDDEN_NEURONS, self.INPUT_DIMENSION))
        self.vW[4] = np.zeros((self.HIDDEN_NEURONS, self.INPUT_DIMENSION))

        self.mB[4] = np.transpose(np.zeros(self.HIDDEN_NEURONS))
        self.vB[4] = np.transpose(np.zeros(self.HIDDEN_NEURONS))

        self.mW[5] = np.zeros((self.OUTPUT_DIMENSION, self.HIDDEN_NEURONS))
        self.vW[5] = np.zeros((self.OUTPUT_DIMENSION, self.HIDDEN_NEURONS))

        self.mB[5] = np.zeros(self.OUTPUT_DIMENSION)
        self.vB[5] = np.zeros(self.OUTPUT_DIMENSION)

        # TODO Generate random graph

        bcScoresTarget = np.zeros(self.nNodes)
        # TODO Calculate BC for all nodes

        bcScoresPredicted = np.zeros(self.nNodes)
        for node in range(self.nNodes):
            self.v = node
            self.feedForward()
            bcScoresPredicted[node] = self.x[5]

        nodePairs, sourceNodes = self.sampleNodes()

        targetDifferences = self.getTargetDifferences(nodePairs, bcScoresTarget)
        targetDifferences = np.ones(5)  # TODO delete later, cheat para poder testar
        predictedDifferences = self.getPredictedDifferences(nodePairs, bcScoresPredicted)

        error = self.lossFunction(targetDifferences, predictedDifferences)
        print("Error:", error)
        WGradientsSum = [None] * self.TOTAL_LAYERS
        UGradientsSum = [None] * self.TOTAL_LAYERS
        BGradientsSum = [None] * self.TOTAL_LAYERS

        for t, p, sourceNode in zip(targetDifferences, predictedDifferences, sourceNodes):
            WGradients, UGradients, BGradients = self.backPropagate(t, p, sourceNode)

            if UGradients is None:
                return WGradients  # in this case, WGradients is an error message

            if WGradientsSum[0] is None:
                WGradientsSum = WGradients
                UGradientsSum = UGradients
                BGradientsSum = BGradients
            else:
                WGradientsSum = WGradientsSum + WGradients
                UGradientsSum = UGradientsSum + UGradients
                BGradientsSum = BGradientsSum + BGradients
        """
        mW = self.beta1 * mW + (1 - self.beta1) * np.array(WGradientsSum)
        vW = self.beta2 * vW + (1 - self.beta2) * np.array(map(lambda x: x ** 2, WGradientsSum))
        mW = mW / (1 - self.beta1)
        vW = vW / (1 - self.beta2)
        updateW = (mW / (map(lambda x: np.sqrt(x), vW) + self.eps))
        """

        for i in range(self.TOTAL_LAYERS):
            self.W[i] -= self.learning_rate * WGradientsSum[i]
            if 1 <= i <= 3:
                self.U[i] -= self.learning_rate * UGradientsSum[i]
            if i >= 4:
                self.B[i] = self.B[i] - np.transpose(self.learning_rate * BGradientsSum[i])

    def backPropagate(self, t, p, sourceNode):
        error_message = "ERROR, INDETERMINATE DERIVATIVE"

        """derivatives and deltas calculation"""
        deltas = [None] * self.TOTAL_LAYERS

        """Gradients"""
        WGradients = [None] * self.TOTAL_LAYERS
        UGradients = [None] * self.TOTAL_LAYERS
        BGradients = [None] * self.TOTAL_LAYERS

        finalWUpdate = [None] * self.TOTAL_LAYERS
        finalUUpdate = [None] * self.TOTAL_LAYERS
        finalBUpdate = [None] * self.TOTAL_LAYERS

        for g in range(1, 4):
            WGradients[g] = np.zeros((self.EMBEDDING_DIMENSION, self.EMBEDDING_DIMENSION))
            UGradients[g] = np.zeros((self.EMBEDDING_DIMENSION, self.EMBEDDING_DIMENSION))

        """Decoder Gradients"""
        currentLayer = self.TOTAL_LAYERS - 1  # layer 5
        deltas[currentLayer] = self.dEdx(t, p)

        currentLayer -= 1  # layer 4
        dx4dz4 = None
        if np.count_nonzero(self.z[4]) > 0:
            dx4dz4 = 1
        else:  # dx4dz4 == 0
            return error_message, None, None

        deltas[currentLayer] = np.transpose(self.W[currentLayer + 1]) * deltas[currentLayer + 1] * dx4dz4

        for currentLayer in range(self.TOTAL_LAYERS - 1, 3, -1):
            if currentLayer == 5:
                WGradients[currentLayer] = deltas[currentLayer] * self.x[currentLayer - 1]  # delta * dxdW
            else:
                WGradients[currentLayer] = np.matmul(deltas[currentLayer], self.x[currentLayer - 1])  # delta * dxdW
            BGradients[currentLayer] = np.array(deltas[currentLayer] * 1)  # delta5 * dx5dB5

        """Encoder Gradients"""
        currentLayer -= 1  # layer 3
        sigmoid_v = np.vectorize(sigmoid)
        tanh_v = np.vectorize(np.tanh)
        dSigmoid_v = np.vectorize(self.dSigmoid)

        # we are making these computations because we need u3's value beforehand
        hN3 = self.encoder.HN[currentLayer, sourceNode]
        h2 = self.encoder.H[currentLayer - 1, sourceNode]
        u3 = sigmoid_v(np.matmul(self.W[1],
                                 np.transpose(hN3)) + np.matmul(self.U[1], h2))

        hN2 = self.encoder.HN[currentLayer - 1, sourceNode]
        h1 = self.encoder.H[currentLayer - 2, sourceNode]
        u2 = sigmoid_v(np.matmul(self.W[1],
                                 np.transpose(hN2)) + np.matmul(self.U[1], h1))

        """This if else block corresponds to the max pool 'derivative' """
        if self.encoder.MaxPoolMaxLayer == 3:
            start = 3
            deltas[3] = np.matmul(np.transpose(self.W[3 + 1]), deltas[3 + 1])
            deltas[2] = np.matmul(1 - u3, deltas[2 + 1])
            deltas[1] = (1 - u2) * deltas[1 + 1]
        elif self.encoder.MaxPoolMaxLayer == 2:
            start = 2
            deltas[2] = np.matmul(np.transpose(self.W[2 + 2]), deltas[2 + 2])
            deltas[1] = np.matmul(1 - u2, deltas[1 + 1])
        else:
            deltas[1] = np.matmul(np.transpose(self.W[1 + 3]), deltas[1 + 3])

        for currentLayer in range(start, 1, -1):
            hN = self.encoder.HN[currentLayer, sourceNode]
            h = self.encoder.H[currentLayer - 1, sourceNode]

            f = tanh_v(np.matmul(self.W[3], np.transpose(hN)) + np.matmul(self.U[3], h))
            u = sigmoid_v(np.matmul(self.W[1], np.transpose(hN)) + np.matmul(self.U[1], h))
            r = sigmoid_v(np.matmul(self.W[2], np.transpose(hN)) + np.matmul(self.U[2], h))

            WGradients[1] += np.matmul(deltas[currentLayer],
                                       np.array([np.multiply(dSigmoid_v(hN), f) - np.multiply(dSigmoid_v(hN), h)]))

            UGradients[1] += np.matmul(deltas[currentLayer],
                                       np.array([np.multiply(dSigmoid_v(h2), f) - np.multiply(dSigmoid_v(h), h)]))

            WGradients[2] += np.matmul(deltas[currentLayer],
                                       np.array([np.matmul(u, np.multiply(dSigmoid_v(hN), h))]))

            UGradients[2] += np.matmul(deltas[currentLayer],
                                       np.array([np.matmul(u, np.multiply(dSigmoid_v(h), h))]))

            WGradients[3] += np.matmul(deltas[currentLayer],
                                       np.array([np.multiply(u, dSigmoid_v(hN))]))

            UGradients[3] += np.matmul(deltas[currentLayer],
                                       np.array([np.matmul(u, dSigmoid_v(np.multiply(r, h)))]))

        currentLayer = 1  # layer 1

        dh1dw0 = np.zeros(self.EMBEDDING_DIMENSION)
        if np.count_nonzero(self.encoder.H[0, sourceNode]) > 0:
            dh1dw0 = self.encoder.H[0, sourceNode]
        else:
            return error_message, None, None

        WGradients[0] = np.matmul(deltas[1], dh1dw0)

        for layer in range(self.TOTAL_LAYERS):
            self.mW[layer], self.vW[layer], finalWUpdate[layer] = self.adamGradient(self.mW[layer],
                                                                                    self.vW[layer],
                                                                                    WGradients[layer])

            if 1 <= layer <= 3:
                self.mU[layer], self.vU[layer], finalUUpdate[layer] = self.adamGradient(self.mU[layer],
                                                                                        self.vU[layer],
                                                                                        UGradients[layer])

            if layer >= 4:
                self.mB[layer], self.vB[layer], finalBUpdate[layer] = self.adamGradient(self.mB[layer],
                                                                                        self.vB[layer],
                                                                                        BGradients[layer])

        return finalWUpdate, finalUUpdate, finalBUpdate

    def adamGradient(self, m, v, gradient):
        m = self.beta1 * m + (1 - self.beta1) * np.array(gradient)
        v = self.beta2 * v + (1 - self.beta2) * np.square(np.array(gradient))
        m = m / (1 - self.beta1)
        v = v / (1 - self.beta2)
        update = m / (np.sqrt(v) + self.eps)

        return m, v, update

    def dSigmoid(self, z):
        return math.exp(-z) / ((math.exp(-z) + 1) ** 2)

    def dEdx(self, t, x):
        return ((-sigmoid(t)) / (x + 0.001)) - ((1 - sigmoid(t)) / (x - 1))

    def dxdW(self, x, z, l):
        return self.x[l - 1]

    def sampleNodes(self):
        nodePairs = [None] * self.SAMPLE_SIZE
        sourceNodes = [None] * self.SAMPLE_SIZE
        for i in range(self.SAMPLE_SIZE):
            nodePairs[i] = (randrange(self.nNodes), randrange(self.nNodes))
            sourceNodes[i] = nodePairs[i][0]

        return nodePairs, sourceNodes

    def getPredictedDifferences(self, nodePairs, bcScoresPredicted):
        predictedDifferences = [None] * self.SAMPLE_SIZE
        for i, pair in enumerate(nodePairs):
            predictedDifferences[i] = bcScoresPredicted[pair[0]] - bcScoresPredicted[pair[1]]

        return predictedDifferences

    def getTargetDifferences(self, nodePairs, bcScoresTarget):
        targetDifferences = [None] * self.SAMPLE_SIZE
        for i, pair in enumerate(nodePairs):
            targetDifferences[i] = bcScoresTarget[pair[0]] - bcScoresTarget[pair[1]]

        return targetDifferences

    """adapted cross entropy - uses sigmoid to turn t into a probability and ReLU to make sure p is larger than 0"""

    def lossFunction(self, targetDifferences, predictedDifferences):
        sumLoss = 0
        for t, p in zip(targetDifferences, predictedDifferences):
            sumLoss += (-sigmoid(t)) * math.log(ReLU(p) + 0.001) - (1 - sigmoid(t)) * math.log(1 - ReLU(p))

        return sumLoss


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def ReLU(x):
    return np.maximum(x, 0)


def main():
    cur_path = os.path.dirname(__file__)
    file_path = os.path.relpath('../graphs/2.txt', cur_path)

    G = csr.CSR(file_path)
    v = 1

    X = np.array([[G.get_degree(v)], [1], [1]])

    encoderDecoder = EncoderDecoder(G, v, X)
    # result = encoderDecoder.feedForward()

    encoderDecoder.fit()

    # return result


if __name__ == "__main__":
    start_time = time.time()
    main()
    print("running time:", str(time.time() - start_time)[:7], "seconds")
    # print("Result:", result)
