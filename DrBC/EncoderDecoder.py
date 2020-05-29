from DrBC import Encoder
import csr
import utils
import numpy as np
import os
import math
import time
import random
from random import randrange
import operator
import matplotlib.pyplot as plt


class EncoderDecoder:
    def __init__(self, n_iterations=10):
        self.N_ITERATIONS = n_iterations
        self.INPUT_DIMENSION = 3
        self.EMBEDDING_DIMENSION = 3
        self.L = 3  # number of encoder layers -1 (0...L)
        self.HIDDEN_NEURONS = 5  # number of decoder hidden neurons
        self.OUTPUT_DIMENSION = 1
        self.TOTAL_LAYERS = 6

        self.learning_rate = 0.0001
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.eps = 10**-8

        self.SAMPLE_SIZE = 5

        self.MIN_VERTICES = 50
        self.MAX_VERTICES = 60
        self.MIN_CONNECTION_PROB = 0.5
        self.MAX_CONNECTION_PROB = 0.9

        self.TOP_N =10

        """Encoder weights"""
        W0 = np.random.rand(self.EMBEDDING_DIMENSION, self.INPUT_DIMENSION)
        W1 = np.random.rand(self.EMBEDDING_DIMENSION, self.EMBEDDING_DIMENSION)
        W2 = np.random.rand(self.EMBEDDING_DIMENSION, self.EMBEDDING_DIMENSION)
        W3 = np.random.rand(self.EMBEDDING_DIMENSION, self.EMBEDDING_DIMENSION)

        U1 = np.random.rand(self.EMBEDDING_DIMENSION, self.EMBEDDING_DIMENSION)
        U2 = np.random.rand(self.EMBEDDING_DIMENSION, self.EMBEDDING_DIMENSION)
        U3 = np.random.rand(self.EMBEDDING_DIMENSION, self.EMBEDDING_DIMENSION)

        """Decoder weights"""
        W4 = np.random.rand(self.HIDDEN_NEURONS, self.EMBEDDING_DIMENSION)
        W5 = np.random.rand(self.OUTPUT_DIMENSION, self.HIDDEN_NEURONS)

        B4 = np.transpose(np.ones(self.HIDDEN_NEURONS))
        B5 = np.ones(self.OUTPUT_DIMENSION)
        """
        W0 = np.ones((self.EMBEDDING_DIMENSION, self.INPUT_DIMENSION))
        W1 = np.ones((self.EMBEDDING_DIMENSION, self.EMBEDDING_DIMENSION))
        W2 = np.ones((self.EMBEDDING_DIMENSION, self.EMBEDDING_DIMENSION))
        W3 = np.ones((self.EMBEDDING_DIMENSION, self.EMBEDDING_DIMENSION))
        U1 = np.ones((self.EMBEDDING_DIMENSION, self.EMBEDDING_DIMENSION))
        U2 = np.ones((self.EMBEDDING_DIMENSION, self.EMBEDDING_DIMENSION))
        U3 = np.ones((self.EMBEDDING_DIMENSION, self.EMBEDDING_DIMENSION))
        W4 = np.ones((self.HIDDEN_NEURONS, self.EMBEDDING_DIMENSION))
        W5 = np.ones((self.OUTPUT_DIMENSION, self.HIDDEN_NEURONS))

        B4 = np.transpose(np.ones(self.HIDDEN_NEURONS))
        B5 = np.ones(self.OUTPUT_DIMENSION)
        """

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

    def fit(self):
        self.mW = [None] * self.TOTAL_LAYERS  # 1st moment vector
        self.vW = [None] * self.TOTAL_LAYERS  # 2nd moment vector
        self.mU = [None] * self.TOTAL_LAYERS
        self.vU = [None] * self.TOTAL_LAYERS
        self.mB = [None] * self.TOTAL_LAYERS
        self.vB = [None] * self.TOTAL_LAYERS

        self.mW[0] = np.zeros((self.EMBEDDING_DIMENSION, self.INPUT_DIMENSION))
        self.vW[0] = np.zeros((self.EMBEDDING_DIMENSION, self.INPUT_DIMENSION))

        for i in range(1, 4):
            self.mW[i] = np.zeros((self.EMBEDDING_DIMENSION, self.EMBEDDING_DIMENSION))
            self.vW[i] = np.zeros((self.EMBEDDING_DIMENSION, self.EMBEDDING_DIMENSION))

            self.mU[i] = np.zeros((self.EMBEDDING_DIMENSION, self.EMBEDDING_DIMENSION))
            self.vU[i] = np.zeros((self.EMBEDDING_DIMENSION, self.EMBEDDING_DIMENSION))

        self.mW[4] = np.zeros((self.HIDDEN_NEURONS, self.EMBEDDING_DIMENSION))
        self.vW[4] = np.zeros((self.HIDDEN_NEURONS, self.EMBEDDING_DIMENSION))

        self.mB[4] = np.transpose(np.zeros(self.HIDDEN_NEURONS))
        self.vB[4] = np.transpose(np.zeros(self.HIDDEN_NEURONS))

        self.mW[5] = np.zeros((self.OUTPUT_DIMENSION, self.HIDDEN_NEURONS))
        self.vW[5] = np.zeros((self.OUTPUT_DIMENSION, self.HIDDEN_NEURONS))

        self.mB[5] = np.zeros(self.OUTPUT_DIMENSION)
        self.vB[5] = np.zeros(self.OUTPUT_DIMENSION)

        loss_vector = np.zeros(self.N_ITERATIONS)
        for i in range(1,self.N_ITERATIONS+1):
            print('epoch:', i)
            """Generate Random Graph"""
            
            n_nodes = randrange(self.MIN_VERTICES, self.MAX_VERTICES)
            connection_prob = random.uniform(self.MIN_CONNECTION_PROB, self.MAX_CONNECTION_PROB)
            G = csr.erdos_renyi_graph(n_nodes, connection_prob)
            """
            cur_path = os.path.dirname(__file__)

            file_path = os.path.relpath('../graphs/1.txt', cur_path)
            print(file_path)
            G = csr.CSR('graphs/1.txt')
            n_nodes = G.n_vertices
            """
            """Init encoder"""
            encoder = Encoder.Encoder(self.L, G.n_vertices, self.EMBEDDING_DIMENSION)

            """Get exact BC values"""
            bcScoresTarget = np.zeros(n_nodes)
            bcScoresDict = utils.betweenness_centrality(G)
            for key in bcScoresDict:
                bcScoresTarget[key] = bcScoresDict[key]

            topTarget = dict(sorted(bcScoresDict.items(), key=operator.itemgetter(1), reverse=True)[:self.TOP_N]) #top N high BC nodes

            """Get Predicted BC values"""
            bcScoresPredicted = np.zeros(n_nodes)
            bcScoresPredictedDict = {}
            for node in range(n_nodes):
                self.feedForward(encoder, G, node)
                bcScoresPredicted[node] = self.x[5]
                bcScoresPredictedDict[node] = self.x[5]

            topPredicted = dict(sorted(bcScoresPredictedDict.items(), key=operator.itemgetter(1), reverse=True)[:self.TOP_N])

            common_nodes = sum(n in topPredicted.keys() for n in topTarget.keys())
            print('common nodes:', common_nodes)

            """Sample node pairs"""
            nodePairs, sourceNodes = self.sampleNodes(n_nodes)

            """Calculate target and predicted differences for each pair"""
            targetDifferences = self.getTargetDifferences(nodePairs, bcScoresTarget)
            predictedDifferences = self.getPredictedDifferences(nodePairs, bcScoresPredicted)
            #print('target:', targetDifferences)
            #print('pred:', predictedDifferences)
            loss = self.lossFunction(targetDifferences, predictedDifferences)
            loss_vector[i-1] = loss
            print("Loss:", loss)

            WGradientsSum = [None] * self.TOTAL_LAYERS
            UGradientsSum = [None] * self.TOTAL_LAYERS
            BGradientsSum = [None] * self.TOTAL_LAYERS

            """Calculate Adam 'Gradients' """
            for t, p, sourceNode in zip(targetDifferences, predictedDifferences, sourceNodes):
                #print('node pair:', nodePair)
                #print('target:', t)
                #print('prediction:', p)
                WGradients, UGradients, BGradients = self.backPropagate(t, p, sourceNode, encoder, i)

                if UGradients is None:
                     print(WGradients) # in this case, WGradients is an error message

                elif WGradientsSum[0] is None:
                    WGradientsSum = WGradients
                    UGradientsSum = UGradients
                    BGradientsSum = BGradients
                else:
                    WGradientsSum = WGradientsSum + WGradients
                    UGradientsSum = UGradientsSum + UGradients
                    BGradientsSum = BGradientsSum + BGradients

            for j in range(self.TOTAL_LAYERS):
                self.W[j] = self.W[j] - (self.learning_rate * WGradientsSum[j])
                if 1 <= j <= 3:
                    self.U[j] = self.U[j] - (self.learning_rate * UGradientsSum[j])
                if j >= 4:
                    self.B[j] = self.B[j] - (self.learning_rate * BGradientsSum[j][0])

        plt.title('DrBC Loss')
        plt.ylim(2,4)
        plt.plot(range(self.N_ITERATIONS), loss_vector)
        plt.show()

    def feedForward(self, encoder, G, node):
        X = np.array([[G.get_degree(node)], [1], [1]])

        """Encoder"""
        self.x[3] = encoder.encode(G, node, X,
                                   self.L,
                                   self.W[0], self.W[1], self.W[2], self.W[3],
                                   self.U[1], self.U[2], self.U[3])

        """MLP Decoder"""
        self.z[4] = self.B[4] + np.matmul(self.W[4], np.transpose(self.x[3][0]))
        self.x[4] = ReLU(self.z[4])

        self.x[5] = (self.B[5] + np.matmul(self.W[5], self.x[4]))[0]
        #print('x5:',self.x[5])

    def backPropagate(self, t, p, source_node, encoder, i):
        error_message = "ERROR, INDETERMINATE DERIVATIVE"

        """derivatives and deltas calculation"""
        delta = [None] * self.TOTAL_LAYERS

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
        delta[currentLayer] = self.dEdx(t, p)
        #print('delta5:', delta[currentLayer])

        currentLayer -= 1  # layer 4
        dx4dz4 = None
        if np.count_nonzero(self.z[4]) > 0:
            dx4dz4 = 1
        else:  # dx4dz4 == 0
            return error_message, None, None

        delta[currentLayer] = np.transpose(self.W[currentLayer + 1]) * delta[currentLayer + 1] * dx4dz4
        #print('delta4:', delta[currentLayer])

        for currentLayer in range(self.TOTAL_LAYERS - 1, 3, -1):
            if currentLayer == 5:
                WGradients[currentLayer] = delta[currentLayer] * self.x[currentLayer - 1]  # delta * dxdW
            else:
                WGradients[currentLayer] = np.matmul(delta[currentLayer], self.x[currentLayer - 1])  # delta * dxdW

            BGradients[currentLayer] = delta[currentLayer] * 1  #delta5 * dx5dB5

            #print('WGradient',currentLayer,':',WGradients[currentLayer])
            #print('BGradient',currentLayer,':',BGradients[currentLayer])

        """Encoder Gradients"""
        currentLayer -= 1  # layer 3
        sigmoid_v = np.vectorize(sigmoid)
        tanh_v = np.vectorize(np.tanh)
        dSigmoid_v = np.vectorize(self.dSigmoid)

        # we are making these computations because we need u3's value beforehand
        hN3 = encoder.HN[currentLayer, source_node]
        h2 = encoder.H[currentLayer - 1][source_node]
        u3 = sigmoid_v(np.matmul(self.W[1],
                                 np.transpose(hN3)) + np.matmul(self.U[1], h2))

        hN2 = encoder.HN[currentLayer - 1, source_node]
        h1 = encoder.H[currentLayer - 2][source_node]
        u2 = sigmoid_v(np.matmul(self.W[1], np.transpose(hN2)) + np.matmul(self.U[1], h1))

        """This if else block corresponds to the max pool 'derivative' """
        if encoder.MaxPoolMaxLayer == 3:
            start = 3
            delta[3] = np.matmul(np.transpose(self.W[3 + 1]), delta[3 + 1])
            delta[2] = np.matmul(1 - u3, delta[2 + 1])
            delta[1] = (1 - u2) * delta[1 + 1]
        elif encoder.MaxPoolMaxLayer == 2:
            start = 2
            delta[2] = np.matmul(np.transpose(self.W[2 + 2]), delta[2 + 2])
            delta[1] = np.matmul(1 - u2, delta[1 + 1])
        else:
            start = 1
            delta[1] = np.matmul(np.transpose(self.W[1 + 3]), delta[1 + 3])
        #print('delta3:', delta[3])
        #print('delta2:', delta[2])
        #print('delta1:', delta[1])
        for currentLayer in range(start, 1, -1):
            hN = encoder.HN[currentLayer, source_node]
            h = encoder.H[currentLayer - 1][source_node]

            f = tanh_v(np.matmul(self.W[3], np.transpose(hN)) + np.matmul(self.U[3], h))
            u = sigmoid_v(np.matmul(self.W[1], np.transpose(hN)) + np.matmul(self.U[1], h))
            r = sigmoid_v(np.matmul(self.W[2], np.transpose(hN)) + np.matmul(self.U[2], h))

            WGradients[1] += np.matmul(delta[currentLayer],
                                       np.array([np.multiply(dSigmoid_v(hN), f) - np.multiply(dSigmoid_v(hN), h)]))

            UGradients[1] += np.matmul(delta[currentLayer],
                                       np.array([np.multiply(dSigmoid_v(h2), f) - np.multiply(dSigmoid_v(h), h)]))

            WGradients[2] += np.matmul(delta[currentLayer],
                                       np.array([np.matmul(u, np.multiply(dSigmoid_v(hN), h))]))

            UGradients[2] += np.matmul(delta[currentLayer],
                                       np.array([np.matmul(u, np.multiply(dSigmoid_v(h), h))]))

            WGradients[3] += np.matmul(delta[currentLayer],
                                       np.array([np.multiply(u, dSigmoid_v(hN))]))

            UGradients[3] += np.matmul(delta[currentLayer],
                                       np.array([np.matmul(u, dSigmoid_v(np.multiply(r, h)))]))
            """
            print('WGradients1:', WGradients[1])
            print('WGradients2:', WGradients[2])
            print('WGradients3:', WGradients[3])

            print('UGradients1:', UGradients[1])
            print('UGradients2:', UGradients[2])
            print('UGradients3:', UGradients[3])
            """

        currentLayer = 1  # layer 1

        dh1dw0 = np.zeros(self.EMBEDDING_DIMENSION)
        if np.count_nonzero(encoder.H[0][source_node]) > 0:
            dh1dw0 = encoder.H[0][source_node]
        else:
            return error_message, None, None

        WGradients[0] = np.matmul(np.transpose(np.array([delta[1]])), np.array([dh1dw0]))
        #print('WGradients0:', WGradients[0])
        for layer in range(self.TOTAL_LAYERS):
            self.mW[layer], self.vW[layer], finalWUpdate[layer] = self.adamGradient(self.mW[layer],
                                                                                    self.vW[layer],
                                                                                    WGradients[layer],
                                                                                    i)
            #print('adam W',layer,':',finalWUpdate[layer])

            if 1 <= layer <= 3:
                self.mU[layer], self.vU[layer], finalUUpdate[layer] = self.adamGradient(self.mU[layer],
                                                                                        self.vU[layer],
                                                                                        UGradients[layer],
                                                                                        i)
                #print('adam U', layer, ':', finalUUpdate[layer])

            if layer >= 4:
                self.mB[layer], self.vB[layer], finalBUpdate[layer] = self.adamGradient(self.mB[layer],
                                                                                        self.vB[layer],
                                                                                        np.transpose(BGradients[layer]),
                                                                                        i)
                #print('adam B', layer, ':', finalBUpdate[layer])


        #return WGradients, UGradients, BGradients
        return finalWUpdate, finalUUpdate, finalBUpdate

    def predict(self, tsv_file=None, txt_file=None, G=None):
        if tsv_file:
            G = csr.CSR(tsv_file=tsv_file)
        elif txt_file:
            G = csr.CSR(txt_file=txt_file)

        encoder = Encoder.Encoder(self.L, G.n_vertices, self.EMBEDDING_DIMENSION)
        predictedBCs = {}
        percentage = 0
        for node in range(G.n_vertices):
            self.feedForward(encoder, G, node)
            predictedBCs[node] = self.x[5]

            if node % (int(G.n_vertices/10)) == 0:
                print(percentage,'%')
                percentage += 10

        topPredicted = list(sorted(predictedBCs.items(), key=operator.itemgetter(1), reverse=True)[:self.TOP_N])
        return topPredicted

    def adamGradient(self, m, v, gradient, t):
        m = self.beta1 * m + (1 - self.beta1) * np.array(gradient)
        v = self.beta2 * v + (1 - self.beta2) * np.square(np.array(gradient))
        m_hat = m / (1 - self.beta1**t)
        v_hat = v / (1 - self.beta2**t)

        update = np.divide(m_hat, np.sqrt(v_hat) + self.eps)

        return m, v, update

    def dSigmoid(self, z):
        return math.exp(-z) / ((math.exp(-z) + 1) ** 2)

    def dEdx(self, t, p):
        return (math.exp(p) - math.exp(t))/(math.exp(p+t) + math.exp(p) + math.exp(t) + 1)

    def dxdW(self, x, z, l):
        return self.x[l - 1]

    def sampleNodes(self, n_nodes):
        nodePairs = [None] * self.SAMPLE_SIZE
        sourceNodes = [None] * self.SAMPLE_SIZE
        for i in range(self.SAMPLE_SIZE):
            nodePairs[i] = (randrange(n_nodes), randrange(n_nodes))
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
            sumLoss += (-sigmoid(t)) * math.log(sigmoid(p)) - (1 - sigmoid(t)) * math.log(1 - sigmoid(p))

        return sumLoss


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def ReLU(x):
    return np.maximum(x, 0)


def main():
    cur_path = os.path.dirname(__file__)
    file_path = os.path.relpath('../graphs/2.txt', cur_path)

    encoderDecoder = EncoderDecoder()
    # result = encoderDecoder.feedForward()

    encoderDecoder.fit()

    # return result


if __name__ == "__main__":
    start_time = time.time()
    main()
    print("running time:", str(time.time() - start_time)[:7], "seconds")
    # print("Result:", result)
