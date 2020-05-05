from DrBC import Encoder
import csr
import numpy as np
import os


class EncoderDecoder:
    def __init__(self, G, v, X):
        INPUT_DIMENSION = 3
        EMBEDDING_DIMENSION = 3
        self.L = 3

        self.G = G
        self.v = v
        self.X = X

        self.encoder = Encoder.Encoder(self.L, G.vertices, INPUT_DIMENSION)

        self.W0 = np.random.rand(INPUT_DIMENSION, EMBEDDING_DIMENSION)
        self.W1 = np.random.rand(EMBEDDING_DIMENSION, EMBEDDING_DIMENSION)
        self.W2 = np.random.rand(EMBEDDING_DIMENSION, EMBEDDING_DIMENSION)
        self.W3 = np.random.rand(EMBEDDING_DIMENSION, EMBEDDING_DIMENSION)

        self.U1 = np.random.rand(EMBEDDING_DIMENSION, EMBEDDING_DIMENSION)
        self.U2 = np.random.rand(EMBEDDING_DIMENSION, EMBEDDING_DIMENSION)
        self.U3 = np.random.rand(EMBEDDING_DIMENSION, EMBEDDING_DIMENSION)

    def encodeDecode(self):
        z = self.encoder.encode(self.G, self.v, self.X,
                                self.L,
                                self.W0, self.W1, self.W2, self.W3,
                                self.U1, self.U2, self.U3)

        return z


def main():
    cur_path = os.path.dirname(__file__)
    file_path = os.path.relpath('../graphs/2.txt', cur_path)

    G = csr.CSR(file_path)
    v = 0

    X = np.array([[G.get_degree(v)], [1], [1]])

    encoderDecoder = EncoderDecoder(G, v, X)
    print(encoderDecoder.encodeDecode())


if __name__ == "__main__":
    main()
