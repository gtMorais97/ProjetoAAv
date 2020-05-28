from DrBC import EncoderDecoder as ed
import random_graph as rg
import csr
import numpy as np
import time
import os
import matplotlib.pyplot as plt

def test_random_graphs():
    number_of_nodes = np.array([10, 100, 1000, 10000])
    running_time = np.zeros(len(number_of_nodes))
    encoder_decoder = ed.EncoderDecoder()

    for i,n_nodes in enumerate(number_of_nodes):
        G = rg.erdos_renyi_graph(n_nodes, 0.3)
        start_time = time.time()
        encoder_decoder.predict(G=G)
        running_time[i] = (time.time() - start_time)[:5]

    plt.title('Running Time for DrBC - random graphs')
    plt.plot(number_of_nodes, running_time)
    plt.show()


def test_real_graphs():
    files = os.listdir('tsv_graphs')
    encoder_decoder = ed.EncoderDecoder()

    number_of_nodes = np.zeros(len(files))
    running_time = np.zeros(len(files))

    for i,file in enumerate(files):
        G = csr.CSR(tsv_file=file)
        number_of_nodes[i] = G.n_vertices
        start_time = time.time()
        encoder_decoder.predict(G=G)
        running_time[i] = (time.time() - start_time)[:5]

    plt.title('Running Time for DrBC - Real Graphs')
    plt.plot(number_of_nodes, running_time)
    plt.show()