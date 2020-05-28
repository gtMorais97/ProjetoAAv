from DrBC import EncoderDecoder as ed
import csr
import numpy as np
import time
import os
import psutil
import matplotlib.pyplot as plt

def test_random_graphs():
    number_of_nodes = np.array([10, 100, 1000])
    running_time = np.zeros(len(number_of_nodes))
    encoder_decoder = ed.EncoderDecoder()

    for i,n_nodes in enumerate(number_of_nodes):
        G = csr.erdos_renyi_graph(n_nodes, 0.3)
        start_time = time.time()
        encoder_decoder.predict(G=G)
        running_time[i] = float((str(time.time() - start_time)[:5]))

    plt.title('Running Time for DrBC - random graphs')
    plt.plot(number_of_nodes, running_time)
    plt.show()


def test_real_graphs():
    files = os.listdir('tsv_graphs')
    encoder_decoder = ed.EncoderDecoder()

    number_of_nodes = np.zeros(len(files))
    running_time = np.zeros(len(files))

    for i,file in enumerate(files):
        full_path = 'tsv_graphs/' + file
        G = csr.CSR(tsv_file=full_path)
        number_of_nodes[i] = G.n_vertices
        start_time = time.time()
        encoder_decoder.predict(G=G)
        running_time[i] = float((str(time.time() - start_time)[:5]))

    running_time = sorted(running_time)
    number_of_nodes = sorted(number_of_nodes)

    plt.title('Running Time for DrBC - Real Graphs')
    plt.plot(number_of_nodes, running_time)
    plt.show()

def memory_test():
    number_of_nodes = np.array([10, 50, 100, 150, 200])
    memory = np.zeros(len(number_of_nodes))

    encoder_decoder = ed.EncoderDecoder()

    for i, n_nodes in enumerate(number_of_nodes):
        G = utils.erdos_renyi_graph(n_nodes, 0.3)
        process = psutil.Process(os.getpid())
        encoder_decoder.predict(G=G)
        memory[i] = process.memory_info().rss

    plt.title('Memory Used by DrBC - Real Graphs')
    plt.plot(number_of_nodes, memory)
    plt.show()


def test_brandes():
    number_of_nodes = np.array([10, 100, 1000, 2000])
    running_time = np.zeros(len(number_of_nodes))

    for i, n_nodes in enumerate(number_of_nodes):
        print(n_nodes)
        G = utils.erdos_renyi_graph(n_nodes, 0.3)
        start_time = time.time()
        utils.betweenness_centrality(G)
        running_time[i] = float((str(time.time() - start_time)[:5]))


    plt.title('Running Time for Brandes Algorithm - Erdos Renyi')
    plt.plot(number_of_nodes, running_time)
    plt.show()

def test_fit():
    encoder_decoder = ed.EncoderDecoder(n_iterations=10)

    start_time = time.time()

test_brandes()