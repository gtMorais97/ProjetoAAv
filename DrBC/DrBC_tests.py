from DrBC import EncoderDecoder as ed
import csr
import numpy as np
import time
import os
import psutil
import matplotlib.pyplot as plt
import utils

def test_brandes_vs_drbc_number_nodes():
    with open('DrBC/results/Brandes_vs_DrBC.txt', 'a') as f:
        number_of_nodes = np.array([10, 100, 500, 1000, 1500, 2000])
        connection_prob = 0.2
        running_time_brandes = np.zeros(len(number_of_nodes))
        running_time_drbc = np.zeros(len(number_of_nodes))
        encoder_decoder = ed.EncoderDecoder()

        for i, n_nodes in enumerate(number_of_nodes):
            G = csr.erdos_renyi_graph(n_nodes, connection_prob)

            start_time = time.time()
            encoder_decoder.predict(G=G)
            running_time_drbc[i] = float((str(time.time() - start_time)[:5]))

            start_time = time.time()
            utils.betweenness_centrality(G)
            running_time_brandes[i] = float((str(time.time() - start_time)[:5]))

        s0 = 'connection probabilities: ' + str(connection_prob)
        s1 = 'number of nodes: ' + str(number_of_nodes)
        s2 = 'DrBc Running time: ' + str(running_time_drbc)
        s3 = 'Brandes running time: ' + str(running_time_brandes)
        f.write(s0)
        f.write(s1)
        f.write(s2)
        f.write(s3)

        plt.title('Running Time for DrBC and Brandes - Erdos Renyi graphs')
        plt.plot(number_of_nodes, running_time_drbc, '-o', color="tab:green")
        plt.plot(number_of_nodes, running_time_brandes, '-o', color="tab:blue")
        plt.xlabel('number of nodes')
        plt.show()

def test_brandes_vs_drbc_connection_prob():
    with open('DrBC/results/Brandes_vs_DrBC.txt', 'a') as f:
        connection_prob = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        number_of_nodes = 500
        running_time_brandes = np.zeros(len(connection_prob))
        running_time_drbc = np.zeros(len(connection_prob))
        encoder_decoder = ed.EncoderDecoder()

        for i, prob in enumerate(connection_prob):
            G = csr.erdos_renyi_graph(number_of_nodes, prob)

            start_time = time.time()
            encoder_decoder.predict(G=G)
            running_time_drbc[i] = float((str(time.time() - start_time)[:5]))

            start_time = time.time()
            utils.betweenness_centrality(G)
            running_time_brandes[i] = float((str(time.time() - start_time)[:5]))

        s0 = 'connection probabilities: ' + str(connection_prob)
        s1 = 'number of nodes: ' + str(number_of_nodes)
        s2 = 'DrBc Running time: ' + str(running_time_drbc)
        s3 = 'Brandes running time: ' + str(running_time_brandes)
        f.write(s0)
        f.write(s1)
        f.write(s2)
        f.write(s3)

        plt.title('Running Time for DrBC and Brandes - Erdos Renyi graphs')
        plt.plot(connection_prob, running_time_drbc, '-o', color="tab:green")
        plt.plot(connection_prob, running_time_brandes, '-o', color="tab:blue")
        plt.xlabel('connection between nodes probability')
        plt.show()

def test_random_graphs():
    with open('results/Brandes_vs_DrBC.txt','a'):

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
        G = csr.erdos_renyi_graph(n_nodes, 0.3)
        process = psutil.Process(os.getpid())
        encoder_decoder.predict(G=G)
        memory[i] = process.memory_info().rss

    plt.title('Memory Used by DrBC - Erdos Renyi Graphs')
    plt.plot(number_of_nodes, memory)
    plt.show()


def test_brandes():
    number_of_nodes = np.array([10, 100, 1000, 2000])
    running_time = np.zeros(len(number_of_nodes))

    for i, n_nodes in enumerate(number_of_nodes):
        print(n_nodes)
        G = csr.erdos_renyi_graph(n_nodes, 0.3)
        start_time = time.time()
        csr.betweenness_centrality(G)
        running_time[i] = float((str(time.time() - start_time)[:5]))


    plt.title('Running Time for Brandes Algorithm - Erdos Renyi')
    plt.plot(number_of_nodes, running_time)
    plt.show()

def test_fit():
    encoder_decoder = ed.EncoderDecoder(n_iterations=10)

    start_time = time.time()

test_brandes_vs_drbc_number_nodes()