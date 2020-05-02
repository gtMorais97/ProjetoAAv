import filetograph as ftg
import os

cwd = os.getcwd()

for i in range(1, 3):
    print(f"Graph {i}")
    graph_csr = ftg.csr(f"{cwd}\\graphs\\{i}.txt")

    print(graph_csr[0])
    print(graph_csr[1])
