from csr import CSR
import os

cwd = os.getcwd()
graph = CSR(f"{cwd}\\graphs\\3.txt")

print(graph.v)
print(graph.offset)
print(graph.vertices)

for i in range(len(graph.offset) - 1):
    print(graph.get_degree(i))
    print(graph.get_neighbors(i))
