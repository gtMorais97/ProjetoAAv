def tsv_to_array(tsv_file):
    with open(tsv_file, "r", encoding='utf-8') as tsv:
        tsv.readline()
        info = tsv.readline().split(" ")
        n_vertices = int(info[2])
        n_edges = int(info[1])

        edges = {i: [] for i in range(n_vertices)}

        for line in tsv.readlines():
            info = line.split(" ")
            vertex_from = int(info[0]) - 1
            vertex_to = int(info[1]) - 1

            edges[vertex_from].append(vertex_to)
            edges[vertex_to].append(vertex_from)

        return (n_vertices, n_edges, edges)