import pandas as pd
from tqdm import tqdm
import networkx as nx
from networkx.generators.atlas import graph_atlas_g

def enumerate_graphlets(graphlet_size=4):
    graphlets = {graph_size: [] for graph_size in range(2, graphlet_size+1)}
    for graph in graph_atlas_g():
        if graph.number_of_nodes() >= 2 and graph.number_of_nodes() <= graphlet_size:
            if nx.is_connected(graph):
                graphlets[graph.number_of_nodes()].append(graph)

    role_idx = 0
    categories = dict()
    for graph_size, graphs in graphlets.items():
        
        categories[graph_size] = dict()
        for graph_idx, graph in enumerate(graphs):
            
            categories[graph_size][graph_idx] = dict()
            degrees = list(set([graph.degree(node) for node in graph.nodes()]))
            for deg in degrees:
                categories[graph_size][graph_idx][deg] = role_idx
                role_idx += 1
    return graphlets, categories, role_idx + 1

def create_node_subsets(nx_graph, graphlet_size):
    print("\nEnumerating subgraphs...")

    node_subsets = dict()
    
    node_set = [[edge[0], edge[1]] for edge in nx_graph.edges()]
    node_subsets[2] = node_set

    cur_node_set = dict()
    for graph_size in range(3, graphlet_size+1):
        print("Enumerating graphlets with size: " +str(graph_size) + ".")
        
        for nodes in tqdm(node_set):
            for node in nodes:
                for neigh in nx_graph.neighbors(node):
                    nodes_neigh = nodes + [neigh]
                    if len(set(nodes_neigh)) == graph_size:
                        nodes_neigh.sort()
                        cur_node_set[tuple(nodes_neigh)] = 1
        node_set = [list(k) for k, v in cur_node_set.items()]
        node_subsets[graph_size] = node_set
        cur_node_set = dict()
    return node_subsets

def cal_graph_degree_vector(nx_graph, graphlets, categories, num_roles, node_subsets):
    print("\nCounting graphlet degree vector...")

    graph_degree_vector = {node: {idx: 0 for idx in range(num_roles)} for node in nx_graph.nodes()}
    for graph_size, node_set in node_subsets.items():
        print("Counting graphlets degree vector with size: " +str(graph_size) + ".")

        graphs = graphlets[graph_size]
        for nodes in tqdm(node_set):
            subgraph = nx_graph.subgraph(nodes)
            for graph_idx, graph in enumerate(graphs):
                if nx.is_isomorphic(subgraph, graph):
                    for node in subgraph.nodes():
                        graph_degree_vector[node][categories[graph_size][graph_idx][subgraph.degree(node)]] += 1
                    # there is only one isomorphic graph
                    break
    return graph_degree_vector

if __name__ == "__main__":
    # from dgl.data import CoraGraphDataset
    # data = CoraGraphDataset()
    # nx_graph = data[0].to_networkx().to_undirected()

    # graphlets, categories, num_roles = enumerate_graphlets(graphlet_size=4)
    # node_subsets = create_node_subsets(nx_graph, graphlet_size=4)    
    # graph_degree_vector = cal_graph_degree_vector(nx_graph, graphlets, categories, num_roles, node_subsets)

    # motifs = [[nid]+[graph_degree_vector[nid][role_idx] for role_idx in range(num_roles)] for nid in nx_graph.nodes()]
    # motifs = pd.DataFrame(motifs)
    # motifs.columns = ["nid"] + ["role_"+str(role_idx) for role_idx in range(num_roles)]

    # motifs.to_csv("./cora.gdv", index=None, sep=" ")
    # motifs = pd.read_csv("./cora_motifs.csv")

    # extract graphlet vector degree using orca (https://file.biolab.si/biolab/supp/orca/)
    import os 
    from data import MovieLens

    data_name = "ml-100k"
    movielens = MovieLens(data_name, testing=True)
    nx_graph = movielens.train_graph.to_networkx().to_undirected()
    
    nx.write_edgelist(nx_graph, "{}.edgelist".format(data_name), data=False)
    os.system("sed -i '1i\{} {}' {}.edgelist".format(
        nx_graph.number_of_nodes(), nx_graph.number_of_edges(), data_name))
    os.system("./orca/orca.exe 4 {}.edgelist {}.gdv".format(data_name, data_name))
