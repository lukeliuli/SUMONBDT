from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import load_iris
from scipy.cluster.hierarchy import dendrogram
from matplotlib import pyplot as plt
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

'''
G = nx.Graph()

G.add_node(1)

G.add_nodes_from([2, 3])

G.add_edge(1, 2)
e = (2, 3)
G.add_edge(*e)  # unpack edge tuple*

G.add_edges_from([(1, 2), (1, 3)])

G.add_node("spam")        # adds node "spam"
G.add_nodes_from("spam")  # adds 4 nodes: 's', 'p', 'a', 'm'
G.add_edge(3, 'm')
print(G.number_of_nodes())
print(G.number_of_edges())
print(list(G.nodes))
print(list(G.edges))

G.remove_node(2)
G.remove_nodes_from("spam")
print(list(G.edges))
G.remove_edge(1, 3)
print(list(G.edges))
G.clear()

G = nx.Graph([(1, 2, {"color": "yellow"})])
print(list(G.nodes))
print(list(G.edges))

print(G.adj[1])
print(G.adj[1][2])
print(G.adj[1][2]['color'])
print(G.edges[1, 2])


FG = nx.Graph()
FG.add_weighted_edges_from(
    [(1, 2, 0.125), (1, 3, 0.75), (2, 4, 1.2), (3, 4, 0.375)])
print(list(FG.edges))
print(FG.adj[1][2])
print("#########################################")
G.clear()
G.add_node(1, time='5pm')
G.add_nodes_from([3], time='2pm')

G.nodes[1]['room'] = 714
G.nodes.data()
print(G.nodes.data())
print(G[1])
print(G.nodes[1])
G.add_edge(1, 2, weight=4.7)
G.add_edges_from([(3, 4), (4, 5)], color='red')
G.add_edges_from([(1, 2, {'color': 'blue'}), (2, 3, {'weight': 8})])
G[1][2]['weight'] = 4.77
G.edges[3, 4]['weight'] = 4.2
print(G.nodes.data())
print(G.edges.data())

print("#########################################")
DG = nx.DiGraph()
DG.add_weighted_edges_from([(1, 2, 0.5), (3, 1, 0.75)])
DG.out_degree(1, weight='weight')
print(DG.nodes.data())
print(DG.edges.data())
print(list(DG.successors(1)))
print(list(DG.neighbors(1)))

print()
print()
'''
print("#########################################")
iris = load_iris()
X = iris.data[:10,:]

# setting distance_threshold=0 ensures we compute the full tree.
model = AgglomerativeClustering(n_clusters=2)
model = model.fit(X)

n_samples =len(X)
print(model.children_)
print(model.labels_)
print(len(X))
#print(model.children_)
#print(model.children_)
#print(model.children_)
G = nx.DiGraph()
counts = np.zeros(100)
for i, merge in enumerate(model.children_):
    current_count = 0
    print("#####################")
    print(i)
    print(merge)
    for child_idx in merge:
        print(child_idx)
        
        
        if child_idx < n_samples:
            current_count += 1  # leaf node
            str1 = "merge"+str(i+n_samples)+"->"+"leaf"+str(child_idx)
            print(str1)
            G.add_edge("m"+str(i+n_samples), "l"+str(child_idx))
        else:
            current_count += counts[child_idx - n_samples]
            str1 = "merge"+str(i+n_samples)+"->"+"merge"+str(child_idx)
            print(str1)
            G.add_edge("m"+str(i+n_samples), "m"+str(child_idx))
    
    counts[i] = current_count
   
    print(counts[i])

print(G.nodes.data())
print(G.edges.data())
pos = nx.circular_layout(G)
pos = nx.spectral_layout(G)
pos = nx.shell_layout(G)


pos = nx.kamada_kawai_layout(G)
nx.draw(G, pos, with_labels=True, node_size=600)
plt.show()
