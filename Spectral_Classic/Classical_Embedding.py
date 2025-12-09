import numpy as np
import scipy as sp
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.cluster import KMeans

# Graph
edges = [
    ("A","B"),("A","C"),("A","D"),
    ("B","F"),("B","E"),("F","E"),
    ("E","K"),
    ("C","H"),("C","G"),("H","G"),
    ("G","L"),
    ("D","I"),("D","J"),("J","I"),
    ("I","M")
]

# Build graph to see
G = nx.Graph()
G.add_edges_from(edges)

# Fix node order
nodes = sorted(G.nodes())
idx = {node:i for i,node in enumerate(nodes)}
A = nx.to_numpy_array(G, nodelist=nodes)

# Laplacian 
degrees = np.array(A.sum(axis=1)).flatten()
D = sp.sparse.diags(degrees)
A_sparse = csr_matrix(A)          # convert matrix to sparse
L = D - A_sparse                  # Laplacian

# Compute bottom eigenvectors 
vals, vecs = eigsh(L, k=4, which='SM')
embedding = vecs[:, 1:4]   

np.save('saved_data/spectral_embedding.npy', embedding)
np.save('saved_data/node_names_spec.npy', np.array(nodes))
print("\nSpectral embedding saved...")

# Clustering
k = 3  # 3 clusters
kmeans = KMeans(n_clusters=k, n_init=20).fit(embedding)
labels = kmeans.labels_

colors = plt.cm.tab10(labels)

# Plotting
fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(projection='3d')

ax.scatter(embedding[:,0], embedding[:,1], embedding[:,2],
           s=100, color=colors)

# labels
for i,node in enumerate(nodes):
    ax.text(embedding[i,0], embedding[i,1], embedding[i,2], node)

# edges
for (u,v) in edges:
    i = idx[u]; j = idx[v]
    ax.plot([embedding[i,0], embedding[j,0]],
            [embedding[i,1], embedding[j,1]],
            [embedding[i,2], embedding[j,2]],
            color='gray')
plt.title("Spectral Node Embedding (ML)")
plt.tight_layout()
fig.savefig("Spectral_Classic/Spectral Node Embedding.png", dpi=300)
plt.show()