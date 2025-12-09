import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import random

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

# Make graph
G = nx.Graph()
G.add_edges_from(edges)
nodes = sorted(G.nodes())
n = len(nodes)
idx = {node:i for i,node in enumerate(nodes)}
edge_index = [(idx[u], idx[v]) for u,v in edges]

all_pairs = [(i,j) for i in range(n) for j in range(i+1,n)]
non_edges = [p for p in all_pairs if (nodes[p[0]], nodes[p[1]]) not in edges]

# Setup model
dim = 3
Z = nn.Parameter(torch.randn(n, dim))
opt = optim.Adam([Z], lr=0.01)

# Loss function
def loss_fn(Z, edges, non_edges, neg_ratio=5):
    att = sum(torch.sum((Z[i]-Z[j])**2) for i,j in edges)
    rep = sum(torch.exp(-torch.sum((Z[i]-Z[j])**2)) for _ in range(neg_ratio*len(edges))
              for i,j in [random.choice(non_edges)])
    return att + rep

epochs = 1000
embeddings_over_time = []

for epoch in range(epochs):
    opt.zero_grad()
    loss = loss_fn(Z, edge_index, non_edges)
    loss.backward()
    opt.step()
    
    # Store every epoch now
    embeddings_over_time.append(Z.detach().numpy().copy())
    
    if epoch % 100 == 0:  # still print progress
        print(f"Epoch {epoch}, Loss {loss.item():.4f}")

final_embedding = Z.detach().numpy()
np.save('saved_data/ml_embedding.npy', final_embedding)
np.save('saved_data/node_names_ml.npy', np.array(nodes))
print("\nML embedding saved...")

# Use consistent colors for nodes
k = 3
from sklearn.cluster import KMeans
final_kmeans = KMeans(n_clusters=k, n_init=20).fit(embeddings_over_time[-1])
node_colors = plt.cm.tab10(final_kmeans.labels_)

# Animation
fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(projection='3d')

# Fix axis limits
all_data = np.vstack(embeddings_over_time)
xlim = (np.min(all_data[:,0]), np.max(all_data[:,0]))
ylim = (np.min(all_data[:,1]), np.max(all_data[:,1]))
zlim = (np.min(all_data[:,2]), np.max(all_data[:,2]))

def update(frame):
    ax.clear()
    embedding = embeddings_over_time[frame]
    
    # Scatter nodes
    ax.scatter(embedding[:,0], embedding[:,1], embedding[:,2],
               s=100, color=node_colors)
    
    # Node labels
    for i,node in enumerate(nodes):
        ax.text(embedding[i,0], embedding[i,1], embedding[i,2], node)
    
    # Edges
    for u,v in edges:
        i = idx[u]; j = idx[v]
        ax.plot([embedding[i,0], embedding[j,0]],
                [embedding[i,1], embedding[j,1]],
                [embedding[i,2], embedding[j,2]], color='gray')
    
    # Fix axis limits
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(zlim)
    ax.set_title(f"Epoch {frame}")

print("Creating animation, might take a minute.")
ani = FuncAnimation(fig, update, frames=len(embeddings_over_time), interval=20)
ani.save("ML_Model/embedding_animation.mp4", dpi=200)
plt.savefig("ML_Model/Clustered Node Embedding.png")
# plt.show()