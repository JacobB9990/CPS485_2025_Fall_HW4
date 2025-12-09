import numpy as np
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, silhouette_score
from scipy.stats import spearmanr


ml_embedding = np.load("saved_data/ml_embedding.npy")
ml_node_names = np.load("saved_data/node_names_ml.npy")

spec_embedding = np.load("saved_data/spectral_embedding.npy")
spec_node_names = np.load("saved_data/node_names_spec.npy")

ml_node_names = [
    n.decode() if isinstance(n, (bytes, bytearray)) else n for n in ml_node_names
]
spec_node_names = [
    n.decode() if isinstance(n, (bytes, bytearray)) else n for n in spec_node_names
]

edges = [
    ("A", "B"),
    ("A", "C"),
    ("A", "D"),
    ("B", "F"),
    ("B", "E"),
    ("F", "E"),
    ("E", "K"),
    ("C", "H"),
    ("C", "G"),
    ("H", "G"),
    ("G", "L"),
    ("D", "I"),
    ("D", "J"),
    ("J", "I"),
    ("I", "M"),
]

G = nx.Graph()
G.add_edges_from(edges)

# Sanity checks
def sanity_check(name, emb):
    print(f"\n{name} SANITY CHECK")
    print("Shape:", emb.shape)
    print("Global Mean:", np.mean(emb))
    print("Global Std:", np.std(emb))
    print("Min / Max:", np.min(emb), np.max(emb))


sanity_check("ML", ml_embedding)
sanity_check("Spectral", spec_embedding)

# Shortest Paths
def shortest_path_correlation(G, embedding, node_order):
    nodes = list(node_order)

    dist_graph = []
    dist_embed = []

    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            try:
                d_g = nx.shortest_path_length(G, nodes[i], nodes[j])
                d_e = np.linalg.norm(embedding[i] - embedding[j])
                dist_graph.append(d_g)
                dist_embed.append(d_e)
            except nx.NetworkXNoPath:
                continue

    corr, _ = spearmanr(dist_graph, dist_embed)
    return corr


ml_sp_corr = shortest_path_correlation(G, ml_embedding, ml_node_names)
spec_sp_corr = shortest_path_correlation(G, spec_embedding, spec_node_names)

print("\nShortest Path Preservation")
print("ML Spearman Correlation:      ", ml_sp_corr)
print("Spectral Spearman Correlation:", spec_sp_corr)

communities = nx.community.louvain_communities(G)
true_labels = np.zeros(len(G))

node_to_idx = {n: i for i, n in enumerate(G.nodes())}

for cid, comm in enumerate(communities):
    for node in comm:
        true_labels[node_to_idx[node]] = cid


def community_score(embedding, true_labels):
    k = len(np.unique(true_labels))
    preds = KMeans(n_clusters=k, n_init=20).fit_predict(embedding)
    nmi = normalized_mutual_info_score(true_labels, preds)
    sil = silhouette_score(embedding, preds)
    return nmi, sil


ml_nmi, ml_sil = community_score(ml_embedding, true_labels)
spec_nmi, spec_sil = community_score(spec_embedding, true_labels)

print("\nCommunity Preservation")
print("ML  -> NMI:", ml_nmi, " Silhouette:", ml_sil)
print("Spec-> NMI:", spec_nmi, " Silhouette:", spec_sil)

# Neighborhood tests
def neighbor_overlap(G, embedding, k=5):
    adj = {n: set(G.neighbors(n)) for n in G.nodes()}
    sim = cosine_similarity(embedding)

    overlaps = []
    for i, node in enumerate(G.nodes()):
        embed_neighbors = np.argsort(sim[i])[-(k + 1) : -1]
        true_neighbors = set(node_to_idx[n] for n in adj[node])

        overlap = len(set(embed_neighbors) & true_neighbors) / max(
            1, len(true_neighbors)
        )
        overlaps.append(overlap)

    return np.mean(overlaps)


ml_overlap = neighbor_overlap(G, ml_embedding)
spec_overlap = neighbor_overlap(G, spec_embedding)

print("\nLocal Neighborhood Preservation")
print("ML Overlap:", ml_overlap)
print("Spectral Overlap:", spec_overlap)

#  Final report
print("\n")
print(f"ML     -> SP Corr: {ml_sp_corr:.3f}, NMI: {ml_nmi:.3f}, Sil: {ml_sil:.3f}, Local: {ml_overlap:.3f}")
print(f"Spect  -> SP Corr: {spec_sp_corr:.3f}, NMI: {spec_nmi:.3f}, Sil: {spec_sil:.3f}, Local: {spec_overlap:.3f}")
