# Answer

## Introduction
 We will compare two different approaches: a spectral embedding (`Classical_Embedding.py`) and a ML based embedding (`ML_Embedding.py`). The embeddings are compared using the metric in `analysis.py`, focusing on community structure, shortest-path, and local neighborhoods.

## Methods

### Spectral embedding
We make the graph Laplacian $L = D - A$ and compute its eigenvectors. The 3D embedding is formed using the second through fourth smallest eigenvectors. This method finds coordinates that vary smoothly across the graph, so connected nodes end up close together. This can reveal large scale structure and some community separation.

### ML embedding
The learned embedding is trained using a combination of loss terms designed to preserve different graph properties:

- **Reconstruction loss $L_{\text{recon}}$:** Makes nodes that are connected in the graph end up close together in the embedding, using a dot-product objective with negative sampling.  
- **Path loss $L_{\text{path}}$:** Tries to make distances in the embedding reflect the graph’s shortest-path distances, with a monotonic mapping of graph distance to embedded distance.  
- **Optional community loss $L_{\text{comm}}$:** Pulls nodes from the same community closer to each other.  
- **Total loss:**  
  $$
  L = \alpha L_{\text{recon}} + \beta L_{\text{path}} + \gamma L_{\text{comm}}
  $$
  where $\alpha, \beta, \gamma$ set the weight of each term.

### Training details
- Optimizer: Adam with learning rate around $10^{-3}$
- Edge batching with 5:1 negative sampling
- Initialization: small Gaussian noise or a spectral warm start.
- Regularization: L2 penalty and early stopping based on validation metrics

## Evaluation metrics
- **Spearman correlation:** Measures how well Euclidean distances in the embedding align with graph shortest-path distances (global structure).
- **NMI:** Compares KMeans clusters in the embedding with Louvain communities (community recovery).
- **Silhouette score:** Measures how tight and well-separated the KMeans clusters are.
- **Neighborhood overlap:** Fraction of true graph neighbors recovered among the top-k nearest neighbors in the embedding (local structure).

## Results (from `analysis.py`)

| Metric | ML Embedding | Spectral Embedding |
|--------|--------------:|------------------:|
| Shape  | (13, 3) | (13, 3) |
| Spearman (SP Corr) | 0.9689 | 0.8794 |
| NMI (vs Louvain) | 0.2752 | 0.2752 |
| Silhouette | 0.5936 | 0.4184 |
| Local neighbor overlap (top-k) | 0.6538 | 0.6667 |

**Summary:**  
- **ML:** SP Corr = 0.969, NMI = 0.275, Silhouette = 0.594, Local = 0.654  
- **Spectral:** SP Corr = 0.879, NMI = 0.275, Silhouette = 0.418, Local = 0.667  

## Discussion
The ML embedding does a much better job at preserving global shortest-path geometry, with a Spearman correlation close to 0.97. This suggests that the training objective places strong emphasis on distance preservation, either directly through the path loss or indirectly through the reconstruction objective. The spectral embedding also preserves global distances reasonably well, but not to the same extent.

Both methods struggle to recover Louvain communities, performing similarly and showing a low NMI score (~0.275). This indicates that the clusters KMeans finds in the embedding space don’t match the graph’s modularity-based communities very well. That said, the ML embedding forms tighter clusters overall, as seen in its higher silhouette score.

Looking at local neighborhood structure, the spectral embedding slightly outperforms the ML embedding in recovering neighbors, although the difference is small. In general, the ML embedding is better at capturing global distances and producing compact clusters, whereas the spectral embedding focuses more on smooth transitions along edges and may reflect a different kind of structure than modularity-based communities.