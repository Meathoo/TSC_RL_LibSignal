import torch
import numpy as np
from sklearn.cluster import KMeans

def spectral_clustering(adjacency_matrix, n_clusters):
    """
    對給定的鄰接矩陣執行譜聚類 (Spectral Clustering)。
    這個過程是實現 Normalized Cut 的一種標準方法。

    Args:
        adjacency_matrix (torch.Tensor): 一個方形的鄰接矩陣 (N, N)，代表圖中節點間的相似度或權重。
        n_clusters (int): 要分成的群組數量 (K)。

    Returns:
        torch.Tensor: 每個節點的聚類標籤 (一個長度為 N 的張量)。
    """
    # 1. 計算度矩陣 (Degree Matrix) D
    # D_ii = sum_j(A_ij)
    degree_matrix = torch.diag(torch.sum(adjacency_matrix, dim=1))

    # 2. 計算正規化的圖拉普拉斯矩陣 (Normalized Graph Laplacian)
    # L_sym = D^{-1/2} L D^{-1/2} = I - D^{-1/2} A D^{-1/2}
    # 避免除以零
    d_inv_sqrt = torch.diag(1.0 / (torch.sqrt(torch.diag(degree_matrix)) + 1e-8))
    laplacian_normalized = torch.eye(adjacency_matrix.shape[0], device=adjacency_matrix.device) - d_inv_sqrt @ adjacency_matrix @ d_inv_sqrt

    # 3. 計算特徵向量與特徵值
    # torch.linalg.eigh 會返回按升序排列的特徵值和對應的特徵向量
    try:
        eigenvalues, eigenvectors = torch.linalg.eigh(laplacian_normalized)
    except torch.linalg.LinAlgError:
        # 如果矩陣在 CUDA 上計算失敗，轉到 CPU 用 numpy 計算
        laplacian_np = laplacian_normalized.cpu().numpy()
        eigenvalues, eigenvectors = np.linalg.eigh(laplacian_np)
        eigenvectors = torch.from_numpy(eigenvectors).to(adjacency_matrix.device)

    # 4. 選取前 k 個最小的特徵向量 (對應最小的 k 個特徵值)
    k_eigenvectors = eigenvectors[:, :n_clusters]

    # 5. 使用 K-means 演算法對新的節點表示 (由特徵向量組成) 進行聚類
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init='auto')
    clusters = kmeans.fit_predict(k_eigenvectors.cpu().numpy())

    return torch.from_numpy(clusters).to(adjacency_matrix.device)