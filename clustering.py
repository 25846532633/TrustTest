# clustering.py
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Dict, List
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import normalize

@dataclass
class ClusterResult: # 聚类结果类
    """
    聚类结果类
    输入:
        labels: 变量标签
        clusters: 聚类结果
    """
    # labels[d] = c - 第d个变量属于第c个簇
    labels: np.ndarray               # (D,) 

    # clusters[c] = [d1, d2, ..., dn] - 第c个簇包含的变量索引列表
    clusters: Dict[int, List[int]]   # cluster_id -> variable indices

# 聚类变量
def cluster_variables(embeddings: np.ndarray, k: int) -> ClusterResult:
    """
    聚类变量
    输入:
        embeddings: 变量embedding矩阵
        k: 聚类数
    输出:
        ClusterResult: 聚类结果
    """
    if k < 1: 
        raise ValueError("k must be >= 1") # 聚类数必须大于等于1
    if k > embeddings.shape[0]:
        k = embeddings.shape[0] # 聚类数不能大于变量数,不然没意义

    X = normalize(embeddings) # 标准化变量embedding矩阵
    model = AgglomerativeClustering(n_clusters=k) # 聚类模型
    # 聚类模型：AgglomerativeClustering(n_clusters=k)
    # 聚类标签：model.fit_predict(X)
    # AgglomerativeClustering：层次聚类
    # 层次聚类：将样本逐步合并成越来越大的簇，直到达到预定的簇数    
    labels = model.fit_predict(X) # 聚类标签

    clusters: Dict[int, List[int]] = {} # 聚类结果
    for idx, c in enumerate(labels):
        clusters.setdefault(int(c), []).append(idx) # 聚类结果

    # 返回聚类结果
    return ClusterResult(labels=labels, clusters=clusters) # 返回聚类结果
