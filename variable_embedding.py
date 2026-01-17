# variable_embedding.py
# 目的：把“变量（channel）”编码成向量 embedding，用于变量聚类
#
# 时序数据形状：X_train shape = (N, L, D)
# - N: 样本数
# - L: 时间长度
# - D: 变量数（我们要聚类的对象）
#
# 我们要为每个变量 d 构建一个向量 e(v_d)，描述该变量在训练集的“整体行为”：
#   1) 统计特征：mean/std/min/max
#   2) 趋势特征：slope（对“平均序列”做线性拟合）
#   3) 自相关：acf1（lag=1 的自相关）
#   4) 频域：domfreq（平均序列 FFT 的主频位置）
#
# 最终：embeddings shape = (D, dim)

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import List

from sklearn.feature_extraction.text import TfidfVectorizer


@dataclass
# 变量embedding类
class VariableEmbeddings:
    """
    变量embedding类
    输入:
        names: 变量名列表
        embeddings: 变量embedding矩阵
    输出:
        VariableEmbeddings: 变量embedding类
            names: 变量名列表
            embeddings: 变量embedding矩阵
                shape为(D, dim_total)
                D为变量数
                dim_total为变量名TF-IDF和时序全局特征的维度之和
    """
    names: List[str]
    # 变量embedding矩阵
    embeddings: np.ndarray  # (D, dim)
    # 变量数，维度

# 计算斜率
def _linear_slope(y: np.ndarray) -> float:
    """
    对一条序列 y(t) 做线性拟合 y ≈ a*t + b，返回斜率 a。
    这里用最小二乘闭式解。
    """
    t = np.arange(len(y), dtype=np.float32)
    t_mean = t.mean()
    y_mean = y.mean()
    num = np.sum((t - t_mean) * (y - y_mean))
    den = np.sum((t - t_mean) ** 2) + 1e-8
    return float(num / den)

# 计算自相关系数
def _acf1(y: np.ndarray) -> float:
    """
    计算序列 y 的 lag=1 自相关（非常轻量的时序形态特征）。
    """
    if len(y) < 2:
        return 0.0
    y0 = y[:-1]
    y1 = y[1:]
    y0 = y0 - y0.mean()
    y1 = y1 - y1.mean()
    num = float(np.sum(y0 * y1))
    den = float(np.sqrt(np.sum(y0**2) * np.sum(y1**2)) + 1e-8)
    return num / den

# 计算主频位置
def _dominant_freq_index(y: np.ndarray) -> float:
    """
    用 FFT 找主频位置（只返回“频率索引”，不做物理单位换算，足够用于聚类）。
    我们忽略 DC 分量（0频），在 1..L/2 找最大幅值的频率索引。
    """
    L = len(y)
    if L < 4:
        return 0.0
    # rfft 只取非负频率
    spec = np.fft.rfft(y)
    mag = np.abs(spec)
    # 排除 DC
    mag[0] = 0.0
    idx = int(np.argmax(mag))
    return float(idx)

# 计算变量特征
def _compute_variable_features_timeseries(X_train: np.ndarray, stats: List[str]) -> np.ndarray:
    """
    输入:X_train shape = (N, L, D)
    输出:var_feat shape = (D, n_stats)
    每一行对应一个变量的“全局行为特征”。
    """
    N, L, D = X_train.shape

    # mean_series[d, t]：变量 d 在时间 t 的跨样本平均
    # 这是一个“变量的代表性轨迹”，用于 slope/acf/freq 等形态特征
    mean_series = X_train.mean(axis=0).T  # (D, L)

    feats = []
    for s in stats:
        if s == "mean":
            # 变量 d 在所有样本与所有时刻上的均值
            feats.append(X_train.mean(axis=(0, 1)))  # (D,)
        elif s == "std":
            # 变量 d 在所有样本与所有时刻上的标准差
            feats.append(X_train.std(axis=(0, 1)))   # (D,)
        elif s == "min":
            # 变量 d 在所有样本与所有时刻上的最小值
            feats.append(X_train.min(axis=(0, 1)))   # (D,)
        elif s == "max":
            # 变量 d 在所有样本与所有时刻上的最大值
            feats.append(X_train.max(axis=(0, 1)))   # (D,)
        elif s == "slope":
            # 变量 d 在所有样本与所有时刻上的斜率
            vals = np.array([_linear_slope(mean_series[d]) for d in range(D)], dtype=np.float32)
            feats.append(vals)
        elif s == "acf1":
            # 变量 d 在所有样本与所有时刻上的自相关系数
            # 自相关系数：衡量序列与其自身滞后版本的相似度
            vals = np.array([_acf1(mean_series[d]) for d in range(D)], dtype=np.float32)
            feats.append(vals)
        elif s == "domfreq":
            # 变量 d 在所有样本与所有时刻上的主频位置
            vals = np.array([_dominant_freq_index(mean_series[d]) for d in range(D)], dtype=np.float32)
            feats.append(vals)
        else:
            raise ValueError(f"Unknown stats feature: {s}")

    # (n_stats, D) -> (D, n_stats)
    return np.stack(feats, axis=1).astype(np.float32)

# 构建变量embedding
def build_variable_embeddings(
    feature_names: List[str],
    X_train: np.ndarray,
    use_name_tfidf: bool = True,
    stats: List[str] | None = None,
) -> VariableEmbeddings:
    """
    构建变量 embedding:变量名 TF-IDF(可选)+ 时序全局特征(可选)
    输入:
        feature_names: 变量名列表
        X_train: 训练集特征矩阵
        use_name_tfidf: 是否使用变量名TF-IDF
        stats: 统计特征列表
    输出:
        VariableEmbeddings: 变量embedding
        变量embedding的shape为(D, dim_total)
            D为变量数
            dim_total为变量名TF-IDF和时序全局特征的维度之和
    """
    parts: List[np.ndarray] = [] # 用于存储变量名TF-IDF和时序全局特征的向量
    D = X_train.shape[2] # 变量数
    #(B,N,L):样本数，时间长度，变量数
    #(D, n_stats):变量数，统计特征数
    #(D, dim_name):变量数，变量名TF-IDF维度
    #(D, dim_total):变量数，变量名TF-IDF和时序全局特征的维度之和
    #(D, dim_total):变量数，变量名TF-IDF和时序全局特征的维度之和

    # ---------- 1) 变量名 embedding（语义线索） ----------
    # 根据变量叫什么名字，获得它和哪些变量更相似的信息
    if use_name_tfidf:
        vec = TfidfVectorizer(analyzer="char", ngram_range=(2, 4)) # 使用字符n-gram 
        # 例如："A" 和 "B" 两个变量，如果它们都包含 "AB" 这个子串，那么它们在TF-IDF空间中会更接近
        #vec是TF-IDF向量器，fit_transform(feature_names) 将变量名转换为TF-IDF向量
        name_emb = vec.fit_transform(feature_names).toarray().astype(np.float32) # (D, dim_name)
        # vec.fit_transform(feature_names) 将变量名转换为TF-IDF向量
        # name_emb是(D, dim_name)的矩阵，每一行对应一个变量的TF-IDF向量
        # 将name_emb转换为float32类型
        parts.append(name_emb) # (D, dim_name)

    # ---------- 2) 时序行为 embedding（形态线索） ----------
    if stats:
        var_feat = _compute_variable_features_timeseries(X_train, stats)  # (D, n_stats)
        # 标准化（按列），避免某一维统治距离
        var_feat = (var_feat - var_feat.mean(axis=0, keepdims=True)) / (var_feat.std(axis=0, keepdims=True) + 1e-8)
        # var_feat是(D, n_stats)的矩阵，每一行对应一个变量的时序全局特征
        parts.append(var_feat)

    if not parts: # 如果两部分都没开，就没有 embedding 了
        raise ValueError("At least one of use_name_tfidf or stats must be enabled.") # 至少一个必须开启

    # 拼接变量名TF-IDF和时序全局特征
    emb = np.concatenate(parts, axis=1)  # (D, dim_total)
    if emb.shape[0] != D:
        raise RuntimeError("Embedding rows must match number of variables (D).") # 行数必须等于变量数
    
    # 返回变量embedding矩阵
    return VariableEmbeddings(names=feature_names, embeddings=emb) # (D, dim_total)
