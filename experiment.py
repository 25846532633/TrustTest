# experiment.py
from __future__ import annotations
import os, json, yaml
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from llm_runner import HFInstructRunner
from variable_embedding import build_variable_embeddings
from clustering import cluster_variables
from prompt_builder import build_flat_prompt_timeseries, build_hier_prompt_timeseries
import numpy as np
from aeon.datasets import load_basic_motions
#from llm_runner import SklearnRunner, OpenAIRunner

# experiment.py 里替换整个 load_basicmotions_numpy()

def load_basicmotions_numpy():
    """
    Load BasicMotions dataset and convert to:
      X: (N, T, D) float32 numpy array
      y: (N,) int labels
      feature_names: list[str] length=D
      label_names: list[str] length=C

    兼容 aeon 的不同返回格式：
      - 有的版本返回 numpy3d: (N, D, T) 或 (N, T, D)
      - 有的版本返回 list[DataFrame] / list[np.ndarray]
    """
    X_train, y_train = load_basic_motions(split="train")
    X_test, y_test = load_basic_motions(split="test")

    def to_numpy_3d(X_any):
        """
        把 aeon 返回的 X 统一成 numpy 3D:
          - 如果已经是 np.ndarray:直接用
          - 如果是 list/tuple:stack 成 (N, ?, ?)
        然后把常见的 (N, D, T) 转成 (N, T, D)
        """
        # 1) 先变成 ndarray
        if isinstance(X_any, np.ndarray):
            X = X_any
        else:
            # list / tuple 情况
            first = X_any[0]
            if hasattr(first, "to_numpy"):  # pandas DataFrame
                arrs = [a.to_numpy() for a in X_any]
            elif hasattr(first, "values"):  # 也可能是 DataFrame/Series
                arrs = [a.values for a in X_any]
            else:
                arrs = [np.asarray(a) for a in X_any]
            X = np.stack(arrs, axis=0)

        X = np.asarray(X, dtype=np.float32)

        if X.ndim != 3:
            raise ValueError(f"BasicMotions X should be 3D, but got shape={X.shape}")

        # 2) 统一到 (N, T, D)
        # 常见情况：BasicMotions 是 6维传感器、长度 100
        # - 若 X 是 (N, D, T) 且 D 比较小、T 比较大，就转置
        if X.shape[1] <= 32 and X.shape[2] > X.shape[1]:
            # 认为是 (N, D, T) -> (N, T, D)
            X = np.transpose(X, (0, 2, 1))

        return X

    X_train = to_numpy_3d(X_train)
    X_test = to_numpy_3d(X_test)

    # 标签编码：把字符串标签映射到 0..C-1
    label_names = sorted(set(y_train))
    label2id = {l: i for i, l in enumerate(label_names)}
    y_train = np.array([label2id[l] for l in y_train], dtype=np.int64)
    y_test = np.array([label2id[l] for l in y_test], dtype=np.int64)

    # 变量名：BasicMotions 固定 6 维（加速度3 + 陀螺仪3）
    D = X_train.shape[2]
    if D == 6:
        feature_names = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
    else:
        feature_names = [f"var_{i}" for i in range(D)]

    return X_train, X_test, y_train, y_test, feature_names, label_names




# 制作合成数据集
def make_synthetic_mts_dataset(
    n_samples: int,
    seq_len: int,
    n_vars: int,
    n_classes: int,
    noise_std: float,
    seed: int,
):
    """
    生成一个“多变量时序分类”合成数据集：
    - 每个类别使用不同的频率/相位模式
    - 不同变量对类别的敏感程度不同（这会让“变量聚类”有点意义）
    输出：
      X: (N, L, D)
      y: (N,)
      feature_names: (D,)
      label_names: (n_classes,)
    """
    rng = np.random.default_rng(seed) # 随机数生成器(可复现)
    # 从[0,2*pi]均匀采样seq_len个点，作为时间序列
    t = np.linspace(0, 2*np.pi, seq_len, dtype=np.float32) # 时间序列

    # X[i,:,d]:样本i的第d个变量的序列
    X = np.zeros((n_samples, seq_len, n_vars), dtype=np.float32) # 样本矩阵 (N, L, D)
    # y[i]:样本i的标签
    y = rng.integers(0, n_classes, size=(n_samples,), dtype=np.int64) # 标签矩阵 (N,)

    # 为每个变量设置“类别敏感系数”，让某些变量更区分某些类
    # 变量×类别权重矩阵:变量d对类别c的敏感程度
    var_class_weight = rng.uniform(0.5, 1.5, size=(n_vars, n_classes)).astype(np.float32) # 变量类别权重矩阵 (D, C)

    # 生成样本
    for i in range(n_samples):
        c = int(y[i]) # 样本i的标签
        base_freq = 1.0 + 0.6 * c           # 类别决定主频
        base_phase = 0.7 * c

        for d in range(n_vars):
            amp = var_class_weight[d, c] # 变量d对类别c的敏感程度
            # 每个变量再加一点自己的频率偏移
            freq = base_freq + 0.05 * d # 频率
            phase = base_phase + 0.1 * d # 相位
            signal = amp * np.sin(freq * t + phase) # 信号
            # 信号 = 振幅 * 正弦函数(频率 * 时间 + 相位)


            noise = rng.normal(0.0, noise_std, size=(seq_len,)).astype(np.float32) # 噪声
            X[i, :, d] = signal + noise # 样本i的第d个变量的序列

    feature_names = [f"var_{d}" for d in range(n_vars)] # 变量名列表
    label_names = [f"class_{k}" for k in range(n_classes)] # 标签名列表
    # 返回数据集

    # (N, L, D), (N,), (D,), (C,)
    return X, y, feature_names, label_names

def main():
    with open("config.yaml", "r", encoding="utf-8") as f: # 加载配置
        cfg = yaml.safe_load(f)

    seed = int(cfg["seed"]) # 随机种子 - 控制随机性
    k = int(cfg["cluster_k"]) # 聚类数 - 聚类的组数
    last_k = int(cfg["prompt"]["last_k"]) # 最近last_k个点 - 提示中展示的最近last_k个点

    # 1) 数据（合成 MTS）
    if cfg["data"]["dataset"] == "basicmotions":
        X_train, X_test, y_train, y_test, feature_names, label_names = load_basicmotions_numpy()
    else:
        dc = cfg["synthetic"]
        X, y, feature_names, label_names = make_synthetic_mts_dataset(
            n_samples=int(dc["n_samples"]),
            seq_len=int(dc["seq_len"]),
            n_vars=int(dc["n_vars"]),
            n_classes=int(dc["n_classes"]),
            noise_std=float(dc["noise_std"]),
            seed=seed,
        ) # 生成合成数据集

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=seed, stratify=y
        ) # 分割数据集

    # 2) 变量 embedding（训练集统计 -> 变量行为特征）
    emb_cfg = cfg["embedding"]
    var_emb = build_variable_embeddings(
        feature_names=feature_names,
        X_train=X_train,
        use_name_tfidf=bool(emb_cfg["use_name_tfidf"]),
        stats=list(emb_cfg["stats"]),
    ) # 构建变量embedding - 每个变量一个向量表示
    # 判断变量之间像不像

    # 3) 变量聚类（对 D 个变量聚类）
    clus = cluster_variables(var_emb.embeddings, k=k)
    # 变量聚类得到的是“输入结构”（clusters），不是模型参数。
    # 它是一个静态结构，一次算出来，整个测试集都用它。

    # 4) runner
    mode = cfg["runner"]["mode"] # 运行模式 - sklearn 或 openai
    if mode == "sklearn":
        runner = SklearnRunner(label_names=label_names, seed=seed) # 离线 runner：保证你框架完整闭环（聚类/提示/评估/日志）先跑通。
        runner.fit(X_train, y_train)
    elif mode == "openai": # 在线 runner：接 LLM 做预测
        runner = OpenAIRunner(label_names=label_names, model_name=cfg["runner"]["openai_model"])
    elif mode == "hf_instruct":
        runner = HFInstructRunner(
            label_names=label_names,
            model_name_or_path=cfg["runner"]["hf_model"],
            device=cfg["runner"].get("device", "auto"),
            max_new_tokens=int(cfg["runner"].get("max_new_tokens", 16)),
            temperature=float(cfg["runner"].get("temperature", 0.0)),
        )
    else: # 未知模式
        raise ValueError(f"Unknown runner mode: {mode}")

    # 5) 对照：flat vs hierarchical prompt
    flat_preds, hier_preds = [], [] # 平铺预测和层次化预测
    for i in range(X_test.shape[0]):
        xi = X_test[i]  # (L, D)

        # 构建提示
        p_flat = build_flat_prompt_timeseries(feature_names, xi, label_names, last_k=last_k) # 平铺提示
        p_hier = build_hier_prompt_timeseries(feature_names, xi, clus.clusters, label_names, last_k=last_k) # 层次化提示

        r1 = runner.run(p_flat)
        r2 = runner.run(p_hier)
        # 运行模型，得到预测结果
        # r1.pred: 平铺预测
        # r2.pred: 层次化预测

        flat_preds.append(r1.pred) # 平铺预测
        hier_preds.append(r2.pred) # 层次化预测

    y_true = [label_names[int(t)] for t in y_test] # 真实标签
    flat_acc = accuracy_score(y_true, flat_preds) # 平铺准确率
    hier_acc = accuracy_score(y_true, hier_preds) # 层次化准确率

    # 6) 记录结果（研究代码必须有 logs）
    os.makedirs("logs", exist_ok=True) # 创建日志目录
    out = {
        "seed": seed,
        "data": dc,
        "cluster_k": k,
        "clusters": {str(cid): [feature_names[i] for i in idxs] for cid, idxs in clus.clusters.items()},
        "flat_acc": float(flat_acc),
        "hier_acc": float(hier_acc),
        "mode": mode,
        "prompt_last_k": last_k,
    }
    with open("logs/result.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print("Clusters:", out["clusters"])
    print(f"Flat Acc: {flat_acc:.4f}")
    print(f"Hier Acc: {hier_acc:.4f}")
    print("Saved to logs/result.json")

if __name__ == "__main__":
    main()
