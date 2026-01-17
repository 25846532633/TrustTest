# prompt_builder.py
from __future__ import annotations
from typing import Dict, List
import numpy as np

# 轻量摘要:对单个变量的一条序列(shape=(L,)）做轻量摘要，用于提示。
def _summarize_series(x_var: np.ndarray) -> dict:
    """
    轻量摘要:对单个变量的一条序列(shape=(L,)）做轻量摘要，用于提示。
    输入:
        x_var: 变量序列 (L,)
    输出:
        dict: 轻量摘要 (mean/std/min/max)
    """
    return {
        "mean": float(x_var.mean()),
        "std": float(x_var.std()),
        "min": float(x_var.min()),
        "max": float(x_var.max()),
    }

# 平铺promt：对每个变量取recent=last_k个点 + 轻量摘要，用于提示 - 把信息丢给模型。
def build_flat_prompt_timeseries(
    feature_names: List[str],
    x: np.ndarray,              # shape = (L, D)
    label_names: List[str],
    last_k: int = 6,
) -> str:
    """
    输入:
        feature_names: 变量名列表 (D,)
        x: 变量序列矩阵 (L, D)
        label_names: 标签名列表 (C,)
        last_k: 最近last_k个点
    输出:
        str: 提示 (D,) 行字符串
    平铺输入：按变量逐个列出：
      - 变量名: last_k个值 + (mean/std/min/max)
    例如：
    - A: recent=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], mean=3.000, std=1.000, min=1.000, max=6.000
    - B: recent=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], mean=3.000, std=1.000, min=1.000, max=6.000
    - C: recent=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], mean=3.000, std=1.000, min=1.000, max=6.000
    """
    L, D = x.shape # (L, D): 样本数，变量数
    labels = ", ".join(label_names) # 标签名列表 (C,)

    lines = [] # 行字符串列表 (D,)
    for d in range(D):
        name = feature_names[d] # 变量名 (D,)
        series = x[:, d] # 变量序列 (L,)
        tail = series[-last_k:] if last_k > 0 else series # 最近last_k个点 (last_k,)
        summ = _summarize_series(series) # 轻量摘要 (mean/std/min/max)
        lines.append( # 行字符串 (D,)
            f"- {name}: recent={np.round(tail, 3).tolist()}, "
            f"mean={summ['mean']:.3f}, std={summ['std']:.3f}, min={summ['min']:.3f}, max={summ['max']:.3f}"
        )

    body = "\n".join(lines)

    # 返回提示
    return f"""You are a time-series classifier.
            Choose exactly one label from: {labels}

            Each variable shows: recent values and summary statistics.

            Variables:
            {body}

            Return only the label name.
            """

# 层次化输入：按变量簇分组输出 - 不再按变量序号列，而是按 clusters 分组列。
# clustering 给出组结构
# prompt_builder 把组结构变成 prompt 的层次结构
def build_hier_prompt_timeseries(
    feature_names: List[str],
    x: np.ndarray,              # (L, D)
    clusters: Dict[int, List[int]],
    label_names: List[str],
    last_k: int = 6,
) -> str:
    """
    层次化输入：按变量簇分组输出。
    """
    labels = ", ".join(label_names) # 标签名列表 (C,)

    group_blocks = [] # 组块列表 (C,)
    for cid, idxs in sorted(clusters.items(), key=lambda t: t[0]):
        lines = [] # 行字符串列表 (D,)
        for d in idxs:
            name = feature_names[d] # 变量名 (D,)
            series = x[:, d] # 变量序列 (L,)
            tail = series[-last_k:] if last_k > 0 else series # 最近last_k个点 (last_k,)
            summ = _summarize_series(series) # 轻量摘要 (mean/std/min/max)
            lines.append( # 行字符串 (D,)
                f"  - {name}: recent={np.round(tail, 3).tolist()}, "
                f"mean={summ['mean']:.3f}, std={summ['std']:.3f}, min={summ['min']:.3f}, max={summ['max']:.3f}"
            )
        """
        Grouped variables:
            Group 0:
            - temp: recent=..., mean=..., ...
            - humidity: recent=..., mean=..., ...

            Group 1:
            - cpu: recent=..., mean=..., ...
            - mem: recent=..., mean=..., ...
        """
        group_blocks.append(f"Group {cid} (clustered variables):\n" + "\n".join(lines))
        # 组块列表 (C,)

    grouped = "\n\n".join(group_blocks) # 组块列表 (C,)

    return f"""You are a time-series classifier.
            Choose exactly one label from: {labels}

            The variables are grouped by a clustering algorithm. Use the grouping structure to reason.

            Grouped variables:
            {grouped}

            Return only the label name.
            """
