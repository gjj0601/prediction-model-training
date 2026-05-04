# src/interpretability_shap.py
from __future__ import annotations
import numpy as np
import pandas as pd

def _get_coef_for_alpha(est, alpha):
    # Coxnet: coef_ shape = (n_features, n_alphas); alphas_ 是路径
    if alpha is None:
        return est.coef_[:, -1]  # 默认最后一个 alpha
    j = int(np.argmin(np.abs(est.alphas_ - float(alpha))))
    return est.coef_[:, j]

def shap_rank_linear_coxnet(bundle, X_background, X_explain, top_k=15):
    pre = bundle.pipeline.named_steps["preprocess"]
    est = bundle.pipeline.named_steps["coxnet"]

    Xb = pre.transform(X_background)
    Xe = pre.transform(X_explain)

    coef = _get_coef_for_alpha(est, bundle.alpha)

    # 背景均值（在“变换后空间”计算）
    mu = np.asarray(Xb).mean(axis=0)

    # 线性模型 SHAP（独立假设）：coef * (x - mean) :contentReference[oaicite:5]{index=5}
    shap_vals = (np.asarray(Xe) - mu) * coef

    mean_abs = np.abs(shap_vals).mean(axis=0)

    # 特征名：你在 train_coxnet 里已经保存了 feature_names_out（最好）
    feat_names = bundle.feature_names_out
    if feat_names is None:
        feat_names = np.array([f"x{i}" for i in range(len(mean_abs))])

    tbl = pd.DataFrame({"feature": feat_names, "mean_abs_shap": mean_abs})
    tbl = tbl.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    return tbl.head(int(top_k))

def plot_top_shap_bar(tbl, top_k, save_path):
    import matplotlib.pyplot as plt
    top = tbl.head(int(top_k)).iloc[::-1]  # 反转：最大在上
    plt.figure()
    plt.barh(top["feature"], top["mean_abs_shap"])
    plt.xlabel("mean(|SHAP|)")
    plt.title(f"Top {int(top_k)} features (base + pro)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
