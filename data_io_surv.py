from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple, Any

import pandas as pd
from sksurv.util import Surv  # structured y for survival analysis :contentReference[oaicite:4]{index=4}


def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def load_dataframe(path: str, columns: List[str] | None = None) -> pd.DataFrame:
    """
    读取 parquet/csv/excel。parquet 支持 columns= 只读需要的列。:contentReference[oaicite:5]{index=5}
    """
    p = Path(path)
    suf = p.suffix.lower()

    if suf == ".parquet":
        return pd.read_parquet(p, columns=columns)  # :contentReference[oaicite:6]{index=6}
    if suf == ".csv":
        # columns 参数对 read_csv 是 usecols
        return pd.read_csv(p, usecols=columns) if columns else pd.read_csv(p)
    if suf in [".xlsx", ".xls"]:
        return pd.read_excel(p, usecols=columns) if columns else pd.read_excel(p)

    raise ValueError(f"Unsupported file type: {suf}")


def _resolve_feature_sets(cfg: Dict[str, Any]) -> Dict[str, List[str]]:
    """
    把 YAML 里的 feature_sets 解析为：{model_name: [col1, col2, ...]}。
    规则：
      - base: 直接列表
      - 其他: base_ref + add
    """
    fs = cfg["data"]["feature_sets"]
    out: Dict[str, List[str]] = {}

    # base
    out["base"] = list(fs["base"])

    # base_plus_protein
    if "base_plus_protein" in fs:
        base_ref = fs["base_plus_protein"]["base_ref"]
        out["base_plus_protein"] = list(out[base_ref]) + list(fs["base_plus_protein"].get("add", []))

    # base_plus_aip
    if "base_plus_aip" in fs:
        base_ref = fs["base_plus_aip"]["base_ref"]
        out["base_plus_aip"] = list(out[base_ref]) + list(fs["base_plus_aip"].get("add", []))

    # 去重保持顺序
    for k, cols in out.items():
        seen = set()
        dedup = []
        for c in cols:
            if c not in seen:
                dedup.append(c)
                seen.add(c)
        out[k] = dedup

    return out


def _apply_ordered_categories(df: pd.DataFrame, ordinal_map: Dict[str, List[str]]) -> pd.DataFrame:
    """
    把 YAML 里声明的 ordinal 列设为有序分类（如 AIP_cat0: low<mid<high）。
    pandas.CategoricalDtype 支持 categories + ordered=True。:contentReference[oaicite:7]{index=7}
    """
    for col, cats in ordinal_map.items():
        if col not in df.columns:
            continue
        dtype = pd.CategoricalDtype(categories=cats, ordered=True)  # :contentReference[oaicite:8]{index=8}
        df[col] = df[col].astype(dtype)
    return df


def _validate_required_columns(df: pd.DataFrame, required: List[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")


def load_survival_datasets(cfg: Dict[str, Any]) -> Tuple[
    Dict[str, pd.DataFrame],   # X_by_model
    Any,                       # y structured array
    Dict[str, Dict[str, List[str]]],  # meta per model: ordinal/nominal/numeric/features
    pd.DataFrame               # df_core (id/time/event) 便于调试或输出
]:
    """
    一次性读入数据，并返回：
      - X_by_model: {'base': X1, 'base_plus_protein': X2, 'base_plus_aip': X3}
      - y: Surv structured array (event,time)
      - meta: 每个模型的列类型（供 ColumnTransformer 用）
      - df_core: 仅含 id/time/event 的小表（可选）
    """
    data_cfg = cfg["data"]
    path = data_cfg["path"]
    id_col = data_cfg["id_col"]
    time_col = data_cfg["time_col"]
    event_col = data_cfg["event_col"]

    feature_sets = _resolve_feature_sets(cfg)

    # 需要读取的最小列集合 = id/time/event + 三套 features + preprocessing 里提到的列
    needed_cols = {id_col, time_col, event_col}
    for cols in feature_sets.values():
        needed_cols.update(cols)

    prep = data_cfg.get("preprocessing", {})
    ordinal_map = prep.get("ordinal", {})
    nominal_cols_all = prep.get("nominal", [])

    needed_cols.update(ordinal_map.keys())
    needed_cols.update(nominal_cols_all)

    df = load_dataframe(path, columns=sorted(needed_cols))

    # 基本字段检查
    _validate_required_columns(df, [id_col, time_col, event_col])

    # 应用有序分类（AIP_cat0）
    df = _apply_ordered_categories(df, ordinal_map)

    # 生存标签 y（scikit-survival 推荐格式）:contentReference[oaicite:9]{index=9}
    # 注意：event_col 最好是 bool 或 0/1；time_col 必须是数值
    # 确保类型正确：event 支持 bool 或 0/1；time 要是 float :contentReference[oaicite:1]{index=1}
    df[event_col] = df[event_col].astype(bool)
    df[time_col] = df[time_col].astype(float)

    # ✅ 正确调用：event/time 是列名（字符串），用位置参数最不容易踩版本差异
    y = Surv.from_dataframe(event_col, time_col, df)

    # 为每个模型生成 X + meta（列类型）
    meta: Dict[str, Dict[str, List[str]]] = {}
    X_by_model: Dict[str, pd.DataFrame] = {}

    for model_name, feats in feature_sets.items():
        # 确保 feats 都存在
        _validate_required_columns(df, feats)

        X = df[feats].copy()

        # 列类型（严格按 YAML 来，不自动猜）
        ordinal_cols = [c for c in feats if c in ordinal_map]
        nominal_cols = [c for c in feats if c in nominal_cols_all]
        numeric_cols = [c for c in feats if c not in set(ordinal_cols) and c not in set(nominal_cols)]

        X_by_model[model_name] = X
        meta[model_name] = {
            "features": feats,
            "ordinal": ordinal_cols,
            "nominal": nominal_cols,
            "numeric": numeric_cols,
        }

    df_core = df[[id_col, time_col, event_col]].copy()
    return X_by_model, y, meta, df_core
