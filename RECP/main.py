# main.py — ReCP 下游任务评估（多 seed 格式化输出版）
"""
用法:
  CUDA_VISIBLE_DEVICES=2 python main.py
  CUDA_VISIBLE_DEVICES=2 python main.py --task crime --city chicago
  CUDA_VISIBLE_DEVICES=2 python main.py --fusion concat
  CUDA_VISIBLE_DEVICES=4 python main.py --task houseprice --city nyc
  CUDA_VISIBLE_DEVICES=4 python main.py --task pm25 --city chicago
"""
import sys
import os
import argparse
import json
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from collections import Counter
from copy import deepcopy
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, f1_score,
)
from pathlib import Path
from scipy.spatial.distance import cdist

# ⭐ 统一的基础路径
BASE_DIR = "/mnt/network_data/personal_workspace/liyanhuang/hly"

sys.path.insert(0, f"{BASE_DIR}/DS/model")
from compute_auc import compute_auc_score
from aggregate_embedding import process_city_pipeline
from train import FlatDataset, SimpleMLP, train_model, _normalize_geoid
from fusion import get_fusion, list_fusion_methods


# ═══════════════════════════════════════════════════════════════
# 0. 全局配置
# ═══════════════════════════════════════════════════════════════
SEEDS   = [42, 62, 82]
RUN_TAG = "recp"

CONFIG = {
    # 修正路径：指向最新的特征输出目录
    "emb_dir":    f"{BASE_DIR}/DS/model/RECP/output",
    "output_dir": f"{BASE_DIR}/DS/model/RECP/tract_embedding/{RUN_TAG}",
    "fusion_method": "concat",

    "tract_shapefiles": {
        "chicago": f"{BASE_DIR}/data_1.16/DS_Census/tl_2021_17_tract",
        "nyc":     f"{BASE_DIR}/data_1.16/DS_Census/tl_2021_36_tract",
        "sf":      f"{BASE_DIR}/data_1.16/DS_Census/tl_2021_06_tract",
    },

    "data_dir": f"{BASE_DIR}/DS/DS_Cleaned_Data",

    # ⭐ 只保留 5 个任务
    "tasks": {
        "crime": {
            "type":              "regression",
            "subfolder":         "Crime",
            "filename_template": "{city}_crime_unified.csv",
            "id_col":            "GEOID",
            "label_col":         "crime_count",
            "log_transform":     True,
            "cities":            ["chicago", "nyc"],
            "display_scale":     1.0,
            "display_unit":      "次",
        },
        "houseprice": {
            "type":              "regression",
            "subfolder":         "HousePrice",
            "filename_template": "{city}_tract_price_labels.csv",
            "id_col":            "tract_geoid",
            "label_col":         "price_median",
            "log_transform":     True,
            "cities":            ["chicago", "nyc"],
            "display_scale":     1000.0,
            "display_unit":      "$k",
        },
        "pm25": {
            "type":              "regression",
            "subfolder":         "PM25",
            "filename_template": "{city}_pm25_tract_2020.csv",
            "id_col":            "tract_geoid",
            "label_col":         "pm25",
            "log_transform":     False,
            "cities":            ["chicago", "nyc"],
            "display_scale":     1.0,
            "display_unit":      "μg/m³",
        },
        "poverty_cls": {
            "type":              "classification_from_continuous",
            "subfolder":         "Census",
            "filename_template": "{city}_census_unified.csv",
            "id_col":            "GEOID",
            "source_col":        "acs_poverty_rate",
            "fallback_cols":     ["poverty_rate"],
            "n_classes":         4,
            "cities":            ["chicago", "nyc"],
        },
        "education_cls": {
            "type":              "classification_from_continuous",
            "subfolder":         "Census",
            "filename_template": "{city}_census_unified.csv",
            "id_col":            "GEOID",
            "source_col":        "acs_higher_edu_rate",
            "fallback_cols":     [],
            "n_classes":         4,
            "cities":            ["chicago", "nyc"],
        },
    },

    # ⭐ 下游 MLP 超参
    "lr":            5e-4,   
    "epochs":        150,
    "patience":      30,
    "batch_size":    32,
    "hidden_dim":    256,
    "dropout":       0.3,    
    "test_size":     0.15,
    "val_size":      0.15,
    "min_tract_per_class": 10,

    "save_dir":           f"{BASE_DIR}/DS/model/RECP/results_{RUN_TAG}",
    "force_rebuild_tract": False,

    # 空间划分
    "spatial_split": False,
    "buffer_km":     1.0,
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ⭐ tract centroid 缓存
_CENTROID_CACHE = {}


# ═══════════════════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════════════════
def _detect_col(df, candidates, col_type="ID"):
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"未找到 {col_type} 列！可用: {df.columns.tolist()}")


def get_data_path(task_name, city):
    task_cfg = CONFIG["tasks"][task_name]
    if "data_path_override" in task_cfg:
        return task_cfg["data_path_override"]
    fname = task_cfg["filename_template"].format(city=city)
    return os.path.join(CONFIG["data_dir"], task_cfg["subfolder"], fname)


def get_embed_path(city):
    # 直接用传进来的 tag 拼接文件名，不再去检查 fusion.py
    tag = CONFIG["fusion_method"]
    return os.path.join(
        CONFIG["output_dir"],
        f"{city}_tract_embedding_{tag}.pickle",
    )


# ═══════════════════════════════════════════════════════════════
# ⭐ 空间 Buffer 切分
# ═══════════════════════════════════════════════════════════════
def _resolve_shp(path_like):
    p = Path(path_like)
    if p.is_dir():
        shps = sorted(p.glob("*.shp"))
        if not shps:
            raise FileNotFoundError(f"目录里没 .shp: {p}")
        return str(shps[0])
    return str(p)


def _load_tract_centroids(city):
    if city in _CENTROID_CACHE:
        return _CENTROID_CACHE[city]

    shp_path = CONFIG["tract_shapefiles"].get(city)
    if shp_path is None or not Path(shp_path).exists():
        print(f"   ⚠️ 找不到 shapefile: {shp_path}")
        _CENTROID_CACHE[city] = {}
        return {}

    try:
        import geopandas as gpd
    except ImportError:
        print("   ⚠️ geopandas 未安装，空间 buffer 切分不可用")
        _CENTROID_CACHE[city] = {}
        return {}

    gdf = gpd.read_file(_resolve_shp(shp_path))

    geoid_col = next(
        (c for c in ["GEOID", "GEOID20", "GEOID10", "TRACTCE"]
         if c in gdf.columns), None)
    if geoid_col is None:
        print(f"   ⚠️ 找不到 GEOID 列！可用: {gdf.columns.tolist()}")
        _CENTROID_CACHE[city] = {}
        return {}

    gdf["geoid_norm"]  = gdf[geoid_col].astype(str).apply(_normalize_geoid)
    centroid_all       = gdf.geometry.unary_union.centroid
    utm_zone = int((centroid_all.x + 180) / 6) + 1
    epsg     = 32600 + utm_zone if centroid_all.y >= 0 else 32700 + utm_zone
    gdf_proj = gdf.to_crs(epsg=epsg)

    centroids = {}
    for _, row in gdf_proj.iterrows():
        c = row.geometry.centroid
        centroids[row["geoid_norm"]] = (c.x, c.y)

    print(f"   🗺️  {len(centroids)} 个 tract 质心 (EPSG:{epsg})")
    _CENTROID_CACHE[city] = centroids
    return centroids


def _spatial_split_with_buffer(labels_dict, task_type, seed, city, buffer_km):
    tr, va, te = _split_random(labels_dict, task_type, seed)
    centroids  = _load_tract_centroids(city)
    if not centroids:
        print("   ⚠️ 质心加载失败，回退随机切分")
        return tr, va, te

    buffer_m = buffer_km * 1000.0

    def _coords(ids):
        valid = [i for i in ids if i in centroids]
        if not valid:
            return valid, np.empty((0, 2))
        return valid, np.array([centroids[i] for i in valid])

    te_ids, te_xy = _coords(te)
    va_ids, va_xy = _coords(va)
    tr_ids, tr_xy = _coords(tr)

    if te_xy.shape[0] == 0 or tr_xy.shape[0] == 0:
        print("   ⚠️  质心不足，退回随机切分")
        return tr, va, te

    dist_tr_te   = cdist(tr_xy, te_xy)
    tr_keep_mask = dist_tr_te.min(axis=1) > buffer_m

    if va_xy.shape[0] > 0:
        dist_tr_va    = cdist(tr_xy, va_xy)
        tr_keep_mask &= dist_tr_va.min(axis=1) > buffer_m

    tr_filtered = [t for t, keep in zip(tr_ids, tr_keep_mask) if keep]

    if va_xy.shape[0] > 0 and te_xy.shape[0] > 0:
        dist_va_te  = cdist(va_xy, te_xy)
        va_filtered = [v for v, d in zip(va_ids, dist_va_te.min(axis=1))
                       if d > buffer_m]
    else:
        va_filtered = va_ids

    tr_final = tr_filtered + [i for i in tr if i not in centroids]
    va_final = va_filtered + [i for i in va if i not in centroids]

    print(f"   🗺️  Buffer {buffer_km}km: "
          f"train {len(tr)}→{len(tr_final)} (-{len(tr)-len(tr_final)}), "
          f"val {len(va)}→{len(va_final)} (-{len(va)-len(va_final)}), "
          f"test {len(te)}")
    return tr_final, va_final, te


def step1_generate_embeddings():
    fuser         = get_fusion(CONFIG["fusion_method"])
    output_dir    = CONFIG["output_dir"]
    force_rebuild = CONFIG.get("force_rebuild_tract", False)
    os.makedirs(output_dir, exist_ok=True)

    emb_dir = Path(CONFIG["emb_dir"])
    if not emb_dir.exists():
        print(f"   ❌ Embedding 目录不存在: {emb_dir}")
        return

    for city, shp_path in CONFIG["tract_shapefiles"].items():
        city_emb_path = emb_dir / f"region_embs_{city}_final.npz"
        if not city_emb_path.exists():
            continue
        if not Path(shp_path).exists():
            continue

        out_path = os.path.join(
            output_dir, f"{city}_tract_embedding_{fuser.name}.pickle")
        if Path(out_path).exists() and not force_rebuild:
            print(f"   ✅ 已存在: {out_path}")
            continue

        process_city_pipeline(
            city=city,
            emb_path=str(city_emb_path),
            tract_shapefile=shp_path,
            output_dir=output_dir,
            fuser=fuser,
        )


def _normalize_regression_labels(values, log_transform=False):
    raw  = np.asarray(values, dtype=np.float32)
    work = np.log1p(raw) if log_transform else raw
    mean = float(work.mean())
    std  = float(work.std())
    norm = (work - mean) / (std + 1e-6)
    info = {
        "label_norm_method": "zscore",
        "label_mean":  mean,
        "label_std":   std,
        "log_transform": log_transform,
    }
    return norm.astype(np.float32), info

def _inverse_regression_labels(values, info):
    arr  = np.asarray(values, dtype=np.float32)
    std  = float(info["label_std"])
    mean = float(info["label_mean"])
    work = arr * (std + 1e-6) + mean
    if info.get("log_transform", False):
        return np.expm1(work)
    return work


def load_labels_crime(data_path):
    df        = pd.read_csv(data_path)
    id_col    = _detect_col(df, ["GEOID", "region_id", "tract_id", "geoid"])
    label_col = _detect_col(df, ["crime_count", "total_crimes", "count"], "Label")
    raw = df[label_col].values.astype(np.float32)
    labels_log  = np.log1p(raw)
    label_mean  = float(labels_log.mean())
    label_std   = float(labels_log.std())
    labels_norm = (labels_log - label_mean) / (label_std + 1e-6)
    labels_dict = {_normalize_geoid(k): float(v)
                   for k, v in zip(df[id_col].values, labels_norm)}
    info = {
        "task_type":     "regression",
        "log_transform": True,
        "label_mean":    label_mean,
        "label_std":     label_std,
        "output_dim":    1,
        "display_scale": CONFIG["tasks"]["crime"]["display_scale"],
        "display_unit":  CONFIG["tasks"]["crime"]["display_unit"],
    }
    print(f"   Crime: {len(labels_dict)} tracts, raw=[{raw.min():.0f}, {raw.max():.0f}]")
    return labels_dict, info

def load_labels_houseprice(data_path):
    df        = pd.read_csv(data_path)
    id_col    = _detect_col(df, ["tract_geoid", "GEOID", "region_id", "geoid"])
    label_col = _detect_col(df, ["price_median", "price_mean", "sale_price", "price"], "Label")
    df = df.dropna(subset=[id_col, label_col])
    df = df[df[label_col] > 0].copy()
    raw = df[label_col].values.astype(np.float32)
    labels_log  = np.log1p(raw)
    label_mean  = float(labels_log.mean())
    label_std   = float(labels_log.std())
    labels_norm = (labels_log - label_mean) / (label_std + 1e-6)
    labels_dict = {_normalize_geoid(k): float(v)
                   for k, v in zip(df[id_col].values, labels_norm)}
    info = {
        "task_type":     "regression",
        "log_transform": True,
        "label_mean":    label_mean,
        "label_std":     label_std,
        "output_dim":    1,
        "display_scale": CONFIG["tasks"]["houseprice"]["display_scale"],
        "display_unit":  CONFIG["tasks"]["houseprice"]["display_unit"],
    }
    print(f"   HousePrice: {len(labels_dict)} tracts, median=${np.median(raw):,.0f}")
    return labels_dict, info

def load_labels_pm25(data_path):
    df        = pd.read_csv(data_path, dtype={"tract_geoid": str})
    id_col    = _detect_col(df, ["tract_geoid", "GEOID", "region_id", "geoid"])
    label_col = _detect_col(df, ["pm25", "pred_wght", "pm25_mean"], "Label")
    df = df.dropna(subset=[id_col, label_col])
    df = df[df[label_col] > 0].copy()
    raw = df[label_col].values.astype(np.float32)
    label_mean  = float(raw.mean())
    label_std   = float(raw.std())
    labels_norm = (raw - label_mean) / (label_std + 1e-6)
    labels_dict = {_normalize_geoid(k): float(v)
                   for k, v in zip(df[id_col].values, labels_norm)}
    info = {
        "task_type":     "regression",
        "log_transform": False,
        "label_mean":    label_mean,
        "label_std":     label_std,
        "output_dim":    1,
        "display_scale": CONFIG["tasks"]["pm25"]["display_scale"],
        "display_unit":  CONFIG["tasks"]["pm25"]["display_unit"],
    }
    print(f"   PM2.5: {len(labels_dict)} tracts, mean={raw.mean():.2f}")
    return labels_dict, info

def load_labels_census_cls(data_path, task_config):
    df         = pd.read_csv(data_path)
    id_col     = _detect_col(df, ["GEOID", "region_id", "tract_id", "geoid"])
    source_col = task_config["source_col"]
    if source_col not in df.columns:
        for fb in task_config.get("fallback_cols", []):
            if fb in df.columns:
                source_col = fb
                break
        else:
            raise ValueError(f"列不存在: {task_config['source_col']}")
    df         = df.dropna(subset=[id_col, source_col]).copy()
    df         = df[df[source_col].apply(lambda x: np.isfinite(float(x)))].copy()
    n_classes  = task_config.get("n_classes", 4)
    labels, _  = pd.qcut(df[source_col], q=n_classes, labels=False, retbins=True, duplicates="drop")
    df["enc"]  = labels.astype(float)
    actual_cls = int(df["enc"].nunique())
    labels_dict = {_normalize_geoid(row[id_col]): float(row["enc"])
                   for _, row in df.iterrows()}
    le = LabelEncoder()
    le.classes_ = np.array([str(i) for i in range(actual_cls)])
    info = {
        "task_type":     "classification",
        "num_classes":   actual_cls,
        "output_dim":    actual_cls,
        "label_encoder": le,
    }
    print(f"   {source_col} → {actual_cls} 分类, {len(labels_dict)} tracts")
    return labels_dict, info

LABEL_LOADERS = {
    "crime":         lambda p, c: load_labels_crime(p),
    "houseprice":    lambda p, c: load_labels_houseprice(p),
    "pm25":          lambda p, c: load_labels_pm25(p),
    "poverty_cls":   lambda p, c: load_labels_census_cls(p, CONFIG["tasks"]["poverty_cls"]),
    "education_cls": lambda p, c: load_labels_census_cls(p, CONFIG["tasks"]["education_cls"]),
}


# ═══════════════════════════════════════════════════════════════
# ⭐ 指标计算（包含对 mse_real 的补充计算）
# ═══════════════════════════════════════════════════════════════
def compute_metrics(preds, labels, info):
    results   = {}
    task_type = info["task_type"]

    if task_type == "regression":
        p, l = preds.squeeze(), labels.squeeze()

        # 归一化空间指标
        mse  = float(mean_squared_error(l, p))
        rmse = float(np.sqrt(mse))
        mae  = float(mean_absolute_error(l, p))
        r2   = float(r2_score(l, p))

        # ⭐ 反归一化到真实空间
        pr = _inverse_regression_labels(p, info)
        lr = _inverse_regression_labels(l, info)

        scale    = float(info.get("display_scale", 1.0))
        unit     = info.get("display_unit", "")
        pr_d     = pr / scale
        lr_d     = lr / scale

        # 补充：计算并返回 mse_real
        mse_real  = float(mean_squared_error(lr_d, pr_d))
        mae_real  = float(mean_absolute_error(lr_d, pr_d))
        rmse_real = float(np.sqrt(mse_real))
        r2_real   = float(r2_score(lr_d, pr_d))

        results = {
            "mse":       mse,
            "rmse":      rmse,
            "mae":       mae,
            "r2":        r2,
            "mse_real":  mse_real,
            "rmse_real": rmse_real,
            "mae_real":  mae_real,
            "r2_real":   r2_real,
        }

        unit_str = f" 单位:{unit}" if unit else ""
        print(f"   📊 [归一化]  mae={mae:.4f}  r2={r2:.4f}")
        print(f"   📊 [真实空间{unit_str}]  "
              f"mse={mse_real:.2f}  mae={mae_real:.2f}  rmse={rmse_real:.2f}  r2={r2_real:.4f}")

    elif task_type == "classification":
        pc = (preds.argmax(-1) if (preds.ndim > 1 and preds.shape[1] > 1)
              else np.round(preds.squeeze()).astype(int))
        li    = labels.squeeze().astype(int)
        n_cls = int(preds.shape[1]) if (preds.ndim > 1 and preds.shape[1] > 1) else 2
        acc      = float(accuracy_score(li, pc))
        macro_f1 = float(f1_score(li, pc, average="macro", zero_division=0))
        auc_val  = compute_auc_score(preds, labels, num_classes=n_cls, average="macro")
        results.update({"accuracy": acc, "macro_f1": macro_f1, "auc": auc_val})
        print(f"   📊 accuracy={acc:.4f}  macro_f1={macro_f1:.4f}  auc={auc_val:.4f}")

    return results

def evaluate(model, test_loader, device, info):
    model.eval()
    ps, ls = [], []
    with torch.no_grad():
        for b in test_loader:
            ps.append(model(b["embedding"].to(device)).cpu())
            ls.append(b["label"])
    return compute_metrics(
        torch.cat(ps).numpy(), torch.cat(ls).numpy(), info)


def _split_random(labels_dict, task_type, seed):
    all_ids  = sorted(labels_dict.keys())
    stratify = None
    if task_type == "classification":
        la = np.array([labels_dict[t] for t in all_ids]).astype(int)
        cc = Counter(la.tolist())
        if not any(v < 2 for v in cc.values()):
            stratify = la
    try:
        tv, te = train_test_split(all_ids, test_size=CONFIG["test_size"],
                                   random_state=seed, stratify=stratify)
    except ValueError:
        tv, te = train_test_split(all_ids, test_size=CONFIG["test_size"],
                                   random_state=seed)
    vr  = CONFIG["val_size"] / (1 - CONFIG["test_size"])
    stv = None
    if stratify is not None:
        stv = np.array([labels_dict[t] for t in tv]).astype(int)
        if any(v < 2 for v in Counter(stv.tolist()).values()):
            stv = None
    try:
        tr, va = train_test_split(tv, test_size=vr,
                                   random_state=seed, stratify=stv)
    except ValueError:
        tr, va = train_test_split(tv, test_size=vr, random_state=seed)
    return tr, va, te

def _split_with_seed(labels_dict, task_type, seed,
                     city=None, spatial_split=False, buffer_km=1.0):
    if spatial_split and city:
        return _spatial_split_with_buffer(
            labels_dict, task_type, seed, city, buffer_km)
    return _split_random(labels_dict, task_type, seed)


def run_experiment(task_name, city, device):
    data_path    = get_data_path(task_name, city)
    ss           = CONFIG.get("spatial_split", False)
    bk           = CONFIG.get("buffer_km", 1.0)
    split_label  = f"spatial(buf={bk}km)" if ss else "random"

    print(f"\n{'='*70}")
    print(f"🧪 ReCP | {task_name} | {city.upper()} | "
          f"fusion={CONFIG['fusion_method']} | split={split_label}")
    print(f"{'='*70}")

    if not Path(data_path).exists():
        print(f"   ❌ 标签文件不存在: {data_path}")
        return None

    labels_dict, info = LABEL_LOADERS[task_name](data_path, city)
    is_cls = info["task_type"] == "classification"

    embed_path = get_embed_path(city)
    if not Path(embed_path).exists():
        print(f"   ❌ tract embedding 不存在: {embed_path}")
        return None

    with open(embed_path, "rb") as f:
        raw_embs = pickle.load(f)
    emb_dict = {_normalize_geoid(k): np.asarray(v, dtype=np.float32)
                for k, v in raw_embs.items()}

    overlap = sorted(set(emb_dict.keys()) & set(labels_dict.keys()))
    print(f"   匹配: Emb={len(emb_dict)}, Label={len(labels_dict)}, 交集={len(overlap)}")
    if len(overlap) < 10:
        print("   ❌ 交集太少，跳过")
        return None

    labels_dict_filtered = {k: labels_dict[k] for k in overlap}
    input_dim  = next(iter(emb_dict.values())).shape[0]
    output_dim = info.get("output_dim", 1)
    n_classes  = info.get("num_classes") if is_cls else None
    print(f"   dim={input_dim}, tracts={len(overlap)}")

    tag_str      = f"{task_name}_{city}"
    seed_results = {}

    for seed in SEEDS:
        torch.manual_seed(seed)
        np.random.seed(seed)

        tr, va, te = _split_with_seed(
            labels_dict_filtered, info["task_type"], seed,
            city=city, spatial_split=ss, buffer_km=bk)

        print(f"\n   --- seed={seed} ---")
        print(f"   划分: train={len(tr)}, val={len(va)}, test={len(te)}")

        tr_ds = FlatDataset(embed_path, labels_dict_filtered, tr)
        va_ds = FlatDataset(embed_path, labels_dict_filtered, va)
        te_ds = FlatDataset(embed_path, labels_dict_filtered, te)

        if not all(len(d) for d in [tr_ds, va_ds, te_ds]):
            print("   ⚠️  某个 split 为空，跳过")
            continue

        bs    = CONFIG["batch_size"]
        tr_ld = DataLoader(tr_ds, batch_size=bs, shuffle=True, drop_last=len(tr_ds) > bs)
        va_ld = DataLoader(va_ds, batch_size=bs, shuffle=False)
        te_ld = DataLoader(te_ds, batch_size=bs, shuffle=False)

        if task_name in ("crime", "houseprice", "pm25"):
            crit = nn.HuberLoss(delta=1.0)
        elif is_cls:
            crit = nn.CrossEntropyLoss()
        else:
            crit = nn.MSELoss()

        train_config = {
            "lr":       CONFIG["lr"],
            "epochs":   CONFIG["epochs"],
            "patience": CONFIG["patience"],
        }

        model = SimpleMLP(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=CONFIG["hidden_dim"],
            is_classification=is_cls,
            n_classes=n_classes,
            dropout=CONFIG["dropout"],
        ).to(device)

        model = train_model(model, tr_ld, va_ld, crit, device, train_config, is_classification=is_cls)
        m = evaluate(model, te_ld, device, info)
        seed_results[seed] = m

        print(f"seed={seed}")
        for k, v in m.items():
            print(f"{tag_str:<35} {k:<25} {v:.4f}")

    print(f"{'-'*70}")
    return seed_results if seed_results else None

def main():
    parser = argparse.ArgumentParser(description="ReCP 下游任务")
    parser.add_argument("--task", type=str, default=None, choices=list(CONFIG["tasks"].keys()))
    parser.add_argument("--city", type=str, default=None, choices=["chicago", "nyc", "sf"])
    parser.add_argument("--device", default="cuda")

    # 下游 MLP 超参
    parser.add_argument("--batch_size", type=int,   default=CONFIG["batch_size"])
    parser.add_argument("--lr",         type=float, default=CONFIG["lr"])
    parser.add_argument("--epochs",     type=int,   default=CONFIG["epochs"])
    parser.add_argument("--hidden_dim", type=int,   default=CONFIG["hidden_dim"])
    parser.add_argument("--patience",   type=int,   default=CONFIG["patience"])

    # 融合/mnt/network_data/personal_workspace/liyanhuang/hly/DS/model/RECP/tract_embedding/recp/chicago_tract_embedding_full.pickle
    # 这里允许你指定你生成的后缀模式如 'full', 'poi', 'flow'
    parser.add_argument("--fusion", type=str, default="full", help="读取哪个后缀的特征")
    parser.add_argument("--build_tract",   action="store_true")
    parser.add_argument("--rebuild_tract", action="store_true")

    # ⭐ 空间 buffer 切分
    parser.add_argument("--spatial_split", action="store_true", help="启用空间 Buffer 切分")
    parser.add_argument("--buffer_km", type=float, default=1.0, help="Buffer 半径（km），默认 1.0")

    args = parser.parse_args()

    global DEVICE
    DEVICE = torch.device(args.device if torch.cuda.is_available() else "cpu")

    CONFIG["batch_size"]          = args.batch_size
    CONFIG["lr"]                  = args.lr
    CONFIG["epochs"]              = args.epochs
    CONFIG["hidden_dim"]          = args.hidden_dim
    CONFIG["patience"]            = args.patience
    CONFIG["fusion_method"]       = args.fusion
    CONFIG["force_rebuild_tract"] = args.rebuild_tract
    CONFIG["spatial_split"]       = args.spatial_split
    CONFIG["buffer_km"]           = args.buffer_km

    split_str = (f"spatial (buffer={args.buffer_km}km)" if args.spatial_split else "random")

    print(f"\n{'='*70}")
    print(f"🏙️  ReCP Downstream")
    print(f"   Device:   {DEVICE}")
    print(f"   Seeds:    {SEEDS}")
    print(f"   Fusion:   {CONFIG['fusion_method']}")
    print(f"   Split:    {split_str}")
    print(f"   Tasks:    {list(CONFIG['tasks'].keys())}")
    print(f"{'='*70}")

    if args.build_tract or args.rebuild_tract:
        step1_generate_embeddings()

    tasks_to_run = (
        {args.task: CONFIG["tasks"][args.task]} if args.task
        else CONFIG["tasks"]
    )

    summary = {}
    for task_name, task_cfg in tasks_to_run.items():
        if args.city:
            if args.city not in task_cfg["cities"]:
                continue
            cities = [args.city]
        else:
            cities = task_cfg["cities"]
        for city in cities:
            key    = f"{task_name}_{city}"
            result = run_experiment(task_name, city, DEVICE)
            if result:
                summary[key] = result

    split_suffix = (f"_spatial_buf{args.buffer_km}km" if args.spatial_split else "")
    save_dir = Path(CONFIG["save_dir"]) / (CONFIG["fusion_method"] + split_suffix)
    save_dir.mkdir(parents=True, exist_ok=True)

    for key, seed_dict in summary.items():
        serializable = {}
        for seed, metrics in seed_dict.items():
            serializable[str(seed)] = {k: float(v) for k, v in metrics.items()}
        with open(save_dir / f"{key}_results.json", "w") as f:
            json.dump({
                "experiment":    key,
                "model":         "ReCP",
                "fusion":        CONFIG["fusion_method"],
                "spatial_split": args.spatial_split,
                "buffer_km":     args.buffer_km,
                "lr":            CONFIG["lr"],
                **serializable,
            }, f, indent=2)

    print(f"\n{'='*70}")
    print(f"📋 总汇: ReCP | fusion={CONFIG['fusion_method']} | {split_str}")
    print(f"{'='*70}")

    rows = []
    for key, seed_dict in summary.items():
        for seed in sorted(seed_dict.keys()):
            m = seed_dict[seed]
            print(f"seed={seed}")
            for k, v in m.items():
                print(f"{key:<35} {k:<25} {v:.4f}")
            row = {"experiment": key, "seed": seed,
                   "fusion":        CONFIG["fusion_method"],
                   "spatial_split": args.spatial_split,
                   "buffer_km":     args.buffer_km}
            row.update(m)
            rows.append(row)
        print(f"{'-'*70}")

    if rows:
        out_csv = save_dir / "all_results_summary.csv"
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        print(f"\n💾 {out_csv}")

    print(f"\n🎉 完成!")

if __name__ == "__main__":
    main()