# aggregate_embedding.py — 静态融合版本

import os
import numpy as np
import pickle
import geopandas as gpd
from pathlib import Path
from shapely.geometry import Polygon
import h3

from fusion import BaseFusion, get_fusion


def h3_to_polygon(h3_idx):
    boundary = h3.cell_to_boundary(h3_idx)
    coords = [(lng, lat) for lat, lng in boundary]
    return Polygon(coords)


def h3_embeddings_to_tract(h3_embeddings, tract_shapefile):
    h3_data = [{'h3': h3_idx, 'geometry': h3_to_polygon(h3_idx)} for h3_idx in h3_embeddings.keys()]
    h3_gdf = gpd.GeoDataFrame(h3_data, crs="EPSG:4326")

    tracts = gpd.read_file(tract_shapefile)
    for col in ['GEOID', 'GEOID20', 'GEOID10']:
        if col in tracts.columns:
            if col != 'GEOID':
                tracts = tracts.rename(columns={col: 'GEOID'})
            break
    tracts = tracts.to_crs("EPSG:4326")
    print(f"   H3 cells: {len(h3_gdf)}, Tracts: {len(tracts)}")

    overlay = gpd.overlay(tracts[['GEOID', 'geometry']], h3_gdf[['h3', 'geometry']], how='intersection')
    overlay['area'] = overlay.to_crs("EPSG:3857").geometry.area
    print(f"   交集数量: {len(overlay)}")

    embedding_dim = len(next(iter(h3_embeddings.values())))
    tract_embeddings = {}

    for geoid, group in overlay.groupby('GEOID'):
        total_area = group['area'].sum()
        if total_area == 0:
            continue
        weights = (group['area'] / total_area).values
        weighted_emb = np.zeros(embedding_dim, dtype=np.float32)
        for w, h3_id in zip(weights, group['h3'].values):
            if h3_id in h3_embeddings:
                weighted_emb += w * h3_embeddings[h3_id]
        tract_embeddings[str(geoid)] = weighted_emb

    all_tracts = set(tracts['GEOID'].astype(str))
    missing = all_tracts - set(tract_embeddings.keys())
    if missing:
        print(f"   最近邻补充 {len(missing)} 个 tract")
        h3_gdf_proj = h3_gdf.to_crs("EPSG:3857")
        h3_gdf_proj['centroid'] = h3_gdf_proj.geometry.centroid
        tracts_proj = tracts.to_crs("EPSG:3857")

        for geoid in missing:
            tract_geom_proj = tracts_proj[tracts_proj['GEOID'].astype(str) == geoid].geometry.values
            if len(tract_geom_proj) == 0:
                continue
            tract_centroid_proj = tract_geom_proj[0].centroid
            distances = h3_gdf_proj['centroid'].distance(tract_centroid_proj)
            nearest_idx = distances.idxmin()
            nearest_h3 = h3_gdf.loc[nearest_idx, 'h3']
            if nearest_h3 in h3_embeddings:
                tract_embeddings[geoid] = h3_embeddings[nearest_h3].copy()

    print(f"   最终 Tract: {len(tract_embeddings)}, dim={embedding_dim}")
    return tract_embeddings


def process_city_pipeline(city, emb_path, tract_shapefile, output_dir,
                          fuser: BaseFusion = None):
    if fuser is None:
        fuser = get_fusion("concat")

    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print(f"🚀 Processing: {city.upper()} | fusion={fuser.name}")
    print("=" * 60)

    print(f"\n📦 Step 1: Loading {emb_path}...")
    data = np.load(emb_path, allow_pickle=True)

    for k in data.files:
        v = data[k]
        if isinstance(v, np.ndarray):
            print(f"   {k}: shape={v.shape}, dtype={v.dtype}")

    dynamic_embs = data["dynamic_embs"]
    static_embs  = data["static_embs"]
    h3_ids       = list(data["h3_ids"])

    assert dynamic_embs.shape[0] == static_embs.shape[0] == len(h3_ids)

    embs = fuser.fuse(dynamic_embs, static_embs)
    out_dim = embs.shape[-1]
    print(f"\n🔗 Step 2: {fuser.name} → ({dynamic_embs.shape[-1]}, {static_embs.shape[-1]}) → {out_dim}D")

    print(f"\n🔄 Step 3: H3 → Tract 空间聚合...")
    h3_emb_dict = {h3_id: embs[i] for i, h3_id in enumerate(h3_ids)}
    tract_embs = h3_embeddings_to_tract(h3_emb_dict, tract_shapefile)

    out_path = os.path.join(output_dir, f"{city}_tract_embedding_{fuser.name}.pickle")
    with open(out_path, 'wb') as f:
        pickle.dump(tract_embs, f)

    sample_geoid = list(tract_embs.keys())[0]
    print(f"\n   ✅ 保存: {out_path}")
    print(f"   格式: {{geoid: np.array({tract_embs[sample_geoid].shape})}}")
    print(f"   Tract 数量: {len(tract_embs)}")
    return out_path
