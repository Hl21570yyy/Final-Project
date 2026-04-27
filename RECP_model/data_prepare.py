"""
输入:
  - /mnt/share/hly/new_code/poi/processed/{city}_poi_processed.npz
  - /mnt/share/hly/new_code/poi/processed/{city}_region_indices.json
  - /mnt/share/hly/new_code/flow/{city}/{city}_flow_tensors.pt

输出 (每个城市一个目录):
  - dataset/{city}/attribute_m.npy    (R, n_cat)
  - dataset/{city}/source_m.npy       (R, 28)
  - dataset/{city}/destina_m.npy      (R, 28)
  - dataset/{city}/common_h3.json     对齐后的区域列表
  - dataset/{city}/meta.json          元信息 (R, n_cat, flow_dim)
"""

import argparse
import numpy as np
import json
import torch
from pathlib import Path


def prepare_city(city,
                 poi_dir='/mnt/share/hly/new_code/poi/processed',
                 flow_dir='/mnt/share/hly/new_code/flow',
                 out_root='dataset'):

    out_dir = Path(out_root) / city
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*50}")
    print(f"🏙️  {city.upper()}")
    print(f"{'='*50}")

    # -------- POI --------
    poi_path = f"{poi_dir}/{city}_poi_processed.npz"
    idx_path = f"{poi_dir}/{city}_region_indices.json"

    if not Path(poi_path).exists():
        print(f"  ❌ 找不到 {poi_path}，跳过")
        return None
    if not Path(idx_path).exists():
        print(f"  ❌ 找不到 {idx_path}，跳过")
        return None

    poi = np.load(poi_path, allow_pickle=True)
    cat_ids = poi["cat_ids"]
    with open(idx_path) as f:
        poi_idx = json.load(f)

    # -------- Flow --------
    flow_path = f"{flow_dir}/{city}/{city}_flow_tensors.pt"
    if not Path(flow_path).exists():
        print(f"  ❌ 找不到 {flow_path}，跳过")
        return None

    fd = torch.load(flow_path, map_location='cpu', weights_only=False)
    flow_volumes = fd['flow_volumes'].numpy()   # (R, 7, 2, 4)
    flow_h3 = list(fd['h3_regions'])

    # -------- 对齐 --------
    common = sorted(set(poi_idx.keys()) & set(flow_h3))
    h3_to_fidx = {h: i for i, h in enumerate(flow_h3)}
    R = len(common)
    n_cat = int(cat_ids.max()) + 1

    print(f"  Regions: POI={len(poi_idx)}, Flow={len(flow_h3)}, Common={R}")
    print(f"  POI categories: {n_cat}")

    if R == 0:
        print(f"  ❌ 无交集区域，跳过")
        return None

    # -------- View A: POI 类别分布 --------
    attr = np.zeros((R, n_cat), dtype=np.float32)
    for i, h3 in enumerate(common):
        s, e = poi_idx[h3]
        for c in cat_ids[s:e]:
            attr[i, int(c)] += 1
    row_sum = attr.sum(1, keepdims=True)
    row_sum[row_sum == 0] = 1
    attr /= row_sum

    # -------- View S/D: Flow 展平 --------
    fidx = [h3_to_fidx[h] for h in common]
    aligned = flow_volumes[fidx]                        # (R, 7, 2, 4)
    source  = aligned[:, :, 0, :].reshape(R, -1)       # (R, 28)
    destina = aligned[:, :, 1, :].reshape(R, -1)       # (R, 28)

    # Min-Max 归一化
    for m in [source, destina]:
        mn, mx = m.min(), m.max()
        if mx > mn:
            m[:] = (m - mn) / (mx - mn)

    # -------- 保存 --------
    np.save(out_dir / 'attribute_m.npy', attr)
    np.save(out_dir / 'source_m.npy', source)
    np.save(out_dir / 'destina_m.npy', destina)
    with open(out_dir / 'common_h3.json', 'w') as f:
        json.dump(common, f)

    # 保存元信息，方便后续读取时自动配置
    meta = dict(city=city, R=R, n_cat=n_cat, flow_dim=source.shape[1])
    with open(out_dir / 'meta.json', 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"  ✅ Saved to {out_dir}/")
    print(f"     attr={attr.shape}, src={source.shape}, dst={destina.shape}")

    return meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cities', nargs='+', default=['chicago', 'nyc', 'sf'])
    parser.add_argument('--poi_dir', type=str, default='/mnt/share/hly/new_code/poi/processed')
    parser.add_argument('--flow_dir', type=str, default='/mnt/share/hly/new_code/flow')
    parser.add_argument('--out_root', type=str, default='dataset')
    args = parser.parse_args()

    print(f"🚀 Data Preparation for ReCP")
    print(f"   Cities: {args.cities}")

    results = {}
    for city in args.cities:
        meta = prepare_city(city, args.poi_dir, args.flow_dir, args.out_root)
        if meta:
            results[city] = meta

    # 汇总
    print(f"\n{'='*50}")
    print(f"📊 Summary")
    print(f"{'='*50}")
    print(f"  {'City':<12} {'R':>6} {'n_cat':>6} {'flow_dim':>9}")
    print(f"  {'-'*35}")
    for city, m in results.items():
        print(f"  {city:<12} {m['R']:>6} {m['n_cat']:>6} {m['flow_dim']:>9}")


if __name__ == '__main__':
    main()
