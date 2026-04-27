"""
python recp2tract.py --suffix full --mode full

python recp2tract.py --suffix full --mode poi

python recp2tract.py --suffix full --mode flow

python recp2tract.py --suffix w_o_intra_cl --mode full

python recp2tract.py --suffix w_o_recon --mode full
"""

import argparse
import json
import pickle
import numpy as np
from pathlib import Path
from aggregate_embedding import h3_embeddings_to_tract

# 统一路径到你的个人工作空间
BASE_PATH = "/mnt/network_data/personal_workspace/liyanhuang/hly"

TRACT_SHAPEFILES = {
    "chicago": f"{BASE_PATH}/data_1.16/DS_Census/tl_2021_17_tract",
    "nyc":     f"{BASE_PATH}/data_1.16/DS_Census/tl_2021_36_tract",
    "sf":      f"{BASE_PATH}/data_1.16/DS_Census/tl_2021_06_tract",
}

def convert_city(city, recp_output_dir='output', tract_output_dir='tract_embedding/recp', mode='full', suffix='full'):
    recp_dir = Path(recp_output_dir)

    # 寻找 common_h3.json
    h3_path = None
    for p in [
        recp_dir / 'common_h3.json',
        Path(f"{BASE_PATH}/RECP/dataset") / city / 'common_h3.json',
        Path(f"{BASE_PATH}/dataset") / city / 'common_h3.json',
        Path('dataset') / city / 'common_h3.json',
    ]:
        if p.exists():
            h3_path = p
            break

    if h3_path is None:
        print(f"  ❌ {city} 找不到 common_h3.json，跳过")
        return

    # === 核心消融逻辑：根据 mode 读取不同的特征 ===
    try:
        if mode == 'poi':
            embeddings = np.load(recp_dir / 'latent_a.npy')
            print(f"  --> [Ablation] 只使用 POI 特征 (w/o Flow)")
        elif mode == 'flow':
            embeddings = np.load(recp_dir / 'latent_m.npy')
            print(f"  --> [Ablation] 只使用 Flow 特征 (w/o POI)")
        elif mode == 'full':
            latent_a = np.load(recp_dir / 'latent_a.npy')
            latent_m = np.load(recp_dir / 'latent_m.npy')
            embeddings = np.concatenate([latent_a, latent_m], axis=1)
            print(f"  --> [Full/Ablation Model] 拼接 POI 和 Flow 特征")
        else:
            raise ValueError("mode 必须是 'poi', 'flow' 或 'full'")
    except FileNotFoundError as e:
        print(f"  ❌ 找不到特征文件，请确认是否跑完了 train.py 或 suffix 是否拼写正确: \n      {e}")
        return

    with open(h3_path) as f:
        h3_ids = json.load(f)

    # ==========================================
    # ⭐ 智能命名逻辑：组合 suffix 和 mode 防止覆盖
    # ==========================================
    if suffix == 'full' and mode == 'full':
        tag = 'full'
    elif suffix == 'full':
        tag = mode  # 'poi' 或 'flow'
    elif mode == 'full':
        tag = suffix # 'w_o_intra_cl' 等等
    else:
        tag = f"{suffix}_{mode}"

    print(f"\n{'='*50}")
    print(f"🏙️  {city.upper()} [Tag: {tag.upper()}]: {len(h3_ids)} H3 regions, {embeddings.shape[1]}-dim")

    # H3 dict → tract 聚合
    h3_dict = {h3_ids[i]: embeddings[i] for i in range(len(h3_ids))}
    tract_embs = h3_embeddings_to_tract(h3_dict, TRACT_SHAPEFILES[city])

    # 保存
    out_dir = Path(tract_output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 动态使用 tag 命名
    out_path = out_dir / f"{city}_tract_embedding_{tag}.pickle"

    with open(out_path, 'wb') as f:
        pickle.dump(tract_embs, f)

    sample_dim = next(iter(tract_embs.values())).shape[0]
    print(f"  💾 {out_path}")
    print(f"     Tracts: {len(tract_embs)}, dim: {sample_dim}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cities', nargs='+', default=['chicago', 'nyc'])
    parser.add_argument('--recp_output_dir', type=str, default=f"{BASE_PATH}/RECP/output") 
    parser.add_argument('--tract_output_dir', type=str, default='tract_embedding/recp')
    
    parser.add_argument('--mode', type=str, default='full', choices=['full', 'poi', 'flow'], help='选择提取什么模态特征')
    parser.add_argument('--suffix', type=str, default='full', help='你在 train.py 训练时生成的文件夹后缀 (如: w_o_intra_cl)')
    args = parser.parse_args()

    print(f"🔄 ReCP → Tract embedding (Mode: {args.mode}, Suffix: {args.suffix})\n")
    for city in args.cities:
        # 动态拼接特征读取的文件夹路径
        city_output_dir = Path(args.recp_output_dir) / f"{city}_{args.suffix}" 
        # 把 suffix 也传进去处理命名
        convert_city(city, city_output_dir, args.tract_output_dir, args.mode, args.suffix)
    print("\n✅ Done!")


if __name__ == '__main__':
    main()