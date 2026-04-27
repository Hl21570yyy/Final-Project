#python main.py --cities chicago nyc  --suffix full
#python main.py --cities chicago nyc  --suffix w_o_recon
#python main.py --cities chicago nyc  --suffix w_o_intra_cl
"""
lambda1=1,
lambda2=1,

lambda1=1,
lambda2=0,  # <-- 关掉重构损失

lambda1=0,  # <-- 关掉视图内对比损失
lambda2=1,

"""
import argparse
import itertools
import warnings
import json
import numpy as np
import random
import torch
from torch.optim import lr_scheduler
from pathlib import Path
from datetime import datetime

from model import ReCP
from recp_data import ReData

def get_config(n_cat, flow_dim):
    return dict(
        Prediction=dict(
            arch1=[48, 96, 48],
            arch2=[48, 96, 48],
        ),
        Autoencoder=dict(
            arch1=[n_cat, 128, 128, 48],
            arch2=[flow_dim, 128, 128, 48],
            activations1='relu',
            activations2='relu',
            batchnorm=True,
        ),
        training=dict(
            seed=42,
            start_dual_prediction=50,
            epoch=1000,
            lr=0.01,
            alpha=9,
            lambda1=0,
            lambda2=1,
            sigma=0.0001,
        ),
        print_num=100,
    )

def train_city(city, data_root='dataset', output_root='output', device_str='cuda', suffix=''):
    warnings.filterwarnings("ignore")
    device = torch.device(device_str if torch.cuda.is_available() else 'cpu')

    print(f"\n{'='*55}")
    print(f"🏙️  Training: {city.upper()}")
    print(f"{'='*55}")

    # ========== 1. 加载数据 ==========
    redata = ReData(city=city, data_root=data_root)

    n_cat    = redata.a_m.shape[1]
    flow_dim = redata.s_m.shape[1]
    R        = redata.a_m.shape[0]

    print(f"  Regions:    {R}")
    print(f"  POI cats:   {n_cat}")
    print(f"  Flow dim:   {flow_dim}")
    print(f"  Device:     {device}")

    # ========== 2. 配置 ==========
    config = get_config(n_cat, flow_dim)

    seed = config['training']['seed']
    np.random.seed(seed)
    random.seed(seed + 1)
    torch.manual_seed(seed + 2)
    torch.cuda.manual_seed(seed + 3)
    torch.backends.cudnn.deterministic = True

    # ========== 3. 构建模型 ==========
    recp = ReCP(config)
    optimizer = torch.optim.Adam(
        itertools.chain(
            recp.autoencoder_a.parameters(),
            recp.autoencoder_s.parameters(),
            recp.autoencoder_d.parameters(),
            recp.a2mo.parameters(),
            recp.mo2a.parameters(),
        ), lr=config['training']['lr'], weight_decay=1e-6)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.3)
    recp.to_device(device)

    # ========== 4. 转 tensor ==========
    xs = [
        torch.from_numpy(redata.a_m).float().to(device),
        torch.from_numpy(redata.s_m).float().to(device),
        torch.from_numpy(redata.d_m).float().to(device),
    ]

    # ========== 5. 训练 ==========
    print(f"\n  🚀 Training {config['training']['epoch']} epochs...")
    recp.train(config, redata, xs, optimizer, scheduler, device)

    # ========== 6. 提取并保存分离嵌入 ==========
    print(f"\n  📦 Extracting single-view embeddings...")
    emb_dict = recp.get_embeddings(xs)

    with open(Path(data_root) / city / 'common_h3.json') as f:
        h3_ids = json.load(f)

    # 根据 suffix 或时间戳生成输出文件夹，避免覆盖
    time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"{city}_{suffix}" if suffix else f"{city}_{time_str}"
    out_dir = Path(output_root) / folder_name
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / 'latent_a.npy', emb_dict['latent_a'])
    np.save(out_dir / 'latent_m.npy', emb_dict['latent_m'])

    h3_to_a = {h3: emb_dict['latent_a'][i].tolist() for i, h3 in enumerate(h3_ids)}
    h3_to_m = {h3: emb_dict['latent_m'][i].tolist() for i, h3 in enumerate(h3_ids)}
    
    with open(out_dir / 'h3_latent_a.json', 'w') as f:
        json.dump(h3_to_a, f)
    with open(out_dir / 'h3_latent_m.json', 'w') as f:
        json.dump(h3_to_m, f)

    print(f"  💾 Saved to {out_dir}/")
    print(f"     latent_a.npy shape={emb_dict['latent_a'].shape}")
    print(f"     latent_m.npy shape={emb_dict['latent_m'].shape}")
    print(f"  ✅ {city.upper()} done!")

    return emb_dict, h3_ids

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cities', nargs='+', default=['chicago', 'nyc'])
    parser.add_argument('--data_root', type=str, default='dataset')
    parser.add_argument('--output_root', type=str, default='output')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--suffix', type=str, default='', help='给输出文件夹加后缀，如: w_o_recon')
    args = parser.parse_args()

    print(f"🚀 ReCP Multi-City Training")
    print(f"   Cities: {args.cities}")

    for city in args.cities:
        city_data_dir = Path(args.data_root) / city
        if not city_data_dir.exists():
            print(f"\n  ❌ {city_data_dir} 不存在，跳过 {city}")
            continue
        train_city(city, args.data_root, args.output_root, args.device, args.suffix)

    print(f"\n{'='*55}")
    print(f"✅ All cities done!")

if __name__ == '__main__':
    main()