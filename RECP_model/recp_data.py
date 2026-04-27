import numpy as np
from pathlib import Path

class ReData:
    def __init__(self, city, data_root='dataset'):
        data_dir = Path(data_root) / city
        self.city = city
        self.a_m = np.load(data_dir / 'attribute_m.npy')   # (R, n_cat)
        self.s_m = np.load(data_dir / 'source_m.npy')      # (R, flow_dim)
        self.d_m = np.load(data_dir / 'destina_m.npy')     # (R, flow_dim)
        print(f"  [{city.upper()}] R={self.a_m.shape[0]}, "
              f"n_cat={self.a_m.shape[1]}, flow_dim={self.s_m.shape[1]}")

    def gaussian_noise(self, matrix, seed, mean=0, sigma=0.03):
        np.random.seed(seed)
        matrix = matrix.copy()
        noise = np.random.normal(mean, sigma, matrix.shape)
        matrix = np.clip(matrix + noise, 0, 1)
        return matrix

    def get_aug(self, seed=42):
        poi_augs = [self.gaussian_noise(self.a_m, seed + i, sigma=0.05)
                    for i in range(3)]
        s_augs = [self.gaussian_noise(self.s_m, seed + 100 + i, sigma=0.03)
                  for i in range(4)]
        d_augs = [self.gaussian_noise(self.d_m, seed + 200 + i, sigma=0.03)
                  for i in range(4)]
        return [poi_augs, s_augs, d_augs]