# train.py — 下游 MLP 训练（纯静态融合版）

import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from copy import deepcopy


def _normalize_geoid(geoid) -> str:
    s = str(geoid).strip()
    if '.' in s:
        s = s.split('.')[0]
    return s


class FlatDataset(Dataset):
    def __init__(self, embed_path, labels_dict, tract_ids=None):
        with open(embed_path, 'rb') as f:
            raw_embs = pickle.load(f)

        emb_dict = {_normalize_geoid(k): v for k, v in raw_embs.items()}
        sample_emb = next(iter(emb_dict.values()))
        print(f"      ✅ 加载 pickle: {len(emb_dict)} tracts, dim={sample_emb.shape}")

        if tract_ids is not None:
            valid_ids = [t for t in tract_ids if t in emb_dict and t in labels_dict]
        else:
            valid_ids = [t for t in emb_dict if t in labels_dict]

        self.embeddings = []
        self.labels = []
        self.tract_ids = []

        for tid in valid_ids:
            self.embeddings.append(emb_dict[tid].astype(np.float32))
            label = labels_dict[tid]
            if isinstance(label, np.ndarray):
                self.labels.append(label.astype(np.float32))
            else:
                self.labels.append(np.float32(label))
            self.tract_ids.append(tid)

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        emb = torch.from_numpy(self.embeddings[idx])
        label = self.labels[idx]
        if isinstance(label, np.ndarray):
            label = torch.from_numpy(label)
        else:
            label = torch.tensor(label, dtype=torch.float32)
        return {"embedding": emb, "label": label}


class SimpleMLP(nn.Module):
    def __init__(self, input_dim, output_dim=1, hidden_dim=256,
                 is_classification=False, n_classes=None, dropout=0.3):
        super().__init__()
        self.is_classification = is_classification
        self.n_classes = n_classes

        if is_classification and n_classes is not None:
            final_dim = n_classes
        else:
            final_dim = output_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.BatchNorm1d(hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout / 2),

            nn.Linear(hidden_dim // 4, final_dim),
        )
        self._final_dim = final_dim

    def forward(self, x):
        out = self.net(x)
        if not self.is_classification and self._final_dim == 1:
            out = out.squeeze(-1)
        return out


def train_model(model, train_loader, val_loader, criterion, device, config,
                is_classification=False):
    lr       = config.get("lr", 1e-3)
    epochs   = config.get("epochs", 100)
    patience = config.get("patience", 15)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

    best_val_loss = float('inf')
    best_state    = None
    wait          = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        n_batches  = 0
        for batch in train_loader:
            x = batch["embedding"].to(device)
            y = batch["label"].to(device)
            if not is_classification and y.ndim == 2 and y.shape[-1] == 1:
                y = y.squeeze(-1)
            pred = model(x)
            if is_classification:
                y = y.long()
                if y.ndim > 1:
                    y = y.squeeze(-1)
            loss = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
            n_batches  += 1

        avg_train = train_loss / max(n_batches, 1)

        model.eval()
        val_loss = 0.0
        n_val    = 0
        with torch.no_grad():
            for batch in val_loader:
                x = batch["embedding"].to(device)
                y = batch["label"].to(device)
                if not is_classification and y.ndim == 2 and y.shape[-1] == 1:
                    y = y.squeeze(-1)
                pred = model(x)
                if is_classification:
                    y = y.long()
                    if y.ndim > 1:
                        y = y.squeeze(-1)
                loss = criterion(pred, y)
                val_loss += loss.item()
                n_val    += 1

        avg_val = val_loss / max(n_val, 1)

        if epoch % 10 == 0 or epoch <= 3:
            print(f"      Epoch {epoch:3d}: train_loss={avg_train:.4f}  "
                  f"val_loss={avg_val:.4f}  lr={lr:.2e}")

        if avg_val < best_val_loss - 1e-6:
            best_val_loss = avg_val
            best_state    = deepcopy(model.state_dict())
            wait          = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"   Early stop at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model
