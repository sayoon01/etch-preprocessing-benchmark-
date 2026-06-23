"""Compact PatchTST 회귀 (정규화 시퀀스 [N,T,C] 입력).

PatchTST 핵심 아이디어 축약: channel-independent 패치 임베딩 → Transformer
encoder(패치 시퀀스) → 패치 평균풀 → 채널 결합 → 회귀 헤드.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class PatchTST(nn.Module):
    def __init__(self, c, seq_len, patch_len=16, stride=8,
                 d_model=64, n_heads=4, layers=2, dropout=0.2):
        super().__init__()
        self.c = c
        self.patch_len = patch_len
        self.stride = stride
        self.n_patches = max(1, (seq_len - patch_len) // stride + 1)
        self.embed = nn.Linear(patch_len, d_model)
        self.pos = nn.Parameter(torch.randn(1, self.n_patches, d_model) * 0.02)
        enc = nn.TransformerEncoderLayer(d_model, n_heads, d_model * 2,
                                         dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc, layers)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(d_model * c, 1)

    def _patch(self, x):                       # x: [B,T,C] -> [B*C, n_patches, patch_len]
        B, T, C = x.shape
        x = x.permute(0, 2, 1)                 # [B,C,T]
        patches = x.unfold(-1, self.patch_len, self.stride)  # [B,C,n_patches,patch_len]
        return patches.reshape(B * C, self.n_patches, self.patch_len), B, C

    def forward(self, x):
        p, B, C = self._patch(x)
        z = self.embed(p) + self.pos           # [B*C, n_patches, d_model]
        z = self.encoder(z)
        z = z.mean(1)                          # [B*C, d_model]  패치 평균풀
        z = z.reshape(B, C * z.shape[-1])      # 채널 결합
        return self.head(self.drop(z)).squeeze(-1)


def train_predict(seq_tr, y_tr, seq_te, seed=0, epochs=200, lr=1e-3):
    torch.manual_seed(seed); np.random.seed(seed)
    N, T, C = seq_tr.shape
    patch_len = min(16, T)
    stride = max(1, patch_len // 2)
    ym, ys = float(y_tr.mean()), float(y_tr.std() + 1e-6)
    Xtr = torch.tensor(seq_tr, dtype=torch.float32, device=DEVICE)
    ytr = torch.tensor((y_tr - ym) / ys, dtype=torch.float32, device=DEVICE)
    Xte = torch.tensor(seq_te, dtype=torch.float32, device=DEVICE)

    model = PatchTST(C, T, patch_len=patch_len, stride=stride).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)
    lossf = nn.MSELoss()
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = lossf(model(Xtr), ytr)
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        out = model(Xte).cpu().numpy() * ys + ym
    return np.asarray(out, dtype=float)
