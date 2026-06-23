"""LSTM 회귀 (정규화 시퀀스 [N,T,C] 입력)."""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class LSTMReg(nn.Module):
    def __init__(self, c, hidden=48, layers=1, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(c, hidden, layers, batch_first=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):                # x: [B, T, C]
        out, (h, _) = self.lstm(x)
        return self.fc(self.drop(h[-1])).squeeze(-1)


def train_predict(seq_tr, y_tr, seq_te, seed=0, epochs=200, lr=1e-3):
    torch.manual_seed(seed); np.random.seed(seed)
    c = seq_tr.shape[-1]
    # y 표준화(학습 통계)
    ym, ys = float(y_tr.mean()), float(y_tr.std() + 1e-6)
    Xtr = torch.tensor(seq_tr, dtype=torch.float32, device=DEVICE)
    ytr = torch.tensor((y_tr - ym) / ys, dtype=torch.float32, device=DEVICE)
    Xte = torch.tensor(seq_te, dtype=torch.float32, device=DEVICE)

    model = LSTMReg(c).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)
    lossf = nn.MSELoss()
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        pred = model(Xtr)
        loss = lossf(pred, ytr)
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        out = model(Xte).cpu().numpy() * ys + ym
    return np.asarray(out, dtype=float)
