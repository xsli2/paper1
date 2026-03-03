import math
import random
from dataclasses import dataclass
from typing import Dict, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


# -----------------------------
# Reproducibility
# -----------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -----------------------------
# Data generating mechanisms
# -----------------------------
def generate_xy(mech: str, n: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    mech:
      - "linear_add": y = x1 + 0.5 x2
      - "nonlinear":  y = x1 + 0.5 x2^2
      - "interaction": y = x1 + 0.5 x2 + 0.3 x1 x2
    """
    x1 = torch.randn(n, 1, device=device)
    x2 = torch.randn(n, 1, device=device)
    x = torch.cat([x1, x2], dim=1)

    if mech == "linear_add":
        y = x1 + 0.5 * x2
    elif mech == "nonlinear":
        y = x1 + 0.5 * (x2 ** 2)
    elif mech == "interaction":
        y = x1 + 0.5 * x2 + 0.3 * x1 * x2
    else:
        raise ValueError(f"Unknown mech: {mech}")

    return x, y


# -----------------------------
# Interventions (do-operations on X1)
# -----------------------------
def intervene(x: torch.Tensor, kind: str, c: float = 1.0, sigma: float = 1.0) -> torch.Tensor:
    """
    kind:
      - "shift":   x1 := x1 + c
      - "scale":   x1 := 2 * x1
      - "noise":   x1 := x1 + eps, eps ~ N(0, sigma^2)
    """
    x_do = x.clone()
    if kind == "shift":
        x_do[:, 0] = x_do[:, 0] + c
    elif kind == "scale":
        x_do[:, 0] = 2.0 * x_do[:, 0]
    elif kind == "noise":
        x_do[:, 0] = x_do[:, 0] + sigma * torch.randn_like(x_do[:, 0])
    else:
        raise ValueError(f"Unknown intervention: {kind}")
    return x_do


# -----------------------------
# Models
# -----------------------------
class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, depth: int):
        super().__init__()
        assert depth >= 1
        layers = []
        d = in_dim
        for _ in range(depth):
            layers.append(nn.Linear(d, hidden))
            layers.append(nn.ReLU())
            d = hidden
        self.feat = nn.Sequential(*layers)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor, return_rep: bool = False):
        h = self.feat(x)
        y = self.head(h)
        if return_rep:
            return y, h
        return y


class ResidualMLP(nn.Module):
    """
    Residual blocks in hidden space:
      h_{k+1} = h_k + F_k(h_k)
    """
    def __init__(self, in_dim: int, hidden: int, blocks: int):
        super().__init__()
        self.in_proj = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU())
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
            )
            for _ in range(blocks)
        ])
        self.act = nn.ReLU()
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor, return_rep: bool = False):
        h = self.in_proj(x)
        for blk in self.blocks:
            h = h + blk(h)
            h = self.act(h)
        y = self.head(h)
        if return_rep:
            return y, h
        return y


# -----------------------------
# Training & Metrics
# -----------------------------
@dataclass
class TrainCfg:
    lr: float = 3e-3
    epochs: int = 300
    batch_size: int = 256


def train_regressor(model: nn.Module, x_train: torch.Tensor, y_train: torch.Tensor, cfg: TrainCfg):
    model.train()
    opt = optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = nn.MSELoss()

    n = x_train.shape[0]
    for ep in range(cfg.epochs):
        idx = torch.randperm(n, device=x_train.device)
        for i in range(0, n, cfg.batch_size):
            b = idx[i:i + cfg.batch_size]
            xb, yb = x_train[b], y_train[b]
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()


@torch.no_grad()
def eval_metrics(model: nn.Module, x: torch.Tensor, y: torch.Tensor, x_do: torch.Tensor) -> Dict[str, float]:
    model.eval()
    # base
    pred, rep = model(x, return_rep=True)
    # intervened
    pred_do, rep_do = model(x_do, return_rep=True)

    mse = torch.mean((pred - y) ** 2).item()
    # Output shift: E |f(x) - f(x_do)|
    s_out = torch.mean(torch.abs(pred - pred_do)).item()
    # Representation shift: E ||h(x) - h(x_do)||_2
    s_rep = torch.mean(torch.norm(rep - rep_do, dim=1)).item()

    return {"mse": mse, "s_out": s_out, "s_rep": s_rep}


# -----------------------------
# Main experiment
# -----------------------------
def main():
    set_seed(7)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    mechs = ["linear_add", "nonlinear", "interaction"]
    interventions = ["shift", "scale", "noise"]
    models = {
        "MLP_shallow": lambda: MLP(in_dim=2, hidden=64, depth=2),
        "MLP_deep": lambda: MLP(in_dim=2, hidden=64, depth=6),
        "ResMLP": lambda: ResidualMLP(in_dim=2, hidden=64, blocks=4),
    }

    cfg = TrainCfg(lr=3e-3, epochs=300, batch_size=256)

    # data sizes
    n_train, n_test = 8000, 4000

    # intervention params
    shift_c = 1.0
    noise_sigma = 1.0

    results = []  # list of dicts for table and plotting

    for mech in mechs:
        x_train, y_train = generate_xy(mech, n_train, device)
        x_test, y_test = generate_xy(mech, n_test, device)

        for model_name, builder in models.items():
            model = builder().to(device)
            train_regressor(model, x_train, y_train, cfg)

            for iv in interventions:
                x_do = intervene(x_test, iv, c=shift_c, sigma=noise_sigma)
                m = eval_metrics(model, x_test, y_test, x_do)
                row = {
                    "mech": mech,
                    "model": model_name,
                    "intervention": iv,
                    **m
                }
                results.append(row)

                print(f"[{mech:11s}] [{model_name:10s}] [{iv:5s}] "
                      f"mse={m['mse']:.4f}  S_out={m['s_out']:.4f}  S_rep={m['s_rep']:.4f}")

    # --------
    # Aggregate and plot
    # --------
    # Convert to arrays for plotting
    # Plot: for each mech, bar-like lines by model across interventions (simple line plots)
    def plot_metric(metric: str, title: str):
        plt.figure()
        for mech in mechs:
            # average across interventions? No: show per intervention, per model
            # We'll plot for each model a line across interventions for this mech, and save separate figures per mech.
            pass

    # Make 2 figures per mechanism: S_out and S_rep
    for mech in mechs:
        # collect
        mech_rows = [r for r in results if r["mech"] == mech]
        x_axis = list(range(len(interventions)))

        # S_out
        plt.figure()
        for model_name in models.keys():
            ys = [next(r["s_out"] for r in mech_rows if r["model"] == model_name and r["intervention"] == iv)
                  for iv in interventions]
            plt.plot(x_axis, ys, marker="o", label=model_name)
        plt.xticks(x_axis, interventions)
        plt.xlabel("Intervention on X1")
        plt.ylabel("E|f(x)-f(x_do)|")
        plt.title(f"S_out (Output Shift) - {mech}")
        plt.legend()
        plt.tight_layout()

        # S_rep
        plt.figure()
        for model_name in models.keys():
            ys = [next(r["s_rep"] for r in mech_rows if r["model"] == model_name and r["intervention"] == iv)
                  for iv in interventions]
            plt.plot(x_axis, ys, marker="o", label=model_name)
        plt.xticks(x_axis, interventions)
        plt.xlabel("Intervention on X1")
        plt.ylabel("E||h(x)-h(x_do)||_2")
        plt.title(f"S_rep (Representation Shift) - {mech}")
        plt.legend()
        plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()