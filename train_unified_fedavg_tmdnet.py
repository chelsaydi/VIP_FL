#!/usr/bin/env python3
"""
Vanilla FedAvg using TMDNet + unified team hyperparameters ONLY.

Does not import SimpleNet or legacy per_fedavg_train hyperparameters.
Run from repo:  python Personalized_Federated_Learning_Implementation/train_unified_fedavg_tmdnet.py
Or cd into Personalized_Federated_Learning_Implementation and:  python train_unified_fedavg_tmdnet.py
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import Dict, List

# Allow `python path/to/train_unified_fedavg_tmdnet.py` from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from unified_cellmob_data import (
    WindowDataset,
    build_unified_client_datasets,
    get_federated_test_tensors,
)
from unified_results_log import save_unified_run
from unified_team_protocol import (
    UNIFIED_BATCH_SIZE,
    UNIFIED_LOCAL_EPOCHS,
    UNIFIED_MAX_PER_CLIENT,
    UNIFIED_NUM_ROUNDS,
    new_tmdnet,
)


def evaluate_model(
    model: nn.Module,
    dataset: WindowDataset,
    device: torch.device,
    batch_size: int,
) -> float:
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    correct, total = 0, 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            pred = logits.argmax(dim=1)
            correct += (pred == yb).sum().item()
            total += yb.numel()
    return correct / total if total > 0 else 0.0


def fedavg_round(
    global_model: nn.Module,
    clients: Dict[str, Dict[str, WindowDataset]],
    client_names: List[str],
    device: torch.device,
    lr: float,
) -> nn.Module:
    """One FedAvg round: local train local_epochs on each client, average weights (weighted by n_k)."""
    total_n = 0
    accum: Dict[str, torch.Tensor] | None = None

    for name in client_names:
        local = copy.deepcopy(global_model).to(device)
        local.train()
        opt = torch.optim.SGD(local.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss()
        train_ds = clients[name]["train"]
        n = len(train_ds)
        total_n += n
        loader = DataLoader(train_ds, batch_size=UNIFIED_BATCH_SIZE, shuffle=True)

        for _ in range(UNIFIED_LOCAL_EPOCHS):
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad()
                loss = loss_fn(local(xb), yb)
                loss.backward()
                opt.step()

        sd = local.state_dict()
        if accum is None:
            accum = {k: sd[k].detach().float() * n for k in sd}
        else:
            for k in sd:
                accum[k] = accum[k] + sd[k].detach().float() * n

    assert accum is not None and total_n > 0
    new_sd = {}
    ref = global_model.state_dict()
    for k in accum:
        new_sd[k] = (accum[k] / float(total_n)).to(ref[k].dtype).to(ref[k].device)

    out = copy.deepcopy(global_model)
    out.load_state_dict(new_sd)
    return out.to(device)


def run_unified_fedavg(
    lr: float = 0.01,
    seed: int = 42,
    verbose: bool = True,
) -> Dict[str, object]:
    """
    FedAvg + TMDNet + unified protocol. Saves JSON under results_unified/.
    Callable from Jupyter via: from train_unified_fedavg_tmdnet import run_unified_fedavg
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if verbose:
        print("Device:", device)
        print(
            "Unified settings: batch_size=%s, max_per_client=%s, local_epochs=%s, num_rounds=%s"
            % (UNIFIED_BATCH_SIZE, UNIFIED_MAX_PER_CLIENT, UNIFIED_LOCAL_EPOCHS, UNIFIED_NUM_ROUNDS)
        )

    clients, feature_cols, input_dim, num_classes = build_unified_client_datasets()
    client_names = sorted(clients.keys())
    if verbose:
        print("feature_cols (len=%d): team common features" % len(feature_cols))
        print("Clients:", client_names, "input_dim:", input_dim, "num_classes:", num_classes)

    X_te, y_te = get_federated_test_tensors(clients)
    if verbose:
        print("Federated test tensor shapes (X_test_fl, y_test_fl):", X_te.shape, y_te.shape)

    model = new_tmdnet(input_dim, num_classes, device)

    rounds_log: List[Dict[str, object]] = []
    for r in range(1, UNIFIED_NUM_ROUNDS + 1):
        model = fedavg_round(model, clients, client_names, device, lr=lr)
        accs = {
            name: evaluate_model(model, clients[name]["test"], device, UNIFIED_BATCH_SIZE)
            for name in client_names
        }
        eq = float(np.mean(list(accs.values())))
        n_test = {name: len(clients[name]["test"]) for name in client_names}
        total_t = sum(n_test.values())
        wtd = sum(accs[k] * n_test[k] for k in client_names) / total_t if total_t else 0.0
        rounds_log.append(
            {
                "round": r,
                "equal_weight_avg_pct": round(eq * 100, 4),
                "sample_weighted_pct": round(wtd * 100, 4),
                "per_client_pct": {k: round(accs[k] * 100, 4) for k in client_names},
            }
        )
        if verbose:
            print(
                f"Round {r}/{UNIFIED_NUM_ROUNDS}  equal-weight avg acc: {eq*100:.2f}%  "
                f"sample-weighted: {wtd*100:.2f}%  per-client: "
                + ", ".join(f"{k}={accs[k]*100:.1f}%" for k in client_names)
            )

    last = rounds_log[-1]
    final = {
        "equal_weight_avg_pct": last["equal_weight_avg_pct"],
        "sample_weighted_pct": last["sample_weighted_pct"],
        "per_client_acc_pct": last["per_client_pct"],
    }
    cfg = {
        "seed": seed,
        "local_sgd_lr": lr,
        "UNIFIED_BATCH_SIZE": UNIFIED_BATCH_SIZE,
        "UNIFIED_MAX_PER_CLIENT": UNIFIED_MAX_PER_CLIENT,
        "UNIFIED_LOCAL_EPOCHS": UNIFIED_LOCAL_EPOCHS,
        "UNIFIED_NUM_ROUNDS": UNIFIED_NUM_ROUNDS,
        "input_dim": input_dim,
        "num_classes": num_classes,
    }
    out = save_unified_run("fedavg_tmdnet", cfg, rounds_log, final)
    if verbose:
        print("Results saved to:", out)
        print("\n=== FINAL test accuracy (FedAvg + TMDNet) ===")
        print(f"  Equal-weight average:    {final['equal_weight_avg_pct']:.4f}%")
        print(f"  Sample-weighted average: {final['sample_weighted_pct']:.4f}%")
        print("  Per client (%):", final["per_client_acc_pct"])
        print("Done (unified FedAvg + TMDNet).")
    return {
        "results_json": str(out),
        "final": final,
        "rounds": rounds_log,
        "config": cfg,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="FedAvg + TMDNet (unified team protocol)")
    parser.add_argument("--lr", type=float, default=0.01, help="Local SGD learning rate")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_unified_fedavg(lr=args.lr, seed=args.seed, verbose=True)


if __name__ == "__main__":
    main()
