#!/usr/bin/env python3
"""
Per-FedAvg (first-order meta-learning) using TMDNet + unified team data pipeline ONLY.

Uses: unified_team_protocol (TMDNet, UNIFIED_*), unified_cellmob_data (no SimpleNet, no legacy per_fedavg_train).

Meta rounds = UNIFIED_NUM_ROUNDS (5). Inner steps / LRs match UNIFIED_INNER_* in unified_team_protocol.py
(same defaults as the original notebook; adjust there for team alignment if needed).

Run:  cd Personalized_Federated_Learning_Implementation && python train_unified_per_fedavg_tmdnet.py
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from unified_cellmob_data import build_unified_client_datasets, get_federated_test_tensors
from unified_results_log import save_unified_run
from unified_team_protocol import (
    UNIFIED_BATCH_SIZE,
    UNIFIED_INNER_LR,
    UNIFIED_INNER_STEPS,
    UNIFIED_MAX_PER_CLIENT,
    UNIFIED_META_LR,
    UNIFIED_NUM_ROUNDS,
    new_tmdnet,
)


def evaluate_personalized(
    model: nn.Module,
    train_ds: Dataset,
    test_ds: Dataset,
    inner_steps: int,
    inner_lr: float,
    device: torch.device,
    batch_size: int,
) -> float:
    local_model = copy.deepcopy(model).to(device)
    optimizer = torch.optim.SGD(local_model.parameters(), lr=inner_lr)
    loss_fn = nn.CrossEntropyLoss()
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    for _ in range(inner_steps):
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = local_model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    correct, total = 0, 0
    local_model.eval()
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = local_model(xb).argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += yb.numel()
    return correct / total if total > 0 else 0.0


def run_unified_per_fedavg(seed: int = 42, verbose: bool = True) -> Dict[str, object]:
    """
    Per-FedAvg + TMDNet + unified protocol. Saves JSON under results_unified/.
    From notebook: from train_unified_per_fedavg_tmdnet import run_unified_per_fedavg
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if verbose:
        print("Device:", device)
        print(
            "Unified: batch=%s max_per_client=%s meta_rounds=%s inner_steps=%s inner_lr=%s meta_lr=%s"
            % (
                UNIFIED_BATCH_SIZE,
                UNIFIED_MAX_PER_CLIENT,
                UNIFIED_NUM_ROUNDS,
                UNIFIED_INNER_STEPS,
                UNIFIED_INNER_LR,
                UNIFIED_META_LR,
            )
        )

    clients, feature_cols, input_dim, num_classes = build_unified_client_datasets()
    client_names = sorted(clients.keys())
    if verbose:
        print("feature_cols len:", len(feature_cols), "clients:", client_names)
    X_te, y_te = get_federated_test_tensors(clients)
    if verbose:
        print("X_test_fl, y_test_fl:", X_te.shape, y_te.shape)

    global_model = new_tmdnet(input_dim, num_classes, device)
    meta_optimizer = torch.optim.Adam(global_model.parameters(), lr=UNIFIED_META_LR)
    loss_fn = nn.CrossEntropyLoss()

    rounds_log: List[Dict[str, Any]] = []
    for epoch in range(1, UNIFIED_NUM_ROUNDS + 1):
        global_model.train()
        meta_optimizer.zero_grad()

        for client_name in client_names:
            train_ds = clients[client_name]["train"]
            train_loader = DataLoader(train_ds, batch_size=UNIFIED_BATCH_SIZE, shuffle=True)

            local_model = copy.deepcopy(global_model).to(device)
            inner_optimizer = torch.optim.SGD(local_model.parameters(), lr=UNIFIED_INNER_LR)

            for _ in range(UNIFIED_INNER_STEPS):
                for xb, yb in train_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    inner_optimizer.zero_grad()
                    logits = local_model(xb)
                    loss = loss_fn(logits, yb)
                    loss.backward()
                    inner_optimizer.step()

            local_model.train()
            inner_grads = [torch.zeros_like(p) for p in global_model.parameters()]
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                local_model.zero_grad()
                logits = local_model(xb)
                loss = loss_fn(logits, yb)
                loss.backward()
                for k, p in enumerate(local_model.parameters()):
                    if p.grad is not None:
                        inner_grads[k] += p.grad.detach()

            num_batches = len(train_loader)
            if num_batches > 0:
                inner_grads = [g / num_batches for g in inner_grads]

            for gp, lg in zip(global_model.parameters(), inner_grads):
                if gp.grad is None:
                    gp.grad = lg.clone()
                else:
                    gp.grad += lg

        num_clients = len(client_names)
        for p in global_model.parameters():
            if p.grad is not None:
                p.grad /= num_clients

        meta_optimizer.step()

        accs: Dict[str, float] = {}
        for client_name in client_names:
            acc = evaluate_personalized(
                global_model,
                clients[client_name]["train"],
                clients[client_name]["test"],
                inner_steps=UNIFIED_INNER_STEPS,
                inner_lr=UNIFIED_INNER_LR,
                device=device,
                batch_size=UNIFIED_BATCH_SIZE,
            )
            accs[client_name] = acc

        avg_acc = float(np.mean(list(accs.values())))
        n_test = {name: len(clients[name]["test"]) for name in client_names}
        total_t = sum(n_test.values())
        wtd = sum(accs[k] * n_test[k] for k in client_names) / total_t if total_t else 0.0
        acc_str = ", ".join(f"{name}: {accs[name]*100:.1f}%" for name in client_names)
        rounds_log.append(
            {
                "meta_round": epoch,
                "equal_weight_avg_pct": round(avg_acc * 100, 4),
                "sample_weighted_pct": round(wtd * 100, 4),
                "per_client_pct": {k: round(accs[k] * 100, 4) for k in client_names},
            }
        )
        if verbose:
            print(
                f"[Meta {epoch}/{UNIFIED_NUM_ROUNDS}] eq-weight avg: {avg_acc*100:.1f}%  "
                f"sample-weighted: {wtd*100:.1f}%  | {acc_str}"
            )

    last = rounds_log[-1]
    final = {
        "equal_weight_avg_pct": last["equal_weight_avg_pct"],
        "sample_weighted_pct": last["sample_weighted_pct"],
        "per_client_acc_pct": last["per_client_pct"],
    }
    cfg = {
        "seed": seed,
        "UNIFIED_BATCH_SIZE": UNIFIED_BATCH_SIZE,
        "UNIFIED_MAX_PER_CLIENT": UNIFIED_MAX_PER_CLIENT,
        "UNIFIED_NUM_ROUNDS": UNIFIED_NUM_ROUNDS,
        "UNIFIED_INNER_STEPS": UNIFIED_INNER_STEPS,
        "UNIFIED_INNER_LR": UNIFIED_INNER_LR,
        "UNIFIED_META_LR": UNIFIED_META_LR,
        "input_dim": input_dim,
        "num_classes": num_classes,
    }
    out = save_unified_run("per_fedavg_tmdnet", cfg, rounds_log, final)
    if verbose:
        print("Results saved to:", out)
        print("\n=== FINAL test accuracy (personalized: Per-FedAvg + TMDNet) ===")
        print(f"  Equal-weight average:    {final['equal_weight_avg_pct']:.4f}%")
        print(f"  Sample-weighted average: {final['sample_weighted_pct']:.4f}%")
        print("  Per client (%):", final["per_client_acc_pct"])
        print("Unified Per-FedAvg + TMDNet complete.")
    return {
        "results_json": str(out),
        "final": final,
        "rounds": rounds_log,
        "config": cfg,
    }


def train_unified_per_fedavg_cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_unified_per_fedavg(seed=args.seed, verbose=True)


if __name__ == "__main__":
    train_unified_per_fedavg_cli()
