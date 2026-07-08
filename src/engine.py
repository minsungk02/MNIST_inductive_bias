"""학습 엔진: train/val 루프, warmup+cosine 스케줄, early stopping.

best model = val_loss 최소 체크포인트 (양쪽 모델 동일 규칙).
history dict 를 반환해서 나중에 compare.py 가 곡선/표를 그릴 수 있게 한다.
"""
from __future__ import annotations
import math
from pathlib import Path
import torch
import torch.nn as nn


def build_optimizer(model, cfg):
    o = cfg["optim"]
    return torch.optim.AdamW(
        model.parameters(), lr=o["lr"],
        weight_decay=o["weight_decay"], betas=tuple(o["betas"]),
    )


def build_scheduler(optimizer, cfg):
    warmup = cfg["schedule"]["warmup_epochs"]
    max_epochs = cfg["train"]["max_epochs"]

    def lr_lambda(epoch):  # epoch 단위 warmup 후 cosine decay
        if epoch < warmup:
            return (epoch + 1) / max(1, warmup)
        progress = (epoch - warmup) / max(1, max_epochs - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        n += x.size(0)
    return total_loss / n, correct / n


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        n += x.size(0)
    return total_loss / n, correct / n


def fit(model, train_loader, val_loader, cfg, device, ckpt_path: Path):
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg["train"]["label_smoothing"])
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)

    patience = cfg["train"]["patience"]
    min_delta = cfg["train"]["min_delta"]

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}
    best_val = float("inf")
    best_epoch = -1
    epochs_no_improve = 0

    for epoch in range(cfg["train"]["max_epochs"]):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        va_loss, va_acc = evaluate(model, val_loader, criterion, device)
        cur_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)
        history["lr"].append(cur_lr)

        improved = va_loss < best_val - min_delta
        if improved:
            best_val, best_epoch = va_loss, epoch
            epochs_no_improve = 0
            torch.save({"model": model.state_dict(), "epoch": epoch,
                        "val_loss": va_loss, "val_acc": va_acc, "cfg": cfg}, ckpt_path)
        else:
            epochs_no_improve += 1

        print(f"[{epoch+1:02d}/{cfg['train']['max_epochs']}] "
              f"lr={cur_lr:.2e} | train {tr_loss:.4f}/{tr_acc:.4f} | "
              f"val {va_loss:.4f}/{va_acc:.4f}"
              f"{'  <-- best' if improved else ''}")

        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch+1} (best epoch {best_epoch+1}, val_loss {best_val:.4f})")
            break

    history["best_epoch"] = best_epoch
    history["best_val_loss"] = best_val
    return history
