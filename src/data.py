"""MNIST 데이터 파이프라인 (양쪽 모델 완전 동일).

- train 60k -> pool 54k / val 6k 를 seed 고정으로 분할. val은 모든 subset 크기에서 고정.
- 데이터 효율성 축: pool에서 stratified(클래스 균등) + nested 서브셋을 뽑는다.
- test 10k 은 개발 중 절대 사용하지 않고 마지막 evaluate.py / eval_robustness.py 에서만.
- train 증강 없음, 표준 정규화만.
"""
from __future__ import annotations
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

NUM_CLASSES = 10


def build_transform(mean: float, std: float) -> transforms.Compose:
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((mean,), (std,)),
    ])


def _stratified_nested_indices(targets, pool_idx, size, subset_seed):
    """pool_idx 안에서 클래스당 size/10 개를 뽑는다.

    per-class 셔플을 subset_seed로 고정하고 prefix를 취하므로,
    같은 seed에서 size=100 의 선택은 size=500 의 선택의 부분집합(nested)이 된다.
    """
    per_class = size // NUM_CLASSES
    assert per_class * NUM_CLASSES == size, "subset_size 는 10의 배수여야 함 (클래스 균등)"
    pool_targets = targets[pool_idx]
    chosen = []
    for c in range(NUM_CLASSES):
        cls_local = (pool_targets == c).nonzero(as_tuple=True)[0]     # pool 내 위치
        cls_global = pool_idx[cls_local].numpy()                      # 원본 dataset 인덱스
        rng = np.random.default_rng(subset_seed + c)                 # size와 무관 -> nested
        perm = rng.permutation(len(cls_global))
        chosen.extend(cls_global[perm[:per_class]].tolist())
    return chosen


def get_dataloaders(cfg: dict):
    d = cfg["data"]
    tfm = build_transform(d["mean"], d["std"])

    full_train = datasets.MNIST(d["root"], train=True, download=True, transform=tfm)
    test_set = datasets.MNIST(d["root"], train=False, download=True, transform=tfm)
    targets = full_train.targets  # LongTensor [60000]

    # seed 고정 분할: pool(54k) + val(6k)
    gen = torch.Generator().manual_seed(cfg["seed"])
    perm = torch.randperm(len(full_train), generator=gen)
    val_idx = perm[: d["val_size"]]
    pool_idx = perm[d["val_size"]:]

    subset_size = d.get("subset_size", None)
    if subset_size is None:
        train_idx = pool_idx.tolist()
    else:
        train_idx = _stratified_nested_indices(targets, pool_idx, subset_size, d["subset_seed"])

    train_set = Subset(full_train, train_idx)
    val_set = Subset(full_train, val_idx.tolist())

    # 작은 n에서는 batch를 데이터 크기로 클램프
    bs = min(d["batch_size"], len(train_set))
    common = dict(num_workers=d["num_workers"], pin_memory=True)
    train_loader = DataLoader(train_set, batch_size=bs, shuffle=True, **common)
    val_loader = DataLoader(val_set, batch_size=d["batch_size"], shuffle=False, **common)
    test_loader = DataLoader(test_set, batch_size=d["batch_size"], shuffle=False, **common)
    return train_loader, val_loader, test_loader
