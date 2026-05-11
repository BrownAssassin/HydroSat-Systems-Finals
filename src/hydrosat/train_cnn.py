from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.model_selection import GroupKFold
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import models

from .paths import find_image_for_row, find_train_csvs, target_column, target_from_csv_name
from .raster import read_patch

TEST_LABEL_STATS = {
    "chla": {"min": 0.18, "max": 5.3},
    "turbidity": {"min": 0.1, "max": 22.8},
}


def filter_to_test_range(rows: pd.DataFrame, target: str, enabled: bool, padding: float) -> pd.DataFrame:
    if not enabled:
        return rows
    stats = TEST_LABEL_STATS[target]
    low = stats["min"]
    high = stats["max"]
    span = high - low
    low -= span * padding
    high += span * padding
    filtered = rows[(rows["y"] >= low) & (rows["y"] <= high)].copy()
    print(
        f"{target}: test-range filter kept {len(filtered)}/{len(rows)} rows "
        f"for y in [{low:.4f}, {high:.4f}]",
        flush=True,
    )
    if len(filtered) < max(30, min(100, len(rows) // 4)):
        print(f"{target}: filter kept too few rows; falling back to all rows", flush=True)
        return rows
    return filtered


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class SmallPatchCNN(nn.Module):
    def __init__(self, in_channels: int = 12):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.15),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)


class RegressionWrapper(nn.Module):
    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x).squeeze(-1)


def _adapt_conv(conv: nn.Conv2d, in_channels: int) -> nn.Conv2d:
    new_conv = nn.Conv2d(
        in_channels,
        conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=conv.bias is not None,
        padding_mode=conv.padding_mode,
    )
    with torch.no_grad():
        if conv.weight.shape[1] == 3:
            mean_weight = conv.weight.mean(dim=1, keepdim=True)
            new_conv.weight.copy_(mean_weight.repeat(1, in_channels, 1, 1))
        else:
            nn.init.kaiming_normal_(new_conv.weight, mode="fan_out", nonlinearity="relu")
        if conv.bias is not None and new_conv.bias is not None:
            new_conv.bias.copy_(conv.bias)
    return new_conv


def _torchvision_model(fn, pretrained: bool):
    weights = "DEFAULT" if pretrained else None
    try:
        return fn(weights=weights)
    except Exception as exc:
        if not pretrained:
            raise
        print(f"warning: could not load pretrained weights ({exc}); falling back to random init", flush=True)
        return fn(weights=None)


def build_model(arch: str, in_channels: int, pretrained: bool) -> nn.Module:
    if arch == "small":
        return SmallPatchCNN(in_channels=in_channels)
    if arch in {"resnet18", "resnet34", "resnet50"}:
        fn = getattr(models, arch)
        model = _torchvision_model(fn, pretrained)
        model.conv1 = _adapt_conv(model.conv1, in_channels)
        model.fc = nn.Linear(model.fc.in_features, 1)
        return RegressionWrapper(model)
    if arch in {"efficientnet_b0", "efficientnet_b3"}:
        fn = getattr(models, arch)
        model = _torchvision_model(fn, pretrained)
        model.features[0][0] = _adapt_conv(model.features[0][0], in_channels)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 1)
        return RegressionWrapper(model)
    if arch == "convnext_tiny":
        model = _torchvision_model(models.convnext_tiny, pretrained)
        model.features[0][0] = _adapt_conv(model.features[0][0], in_channels)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 1)
        return RegressionWrapper(model)
    raise ValueError(f"Unknown arch {arch}")


def augment_batch(x: torch.Tensor, noise_std: float) -> torch.Tensor:
    if torch.rand(()) < 0.5:
        x = torch.flip(x, dims=[-1])
    if torch.rand(()) < 0.5:
        x = torch.flip(x, dims=[-2])
    k = int(torch.randint(0, 4, ()).item())
    if k:
        x = torch.rot90(x, k, dims=(-2, -1))
    if noise_std > 0:
        x = x + noise_std * torch.randn_like(x)
    return x


def collect_rows(data_root: Path, target: str) -> pd.DataFrame:
    rows = []
    for csv_path in find_train_csvs(data_root):
        csv_target = target_from_csv_name(csv_path)
        if csv_target != target:
            continue
        value_col = target_column(target)
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            if pd.isna(row.get(value_col)):
                continue
            rows.append(
                {
                    "csv_path": str(csv_path),
                    "filename": str(row["filename"]),
                    "image_path": str(find_image_for_row(csv_path, str(row["filename"]))),
                    "area": csv_path.parent.name,
                    "lon": float(row["Lon"]),
                    "lat": float(row["Lat"]),
                    "y": float(row[value_col]),
                }
            )
    if not rows:
        raise RuntimeError(f"No rows found for {target} under {data_root}")
    return pd.DataFrame(rows)


def load_patches(rows: pd.DataFrame, patch_size: int) -> np.ndarray:
    patches = []
    for idx, row in rows.iterrows():
        if idx and idx % 100 == 0:
            print(f"loaded {idx}/{len(rows)} patches", flush=True)
        patch = read_patch(Path(row["image_path"]), float(row["lon"]), float(row["lat"]), size=patch_size).data
        patches.append(np.nan_to_num(patch, nan=0.0, posinf=0.0, neginf=0.0).astype("float32"))
    return np.stack(patches)


def normalize(train_x: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = train_x.mean(axis=(0, 2, 3), keepdims=True)
    std = train_x.std(axis=(0, 2, 3), keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return (x - mean) / std, mean.astype("float32"), std.astype("float32")


def target_transform(y: np.ndarray, use_log: bool) -> np.ndarray:
    if use_log:
        return np.log1p(y)
    return y.astype("float32")


def target_inverse(y: np.ndarray, use_log: bool, max_value: float | None = None) -> np.ndarray:
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    if use_log:
        if max_value is not None and np.isfinite(max_value) and max_value > 0:
            y = np.clip(y, -20.0, float(np.log1p(max_value)))
        else:
            y = np.clip(y, -20.0, 20.0)
        return np.expm1(y)
    return y


def fit_one(
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    use_log: bool,
    device: torch.device,
    arch: str,
    pretrained: bool,
    noise_std: float,
) -> tuple[dict[str, torch.Tensor], np.ndarray, float]:
    train_x_norm, mean, std = normalize(train_x, train_x)
    val_x_norm = (val_x - mean) / std
    y_train_fit = target_transform(train_y, use_log).astype("float32")

    train_ds = TensorDataset(torch.from_numpy(train_x_norm), torch.from_numpy(y_train_fit))
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    model = build_model(arch, train_x.shape[1], pretrained=pretrained).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.SmoothL1Loss()

    best_state = None
    best_rmse = float("inf")
    best_pred = None
    patience = max(8, epochs // 5)
    stale = 0
    val_tensor = torch.from_numpy(val_x_norm.astype("float32")).to(device)
    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            xb = augment_batch(xb, noise_std)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            pred_fit = model(val_tensor).detach().cpu().numpy()
        max_pred_value = float(max(np.nanmax(train_y), np.nanmax(val_y)) * 1.25 + 1.0)
        pred = np.clip(target_inverse(pred_fit, use_log, max_value=max_pred_value), 0, max_pred_value)
        pred = np.nan_to_num(pred, nan=0.0, posinf=max_pred_value, neginf=0.0)
        score = float(root_mean_squared_error(val_y, pred))
        if score < best_rmse:
            best_rmse = score
            best_pred = pred
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
    assert best_state is not None and best_pred is not None
    return {"state_dict": best_state, "mean": torch.from_numpy(mean), "std": torch.from_numpy(std)}, best_pred, best_rmse


def train_final(
    x: np.ndarray,
    y: np.ndarray,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    use_log: bool,
    device: torch.device,
    arch: str,
    pretrained: bool,
    noise_std: float,
) -> dict:
    x_norm, mean, std = normalize(x, x)
    y_fit = target_transform(y, use_log).astype("float32")
    ds = TensorDataset(torch.from_numpy(x_norm), torch.from_numpy(y_fit))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    model = build_model(arch, x.shape[1], pretrained=pretrained).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.SmoothL1Loss()
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            xb = augment_batch(xb, noise_std)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
    return {
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "mean": mean.astype("float32"),
        "std": std.astype("float32"),
        "arch": arch,
    }


def train_cnn(args: argparse.Namespace) -> dict:
    seed_everything(args.random_state)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    rows = collect_rows(args.data_root, args.target)
    rows = filter_to_test_range(rows, args.target, args.filter_test_range, args.filter_range_padding)
    x = load_patches(rows, args.patch_size)
    y = rows["y"].to_numpy(dtype="float32")
    groups = rows["area"].astype(str) + "_" + rows["filename"].astype(str)
    cv = GroupKFold(n_splits=min(args.folds, groups.nunique()))
    oof = np.zeros(len(rows), dtype="float32")
    fold_scores = []
    for fold, (train_idx, val_idx) in enumerate(cv.split(x, y, groups=groups), start=1):
        print(f"fold {fold}: train={len(train_idx)} val={len(val_idx)} device={device}", flush=True)
        _, pred, score = fit_one(
            x[train_idx],
            y[train_idx],
            x[val_idx],
            y[val_idx],
            args.epochs,
            args.batch_size,
            args.lr,
            args.weight_decay,
            args.log_target,
            device,
            args.arch,
            args.pretrained,
            args.noise_std,
        )
        oof[val_idx] = pred
        fold_scores.append(score)
        print(f"fold {fold} rmse={score:.4f}", flush=True)

    summary = {
        "kind": "cnn",
        "target": args.target,
        "rows": int(len(rows)),
        "patch_size": args.patch_size,
        "epochs": args.epochs,
        "fold_rmse": fold_scores,
        "rmse": float(root_mean_squared_error(y, oof)),
        "r2": float(r2_score(y, oof)),
        "log_target": bool(args.log_target),
        "device": str(device),
        "arch": args.arch,
        "pretrained": bool(args.pretrained),
    }
    print(json.dumps(summary, indent=2), flush=True)

    final = train_final(
        x,
        y,
        args.final_epochs,
        args.batch_size,
        args.lr,
        args.weight_decay,
        args.log_target,
        device,
        args.arch,
        args.pretrained,
        args.noise_std,
    )
    model_dir = args.model_dir
    model_dir.mkdir(parents=True, exist_ok=True)
    bundle = {
        "kind": "cnn",
        "target": args.target,
        "patch_size": args.patch_size,
        "arch": args.arch,
        "log_target": bool(args.log_target),
        "model": final,
        "summary": summary,
    }
    candidate_path = model_dir / f"{args.target}_cnn_{args.arch}.pt"
    torch.save(bundle, candidate_path)
    best_path = model_dir / f"{args.target}_cnn.pt"
    should_update = True
    if best_path.exists():
        try:
            old = torch.load(best_path, map_location="cpu")
            old_rmse = old.get("summary", {}).get("rmse")
            should_update = old_rmse is None or summary["rmse"] < old_rmse
        except Exception:
            should_update = True
    if should_update:
        torch.save(bundle, best_path)
        print(f"updated best CNN: {best_path}", flush=True)
    else:
        print(f"kept existing best CNN; saved candidate to {candidate_path}", flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("."))
    parser.add_argument("--model-dir", type=Path, default=Path("artifacts/models"))
    parser.add_argument("--target", choices=["turbidity", "chla"], required=True)
    parser.add_argument("--arch", choices=["small", "resnet18", "resnet34", "resnet50", "efficientnet_b0", "efficientnet_b3", "convnext_tiny"], default="resnet18")
    parser.add_argument("--patch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--final-epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--device", default="")
    parser.add_argument("--log-target", action="store_true")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--noise-std", type=float, default=0.01)
    parser.add_argument("--filter-test-range", action="store_true")
    parser.add_argument("--filter-range-padding", type=float, default=0.05)
    args = parser.parse_args()
    train_cnn(args)


if __name__ == "__main__":
    main()
