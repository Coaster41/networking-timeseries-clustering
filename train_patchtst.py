#!/usr/bin/env python3
import os
import argparse
import json
from typing import Tuple, Optional, Sequence, Dict, Any, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DataParallel, DistributedDataParallel as DDP

from tqdm import tqdm
from transformers import PatchTSTConfig, PatchTSTForClassification
import torch.optim as optim


class MultiFolderFirstSecondsTimeseriesDataset(Dataset):
    """
    Recursively scan `root_dir` for subfolders that contain both:
      - cluster_results.csv (cluster labels and block start timestamps)
      - clustering_data.csv (long per-timestep series)

    For each found pair (directory), extract the first `window_seconds` of each valid block (id)
    and build a global dataset of windows.

    Returns:
      - x: Tensor of shape (T, C), where T = window timesteps and C depends on `channels`
      - y: LongTensor scalar with the (optionally remapped) cluster label
    """

    def __init__(
        self,
        root_dir: str,
        cluster_filename: str = "cluster_results.csv",
        timeseries_filename: str = "clustering_data.csv",
        window_seconds: float = 1.0,
        freq_ms: int = 10,
        channels: Sequence[str] = ("nearest", "token"),
        restrict_clusters: Optional[Sequence[int]] = None,
        ensure_exact_len: bool = True,
        nearest_pad_value: Optional[float] = None,
        dtype_token: np.dtype = np.int16,
        dtype_nearest: np.dtype = np.float32,
        remap_labels: bool = False,
        verbose: bool = True,
        max_pairs: Optional[int] = None,
    ):
        assert set(channels).issubset({"nearest", "token"}), "channels must be subset of {'nearest','token'}"

        self.root_dir = root_dir
        self.cluster_filename = cluster_filename
        self.timeseries_filename = timeseries_filename
        self.window_seconds = float(window_seconds)
        self.freq_ms = int(freq_ms)
        self.channels = tuple(channels)
        self.ensure_exact_len = ensure_exact_len
        self.nearest_pad_value = nearest_pad_value
        self.dtype_token = dtype_token
        self.dtype_nearest = dtype_nearest
        self.remap_labels = remap_labels
        self.verbose = verbose
        self.max_pairs = max_pairs

        self.samples_per_sec = int(1000 // self.freq_ms)
        self.T = int(round(self.window_seconds * self.samples_per_sec))

        # 0) Discover directory pairs
        pairs = self._discover_pairs()
        if self.max_pairs is not None:
            pairs = pairs[: self.max_pairs]
        if not pairs:
            raise RuntimeError(f"No directory pairs found under {self.root_dir} containing both CSVs.")

        # 1) Process each pair and collect windows
        records = []
        expected_delta = pd.Timedelta(milliseconds=self.freq_ms)

        for source_idx, (dir_path, cluster_path, ts_path) in enumerate(pairs):
            if self.verbose:
                print(f"[{source_idx+1}/{len(pairs)}] Processing {dir_path}")

            clusters_df = pd.read_csv(cluster_path)
            if "timestamp" in clusters_df.columns:
                clusters_df["timestamp"] = pd.to_datetime(clusters_df["timestamp"])
            else:
                if self.verbose:
                    print(f"  Skipping (no 'timestamp' in {cluster_path})")
                continue

            clusters_df = clusters_df[["id", "timestamp", "cluster"]].drop_duplicates()
            clusters_df["id"] = clusters_df["id"].astype(np.int32)
            clusters_df["cluster"] = clusters_df["cluster"].astype(np.int64)

            if restrict_clusters is not None:
                clusters_df = clusters_df[clusters_df["cluster"].isin(restrict_clusters)].copy()

            try:
                long_df = pd.read_csv(
                    ts_path,
                    usecols=["id", "timestamp", "rtt_token", "rtt_nearest"],
                    parse_dates=["timestamp"],
                )
            except Exception as e:
                if self.verbose:
                    print(f"  Failed to read {ts_path}: {e}")
                continue

            long_df["id"] = long_df["id"].astype(np.int32)
            long_df["rtt_token"] = long_df["rtt_token"].astype(self.dtype_token)
            long_df["rtt_nearest"] = long_df["rtt_nearest"].astype(self.dtype_nearest)
            long_df = long_df.sort_values(["id", "timestamp"]).reset_index(drop=True)

            valid_ids = np.intersect1d(clusters_df["id"].unique(), long_df["id"].unique())
            if len(valid_ids) == 0:
                if self.verbose:
                    print(f"  No overlapping ids between {cluster_path} and {ts_path}")
                continue
            clusters_df = clusters_df[clusters_df["id"].isin(valid_ids)].copy()
            long_df = long_df[long_df["id"].isin(valid_ids)].copy()

            id_groups: Dict[int, pd.DataFrame] = {
                int(id_val): grp for id_val, grp in long_df.groupby("id", sort=True)
            }

            for _, row in clusters_df.iterrows():
                local_id = int(row["id"])
                start_ts: pd.Timestamp = row["timestamp"]
                cluster = int(row["cluster"])

                end_ts = start_ts + (self.T - 1) * expected_delta
                grp = id_groups.get(local_id, None)
                if grp is None:
                    continue

                window_df = grp[(grp["timestamp"] >= start_ts) & (grp["timestamp"] <= end_ts)]
                w_token = window_df["rtt_token"].to_numpy(dtype=self.dtype_token, copy=True)
                w_near = window_df["rtt_nearest"].to_numpy(dtype=self.dtype_nearest, copy=True)

                w_token = self._pad_or_truncate_token(w_token, self.T)
                w_near = self._pad_or_truncate_nearest(w_near, self.T)

                records.append((source_idx, dir_path, local_id, start_ts, cluster, w_near, w_token))

        if not records:
            raise RuntimeError("No windows collected. Check directory structure and CSV contents.")

        self.source_idx = np.array([r[0] for r in records], dtype=np.int32)
        self.source_dir = np.array([r[1] for r in records], dtype=object)
        self.local_ids = np.array([r[2] for r in records], dtype=np.int32)
        self.start_ts = np.array([r[3] for r in records], dtype="datetime64[ns]")
        original_labels = np.array([r[4] for r in records], dtype=np.int64)

        if self.remap_labels:
            uniq = np.unique(original_labels)
            self.label_map: Dict[int, int] = {orig: i for i, orig in enumerate(sorted(uniq))}
            self.labels = np.array([self.label_map[int(v)] for v in original_labels], dtype=np.int64)
        else:
            self.labels = original_labels
            self.label_map = None

        self.X_nearest = np.stack([r[5] for r in records], axis=0).astype(self.dtype_nearest)  # (N, T)
        self.X_token = np.stack([r[6] for r in records], axis=0).astype(self.dtype_token)      # (N, T)

        self.num_clusters = int(len(np.unique(self.labels)))

        if self.verbose:
            print(f"Built dataset with {len(self.labels)} samples from {len(np.unique(self.source_idx))} directory pairs.")
            print(f"Window length T={self.T} timesteps, channels={self.channels}, num_clusters={self.num_clusters}")

    def _discover_pairs(self) -> List[Tuple[str, str, str]]:
        pairs = []
        for dirpath, dirnames, filenames in os.walk(self.root_dir):
            if self.cluster_filename in filenames and self.timeseries_filename in filenames:
                cluster_path = os.path.join(dirpath, self.cluster_filename)
                ts_path = os.path.join(dirpath, self.timeseries_filename)
                pairs.append((dirpath, cluster_path, ts_path))
        return sorted(pairs, key=lambda x: x[0])

    def _pad_or_truncate_token(self, arr: np.ndarray, T: int) -> np.ndarray:
        if not self.ensure_exact_len:
            return arr
        if len(arr) == T:
            return arr
        if len(arr) > T:
            return arr[:T]
        pad_len = T - len(arr)
        pad = np.full(pad_len, -1, dtype=self.dtype_token)
        return np.concatenate([arr, pad], axis=0)

    def _pad_or_truncate_nearest(self, arr: np.ndarray, T: int) -> np.ndarray:
        if not self.ensure_exact_len:
            return arr
        if len(arr) == T:
            return arr
        if len(arr) > T:
            return arr[:T]
        pad_len = T - len(arr)
        if self.nearest_pad_value is not None:
            pad_val = self.nearest_pad_value
        else:
            pad_val = arr[-1] if len(arr) > 0 else np.nan
        pad = np.full(pad_len, pad_val, dtype=self.dtype_nearest)
        return np.concatenate([arr, pad], axis=0)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        chans = []
        if "nearest" in self.channels:
            chans.append(torch.from_numpy(self.X_nearest[idx]).float())
        if "token" in self.channels:
            chans.append(torch.from_numpy(self.X_token[idx]).float())

        x = torch.stack(chans, dim=0).T  # (T, C)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y

    def get_meta(self, idx: int) -> Dict[str, Any]:
        return {
            "source_idx": int(self.source_idx[idx]),
            "source_dir": str(self.source_dir[idx]),
            "local_id": int(self.local_ids[idx]),
            "start_ts": pd.Timestamp(self.start_ts[idx]),
            "original_label": int(self.labels[idx]),
            "label_map": self.label_map,
        }


class AverageMeter:
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0
    def update(self, val, n=1):
        self.val = float(val)
        self.sum += float(val) * n
        self.count += n
        self.avg = self.sum / max(1, self.count)


def parse_channels(channels_str: str) -> Tuple[str, ...]:
    if not channels_str:
        return ("nearest",)
    parts = [p.strip() for p in channels_str.split(",") if p.strip()]
    if not parts:
        parts = ["nearest"]
    for p in parts:
        if p not in {"nearest", "token"}:
            raise ValueError("channels must be a comma-separated subset of {'nearest','token'}")
    return tuple(parts)


def parse_int_list(list_str: Optional[str]) -> Optional[List[int]]:
    if list_str is None or list_str.strip() == "":
        return None
    return [int(s.strip()) for s in list_str.split(",") if s.strip()]


def stratified_split_indices(labels: np.ndarray, test_size: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    unique, inverse = np.unique(labels, return_inverse=True)
    train_indices = []
    test_indices = []
    for cls_idx in range(len(unique)):
        cls_indices = np.where(inverse == cls_idx)[0]
        rng.shuffle(cls_indices)
        n_test = max(1, int(round(len(cls_indices) * test_size)))
        n_test = min(n_test, len(cls_indices) - 1) if len(cls_indices) > 1 else 1
        test_indices.extend(cls_indices[:n_test].tolist())
        train_indices.extend(cls_indices[n_test:].tolist())
    return np.array(train_indices, dtype=np.int64), np.array(test_indices, dtype=np.int64)


def build_dataloaders(dataset: MultiFolderFirstSecondsTimeseriesDataset,
                      batch_size: int,
                      num_workers: int,
                      test_size: float,
                      seed: int,
                      distributed: bool,
                      drop_last: bool = False) -> Tuple[DataLoader, DataLoader, Optional[DistributedSampler], Optional[DistributedSampler]]:
    train_idx, test_idx = stratified_split_indices(dataset.labels, test_size=test_size, seed=seed)

    train_subset = torch.utils.data.Subset(dataset, train_idx.tolist())
    test_subset = torch.utils.data.Subset(dataset, test_idx.tolist())

    train_sampler = None
    test_sampler = None
    if distributed:
        train_sampler = DistributedSampler(train_subset, shuffle=True, drop_last=drop_last)
        test_sampler = DistributedSampler(test_subset, shuffle=False, drop_last=False)

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=(not distributed),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last,
        persistent_workers=(num_workers > 0),
    )
    test_loader = DataLoader(
        test_subset,
        batch_size=batch_size,
        shuffle=False,
        sampler=test_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=(num_workers > 0),
    )
    return train_loader, test_loader, train_sampler, test_sampler


def setup_device_and_distributed(args):
    use_cuda = torch.cuda.is_available()

    if not args.distributed:
        device = torch.device("cuda", 0) if use_cuda else torch.device("cpu")
        return device, False, 0, 1, True, 0

    # Detect torchrun env first
    rank = None
    world_size = None
    local_rank = None

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
    elif "SLURM_PROCID" in os.environ and "SLURM_NTASKS" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        world_size = int(os.environ["SLURM_NTASKS"])
        if "SLURM_LOCALID" in os.environ:
            local_rank = int(os.environ["SLURM_LOCALID"])
        else:
            ngpus = torch.cuda.device_count() if use_cuda else 1
            local_rank = rank % max(1, ngpus)
    else:
        raise RuntimeError(
            "Distributed mode requested but neither torchrun nor Slurm environment variables were found."
        )

    if use_cuda:
        torch.cuda.set_device(local_rank)
        backend = "nccl"
    else:
        backend = "gloo"

    torch.distributed.init_process_group(backend=backend, init_method="env://", rank=rank, world_size=world_size)

    device = torch.device("cuda", local_rank) if use_cuda else torch.device("cpu")
    is_main = (rank == 0)
    return device, True, rank, world_size, is_main, local_rank


def unwrap_model(model, distributed: bool, data_parallel: bool):
    if distributed or data_parallel:
        return model.module
    return model


def get_logits(outputs):
    # Robustly get logits from transformers ModelOutput or dict
    if hasattr(outputs, "logits"):
        return outputs.logits
    if isinstance(outputs, dict):
        if "logits" in outputs:
            return outputs["logits"]
        if "prediction_logits" in outputs:
            return outputs["prediction_logits"]
    if hasattr(outputs, "prediction_logits"):
        return outputs.prediction_logits
    raise AttributeError("Could not find logits in model outputs.")


def evaluate(model, data_loader, device, distributed: bool, amp_eval: bool = False):
    model.eval()
    total_samples = 0
    correct = 0
    loss_sum = 0.0

    with torch.no_grad():
        iterator = data_loader
        for x_batch, y_batch in iterator:
            x_batch = x_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)

            if amp_eval and device.type == "cuda":
                with torch.cuda.amp.autocast():
                    outputs = model(past_values=x_batch, target_values=y_batch)
            else:
                outputs = model(past_values=x_batch, target_values=y_batch)

            logits = get_logits(outputs)
            preds = torch.argmax(logits, dim=1)
            batch_correct = (preds == y_batch).sum().item()
            batch_total = y_batch.shape[0]
            # Loss is average per batch; weight by batch size to sum over samples
            batch_loss = outputs.loss.item() * batch_total if hasattr(outputs, "loss") and outputs.loss is not None else 0.0

            correct += batch_correct
            total_samples += batch_total
            loss_sum += batch_loss

    if distributed:
        t_correct = torch.tensor([correct], dtype=torch.long, device=device)
        t_total = torch.tensor([total_samples], dtype=torch.long, device=device)
        t_loss = torch.tensor([loss_sum], dtype=torch.float32, device=device)
        torch.distributed.all_reduce(t_correct, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(t_total, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(t_loss, op=torch.distributed.ReduceOp.SUM)
        correct = int(t_correct.item())
        total_samples = int(t_total.item())
        loss_sum = float(t_loss.item())

    val_acc = correct / max(1, total_samples)
    val_loss = loss_sum / max(1, total_samples)
    return {"val_acc": val_acc, "val_loss": val_loss, "total_samples": total_samples}


def save_checkpoint(output_dir: str,
                    epoch: int,
                    model,
                    optimizer,
                    is_main: bool,
                    distributed: bool,
                    data_parallel: bool,
                    metrics: Dict[str, Any],
                    save_hf: bool = False,
                    tag: str = ""):
    if not is_main:
        return
    os.makedirs(output_dir, exist_ok=True)
    ckpt_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    base = f"epoch_{epoch:03d}" + (f"_{tag}" if tag else "")
    ckpt_path = os.path.join(ckpt_dir, base + ".pt")

    to_save = unwrap_model(model, distributed, data_parallel)
    state = {
        "epoch": epoch,
        "model_state": to_save.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "metrics": metrics,
    }
    torch.save(state, ckpt_path)

    # Optionally also save HuggingFace-style weights for easy re-load with from_pretrained
    if save_hf:
        hf_dir = os.path.join(ckpt_dir, base + "_hf")
        os.makedirs(hf_dir, exist_ok=True)
        to_save.save_pretrained(hf_dir)

    # Save a small metadata file
    with open(os.path.join(ckpt_dir, base + "_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Train PatchTST on multi-folder windowed timeseries dataset (classification).")
    # Data args
    parser.add_argument("--root-dir", type=str, required=True, help="Root directory containing folders with cluster_results.csv and clustering_data.csv")
    parser.add_argument("--cluster-filename", type=str, default="cluster_results.csv")
    parser.add_argument("--timeseries-filename", type=str, default="clustering_data.csv")
    parser.add_argument("--window-seconds", type=float, default=1.0)
    parser.add_argument("--freq-ms", type=int, default=10)
    parser.add_argument("--channels", type=str, default="nearest", help="Comma-separated: nearest,token")
    parser.add_argument("--restrict-clusters", type=str, default="", help="Comma-separated cluster ids to include")
    parser.add_argument("--ensure-exact-len", action="store_true", default=True)
    parser.add_argument("--nearest-pad-value", type=float, default=None)
    parser.add_argument("--remap-labels", action="store_true", default=True)
    parser.add_argument("--max-pairs", type=int, default=None)

    # Training args
    parser.add_argument("--batch-size", type=int, default=64, help="Per-process batch size")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--drop-last", action="store_true", default=False)
    parser.add_argument("--amp", action="store_true", default=False, help="Use mixed precision for training")
    parser.add_argument("--output-dir", type=str, default="saved_models/patchtst")
    parser.add_argument("--weight-decay", type=float, default=0.01)

    # Eval/checkpoint/early stopping
    parser.add_argument("--eval-interval", type=int, default=1, help="Evaluate every N epochs")
    parser.add_argument("--save-interval", type=int, default=5, help="Save checkpoint every N epochs")
    parser.add_argument("--monitor-metric", type=str, default="val_acc", choices=["val_acc", "val_loss"], help="Metric to monitor for best/early stopping")
    parser.add_argument("--min-delta", type=float, default=0.0, help="Minimum change to qualify as improvement")
    parser.add_argument("--early-stopping", action="store_true", default=False, help="Enable early stopping")
    parser.add_argument("--patience", type=int, default=10, help="Number of eval steps without improvement before stopping")
    parser.add_argument("--save-hf-checkpoints", action="store_true", default=False, help="Also save HuggingFace-style checkpoints at each save")

    # Multi-GPU args
    parser.add_argument("--distributed", action="store_true", default=False, help="Use DistributedDataParallel (launch with torchrun or Slurm)")
    parser.add_argument("--data-parallel", action="store_true", default=False, help="Use DataParallel (single process, multiple GPUs)")

    # Model args
    parser.add_argument("--context-length", type=int, default=None, help="Sequence length; default: inferred from dataset (T)")
    parser.add_argument("--patch-length", type=int, default=10)
    parser.add_argument("--num-hidden-layers", type=int, default=3)
    parser.add_argument("--ffn-dim", type=int, default=512)
    parser.add_argument("--channel-attention", action="store_true", default=False)
    parser.add_argument("--ff-dropout", type=float, default=0.2)
    parser.add_argument("--attention-dropout", type=float, default=0.1)
    parser.add_argument("--positional-dropout", type=float, default=0.05)
    parser.add_argument("--path-dropout", type=float, default=0)


    args = parser.parse_args()

    # Seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    channels = parse_channels(args.channels)
    restrict_clusters = parse_int_list(args.restrict_clusters)

    # Device & distributed setup
    device, distributed, rank, world_size, is_main, local_rank = setup_device_and_distributed(args)

    if is_main:
        print(f"Device: {device}, Distributed: {distributed}, World Size: {world_size}, Rank: {rank}")
        print(f"Channels: {channels}, Restrict clusters: {restrict_clusters}")

    # Build dataset
    dataset = MultiFolderFirstSecondsTimeseriesDataset(
        root_dir=args.root_dir,
        cluster_filename=args.cluster_filename,
        timeseries_filename=args.timeseries_filename,
        window_seconds=args.window_seconds,
        freq_ms=args.freq_ms,
        channels=channels,
        restrict_clusters=restrict_clusters,
        ensure_exact_len=args.ensure_exact_len,
        nearest_pad_value=args.nearest_pad_value,
        remap_labels=args.remap_labels,
        verbose=is_main,
        max_pairs=args.max_pairs,
    )

    # Context length
    context_length = args.context_length if args.context_length is not None else dataset.T
    num_input_channels = len(channels)

    # DataLoaders
    train_loader, test_loader, train_sampler, _ = build_dataloaders(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        test_size=args.test_size,
        seed=args.seed,
        distributed=distributed,
        drop_last=args.drop_last,
    )

    # Model and optimizer
    configuration = PatchTSTConfig(
        context_length=context_length,
        patch_length=args.patch_length,
        num_input_channels=num_input_channels,
        num_hidden_layers=args.num_hidden_layers,
        ffn_dim=args.ffn_dim,
        channel_attention=args.channel_attention,
        num_targets=dataset.num_clusters,
        attention_dropout = args.attention_dropout,
        positional_dropout = args.positional_dropout,
        path_dropout = args.path_dropout,
        ff_dropout = args.ff_dropout,
    )
    model = PatchTSTForClassification(configuration).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Multi-GPU wrapping
    data_parallel = False
    if distributed:
        model = DDP(model, device_ids=[local_rank] if device.type == "cuda" else None, output_device=local_rank if device.type == "cuda" else None)
    elif args.data_parallel and torch.cuda.device_count() > 1:
        data_parallel = True
        model = DataParallel(model)

    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")

    # Training loop with periodic eval & checkpointing
    best_metric = None
    epochs_no_improve = 0
    history = []

    for epoch in range(args.epochs):
        if distributed and train_sampler is not None:
            train_sampler.set_epoch(epoch)

        iterator = train_loader
        if is_main:
            iterator = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")

        running_meter = AverageMeter()
        model.train()

        for x_batch, y_batch in iterator:
            x_batch = x_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            if args.amp and device.type == "cuda":
                with torch.cuda.amp.autocast():
                    outputs = model(past_values=x_batch, target_values=y_batch)
                    loss = outputs.loss
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(past_values=x_batch, target_values=y_batch)
                loss = outputs.loss
                loss.backward()
                optimizer.step()

            running_meter.update(loss.item(), n=x_batch.shape[0])

        train_loss = running_meter.avg
        metrics = {"epoch": epoch + 1, "train_loss": train_loss}

        # Periodic evaluation
        do_eval = ((epoch + 1) % max(1, args.eval_interval) == 0)
        if do_eval:
            val_metrics = evaluate(model, test_loader, device, distributed=distributed, amp_eval=False)
            metrics.update(val_metrics)

            if is_main:
                print(f"Epoch {epoch+1}: train_loss={train_loss:.4f} | val_loss={val_metrics['val_loss']:.4f} | val_acc={val_metrics['val_acc']:.4f}")

            # Early stopping and best checkpointing
            current = metrics[args.monitor_metric]
            improved = False
            if best_metric is None:
                improved = True
            else:
                if args.monitor_metric == "val_acc":
                    improved = (current > best_metric + args.min_delta)
                else:  # val_loss
                    improved = (current < best_metric - args.min_delta)

            if improved:
                best_metric = current
                epochs_no_improve = 0
                save_checkpoint(
                    output_dir=args.output_dir,
                    epoch=epoch + 1,
                    model=model,
                    optimizer=optimizer,
                    is_main=is_main,
                    distributed=distributed,
                    data_parallel=data_parallel,
                    metrics=metrics,
                    save_hf=True,  # save best model in HF format too
                    tag="best",
                )
            else:
                epochs_no_improve += 1
                if args.early_stopping and epochs_no_improve >= args.patience:
                    if is_main:
                        print(f"Early stopping at epoch {epoch+1}: no improvement in {args.patience} eval steps.")
                    break

        # Periodic checkpoint saving (regardless of improvement)
        if (epoch + 1) % max(1, args.save_interval) == 0:
            save_checkpoint(
                output_dir=args.output_dir,
                epoch=epoch + 1,
                model=model,
                optimizer=optimizer,
                is_main=is_main,
                distributed=distributed,
                data_parallel=data_parallel,
                metrics=metrics,
                save_hf=args.save_hf_checkpoints,
                tag="ckpt",
            )

        history.append(metrics)

        if distributed:
            torch.distributed.barrier()

    # Final evaluation (if last epoch didn't run eval)
    if not ((history and "val_acc" in history[-1]) or (history and "val_loss" in history[-1])):
        final_val = evaluate(model, test_loader, device, distributed=distributed, amp_eval=False)
        if is_main:
            print(f"Final Eval: val_loss={final_val['val_loss']:.4f} | val_acc={final_val['val_acc']:.4f}")
        # Save final checkpoint
        save_checkpoint(
            output_dir=args.output_dir,
            epoch=len(history),
            model=model,
            optimizer=optimizer,
            is_main=is_main,
            distributed=distributed,
            data_parallel=data_parallel,
            metrics={"epoch": len(history), **final_val},
            save_hf=False,
            tag="final",
        )

    # Save summary on main rank
    if is_main:
        os.makedirs(args.output_dir, exist_ok=True)
        summary = {
            "epochs_run": len(history),
            "best_metric": best_metric,
            "monitor_metric": args.monitor_metric,
            "train_history": history,
            "num_classes": dataset.num_clusters,
            "context_length": context_length,
            "channels": list(channels),
            "output_dir": args.output_dir,
        }
        with open(os.path.join(args.output_dir, "train_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

    if distributed:
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()