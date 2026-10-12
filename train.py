"""Minimal manifest-driven trainer for CA-HB2V-MoE.

Example::

    python CA-HB2V-MoE/train.py --config CA-HB2V-MoE/config.yaml

The trainer intentionally uses the repository's ``EmotionAudioDataset`` and
``AudioBatchCollator`` so manifests and audio preprocessing are consistent
with the other training entrypoints in this project.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from model import CAHB2VMoE, cahb2v_loss  # noqa: E402
from ablations import build_ablation_model, parameter_report  # noqa: E402
from lcmr_tff_two_stage.data import AudioBatchCollator, EmotionAudioDataset  # noqa: E402


def _read_config(path: str | Path) -> dict:
    import yaml

    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class _TqdmLoggingHandler(logging.Handler):
    """Write log records above active tqdm bars without corrupting them."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            tqdm.write(self.format(record))
        except Exception:
            self.handleError(record)


def _create_logger(output_dir: Path, log_filename: str = "train.log") -> logging.Logger:
    """Create a per-run logger that writes to both console and a UTF-8 file."""
    logger = logging.getLogger("ca_hb2v_moe")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")
    console_handler = _TqdmLoggingHandler()
    console_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(output_dir / log_filename, mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger


def _loader(
    dataset_config: dict,
    split: str,
    label_to_id: dict[str, int] | None,
    batch_size: int,
    evaluation: bool = False,
) -> tuple[EmotionAudioDataset, DataLoader]:
    dataset = EmotionAudioDataset(
        manifest_path=dataset_config[f"{split}_manifest"],
        sample_rate=int(dataset_config.get("sample_rate", 16000)),
        max_duration_seconds=float(dataset_config.get("max_duration_seconds", 6.0)),
        audio_root=dataset_config.get("audio_root"),
        audio_path_column=dataset_config.get("audio_path_column", "audio_path"),
        id_column=dataset_config.get("id_column", "sample_id"),
        label_column=dataset_config.get("label_column", "label"),
        label_to_id=label_to_id,
        normalize_waveform=bool(dataset_config.get("normalize_waveform", True)),
        crop_strategy="head" if evaluation or split != "train" else "random",
        augmentation_config=dataset_config.get("augmentation") if not evaluation and split == "train" else None,
    )
    workers = int(dataset_config.get("num_workers", 0))
    sampler = None
    if split == "train" and not evaluation and bool(dataset_config.get("use_weighted_sampler", False)):
        # Stage-1 uses the same inverse-frequency sampler.  This is useful
        # for LOSO folds where the sad/angry classes are consistently smaller
        # than neutral/happy.  Keep validation/test loaders untouched.
        sampler = WeightedRandomSampler(
            weights=dataset.get_sample_weights(
                mode=str(dataset_config.get("sample_weighting", "inverse_freq"))
            ),
            num_samples=len(dataset),
            replacement=True,
        )
    loader_options = {
        "dataset": dataset,
        "batch_size": int(batch_size),
        "shuffle": split == "train" and not evaluation and sampler is None,
        "sampler": sampler,
        "num_workers": workers,
        "pin_memory": bool(dataset_config.get("pin_memory", True)),
        "collate_fn": AudioBatchCollator(),
    }
    if workers > 0:
        loader_options["persistent_workers"] = bool(dataset_config.get("persistent_workers", True))
    return dataset, DataLoader(
        **loader_options,
    )


def _scheduler(optimizer: AdamW, total_steps: int, warmup_steps: int, min_lr_ratio: float) -> LambdaLR:
    def rate(step: int) -> float:
        if step < warmup_steps:
            return max(float(step + 1) / max(1, warmup_steps), 1e-8)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + np.cos(np.pi * min(max(progress, 0.0), 1.0)))
        return float(min_lr_ratio + (1.0 - min_lr_ratio) * cosine)

    return LambdaLR(optimizer, rate)


def _run_epoch(
    model,
    loader,
    device,
    num_classes: int,
    optimizer=None,
    scaler=None,
    amp_dtype=None,
    lb_weight=1e-2,
    scheduler=None,
    class_weights=None,
    label_smoothing=0.0,
    grad_clip=1.0,
    gradient_accumulation_steps=1,
    progress_desc: str | None = None,
    progress_enabled: bool = True,
):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long, device=device)
    total_count = 0
    accumulation_steps = max(1, int(gradient_accumulation_steps))
    if training:
        optimizer.zero_grad(set_to_none=True)
    progress = tqdm(
        loader,
        desc=progress_desc,
        leave=False,
        dynamic_ncols=True,
        disable=not progress_enabled,
    )
    for batch_index, batch in enumerate(progress):
        labels = batch.get("labels")
        if labels is None:
            raise ValueError("Training/evaluation manifest must contain a label column.")
        waveform = batch["input_values"].to(device, non_blocking=True)
        mask = batch["attention_mask"].to(device, non_blocking=True)
        lengths = batch["waveform_lengths"].to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        autocast_enabled = device.type == "cuda" and amp_dtype is not None
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=autocast_enabled):
            output = model(waveform, attention_mask=mask, waveform_lengths=lengths)
            loss = cahb2v_loss(
                output,
                labels,
                lb_weight,
                class_weights=class_weights,
                label_smoothing=label_smoothing,
            )
        if not torch.isfinite(loss):
            raise FloatingPointError(
                "Non-finite CA-HB2V-MoE loss. Check waveform normalization, AMP settings, "
                "and the matched WavLM checkpoint."
            )
        if training:
            # Divide by the actual size of this accumulation group.  The
            # final group can contain fewer micro-batches when the epoch
            # length is not divisible by ``accumulation_steps``.
            group_size = min(accumulation_steps, len(loader) - batch_index)
            backward_loss = loss / group_size
            if scaler is not None:
                scaler.scale(backward_loss).backward()
            else:
                backward_loss.backward()
            is_update = ((batch_index + 1) % accumulation_steps == 0) or ((batch_index + 1) == len(loader))
            if is_update:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                if grad_clip is not None and float(grad_clip) > 0:
                    torch.nn.utils.clip_grad_norm_(
                        (p for p in model.parameters() if p.requires_grad), float(grad_clip)
                    )
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if scheduler is not None:
                    scheduler.step()
        total_loss += float(loss.detach()) * labels.numel()
        predictions = output.logits.argmax(dim=-1)
        confusion += torch.bincount(
            labels * num_classes + predictions,
            minlength=num_classes * num_classes,
        ).reshape(num_classes, num_classes)
        total_count += labels.numel()
        progress.set_postfix(loss=f"{float(loss.detach()):.4f}", refresh=False)
    support = confusion.sum(dim=1)
    present_classes = support > 0
    recall = confusion.diag().float() / support.clamp_min(1).float()
    ua = recall[present_classes].mean().item() if present_classes.any() else 0.0
    wa = confusion.diag().sum().float().div(max(int(confusion.sum()), 1)).item()
    return total_loss / max(total_count, 1), ua, wa


def _compute_metrics(labels: list[int], predictions: list[int], num_classes: int) -> dict[str, Any]:
    """Compute the two paper metrics and the complete confusion summary.

    WA is the sample-weighted accuracy and UA is the unweighted mean recall
    over classes present in the split.  Keeping the confusion matrix and
    per-class recalls makes each retained top-5 result self-contained.
    """
    if not labels:
        return {
            "num_samples": 0,
            "ua": 0.0,
            "wa": 0.0,
            "confusion_matrix": [],
            "class_recalls": [],
        }
    target = torch.tensor(labels, dtype=torch.long)
    predicted = torch.tensor(predictions, dtype=torch.long)
    confusion = torch.bincount(
        target * int(num_classes) + predicted,
        minlength=int(num_classes) * int(num_classes),
    ).reshape(int(num_classes), int(num_classes))
    support = confusion.sum(dim=1)
    recalls = confusion.diag().float() / support.clamp_min(1).float()
    present = support > 0
    return {
        "num_samples": len(labels),
        "ua": float(recalls[present].mean()) if present.any() else 0.0,
        "wa": float((target == predicted).float().mean()),
        "confusion_matrix": confusion.tolist(),
        "class_recalls": [float(value) if bool(support[index]) else None for index, value in enumerate(recalls)],
        "class_support": [int(value) for value in support.tolist()],
    }


def _summarize_routes(gates: list[list[float]] | None, num_experts: int) -> dict[str, Any] | None:
    """Return aggregate routing statistics while retaining per-sample gates separately."""
    if not gates:
        return None
    values = np.asarray(gates, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != int(num_experts):
        return None
    values = np.clip(values, 1e-12, 1.0)
    choices = values.argmax(axis=1)
    counts = np.bincount(choices, minlength=int(num_experts))
    return {
        "num_samples": int(values.shape[0]),
        "mean_gates": values.mean(axis=0).tolist(),
        "std_gates": values.std(axis=0).tolist(),
        "mean_entropy": float(-(values * np.log(values)).sum(axis=1).mean()),
        "argmax_counts": counts.astype(int).tolist(),
        "argmax_frequencies": (counts / max(values.shape[0], 1)).tolist(),
    }


@torch.inference_mode()
def _evaluate_split(
    model,
    loader,
    device,
    num_classes: int,
    lb_weight: float,
    class_weights: torch.Tensor | None,
    label_smoothing: float,
    amp_dtype,
    progress_desc: str,
    progress_enabled: bool,
) -> dict[str, Any]:
    """Evaluate a split and retain predictions plus Band/View routing state.

    The loaders used here are deterministic (head crop, no augmentation), so
    rerunning the five selected checkpoints gives reproducible train/val/test
    artifacts.  Gate vectors are kept per sample and summarized by expert.
    """
    model.eval()
    records: list[dict[str, Any]] = []
    labels_all: list[int] = []
    predictions_all: list[int] = []
    total_loss = 0.0
    total_count = 0
    band_gates: list[list[float]] = []
    view_gates: list[list[float]] = []
    progress = tqdm(
        loader,
        desc=progress_desc,
        leave=False,
        dynamic_ncols=True,
        disable=not progress_enabled,
    )
    for batch in progress:
        waveform = batch["input_values"].to(device, non_blocking=True)
        mask = batch["attention_mask"].to(device, non_blocking=True)
        lengths = batch["waveform_lengths"].to(device, non_blocking=True)
        labels = batch.get("labels")
        labels_device = labels.to(device, non_blocking=True) if labels is not None else None
        autocast_enabled = device.type == "cuda" and amp_dtype is not None
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=autocast_enabled):
            output = model(waveform, attention_mask=mask, waveform_lengths=lengths)
            loss = None
            if labels_device is not None:
                loss = cahb2v_loss(
                    output,
                    labels_device,
                    lb_weight,
                    class_weights=class_weights,
                    label_smoothing=label_smoothing,
                )
        probabilities = output.probabilities.detach().float().cpu()
        predictions = probabilities.argmax(dim=-1).tolist()
        batch_labels = labels.tolist() if labels is not None else [None] * len(predictions)
        raw_band = getattr(output, "band_gates", None)
        raw_view = getattr(output, "view_gates", None)
        batch_band = raw_band.detach().float().cpu().tolist() if raw_band is not None else [None] * len(predictions)
        batch_view = raw_view.detach().float().cpu().tolist() if raw_view is not None else [None] * len(predictions)
        for index, prediction in enumerate(predictions):
            label = batch_labels[index]
            band = batch_band[index]
            view = batch_view[index]
            sample_id = batch.get("sample_ids", [str(len(records) + index)])[index]
            record = {
                "sample_id": str(sample_id),
                "label": int(label) if label is not None else None,
                "prediction": int(prediction),
                "probabilities": probabilities[index].tolist(),
                "band_gates": band,
                "view_gates": view,
            }
            records.append(record)
            if label is not None:
                labels_all.append(int(label))
                predictions_all.append(int(prediction))
            if band is not None:
                band_gates.append(band)
            if view is not None:
                view_gates.append(view)
        if loss is not None:
            total_loss += float(loss.detach()) * len(predictions)
            total_count += len(predictions)
        progress.set_postfix(loss=f"{float(loss.detach()):.4f}" if loss is not None else "n/a", refresh=False)
    metrics = _compute_metrics(labels_all, predictions_all, num_classes)
    metrics["loss"] = total_loss / max(total_count, 1) if total_count else None
    metrics["records"] = records
    metrics["band_routing"] = _summarize_routes(band_gates, 4)
    metrics["view_routing"] = _summarize_routes(view_gates, 3)
    return metrics


def _write_split_results(path: Path, metrics: dict[str, Any]) -> None:
    """Write complete metrics and per-sample prediction/routing records."""
    path.mkdir(parents=True, exist_ok=True)
    summary = {key: value for key, value in metrics.items() if key != "records"}
    (path / "metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (path / "routing.json").write_text(
        json.dumps(
            {
                "band_moe": metrics.get("band_routing"),
                "view_moe": metrics.get("view_routing"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    records = metrics.get("records", [])
    with (path / "records.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    # A compact CSV is useful for spreadsheet inspection; vectors stay in the
    # JSONL file to avoid lossy string flattening.
    if records:
        with (path / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
            fields = ["sample_id", "label", "prediction", "probabilities", "band_gates", "view_gates"]
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for record in records:
                writer.writerow({
                    "sample_id": record.get("sample_id"),
                    "label": record.get("label"),
                    "prediction": record.get("prediction"),
                    "probabilities": json.dumps(record.get("probabilities", []), ensure_ascii=False),
                    "band_gates": json.dumps(record.get("band_gates"), ensure_ascii=False),
                    "view_gates": json.dumps(record.get("view_gates"), ensure_ascii=False),
                })


def _optimizer_param_groups(model, lr: float, backbone_lr: float, weight_decay: float):
    """Use a smaller LR for the already task-adapted WavLM projection."""
    groups: dict[tuple[float, bool], list[torch.nn.Parameter]] = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        parameter_lr = backbone_lr if name.startswith("backbone.wavlm.feature_projection.") else lr
        no_decay = parameter.ndim <= 1 or name.endswith(".bias")
        groups.setdefault((parameter_lr, no_decay), []).append(parameter)
    return [
        {
            "params": params,
            "lr": group_lr,
            "weight_decay": 0.0 if no_decay else weight_decay,
        }
        for (group_lr, no_decay), params in groups.items()
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="YAML configuration")
    parser.add_argument("--log-file", default=None, help="Log filename (default: <output_dir>/train.log)")
    args = parser.parse_args()
    config = _read_config(args.config)
    _set_seed(int(config.get("seed", 42)))
    device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    dataset_config = config.get("dataset", config)
    model_config = config.get("model", {})
    trainer_config = config.get("trainer", {})
    batch_size = int(trainer_config.get("batch_size", 16))
    train_set, train_loader = _loader(dataset_config, "train", None, batch_size)
    val_set, val_loader = _loader(dataset_config, "val", train_set.label_to_id, batch_size)
    test_loader = None
    if dataset_config.get("test_manifest"):
        _, test_loader = _loader(dataset_config, "test", train_set.label_to_id, batch_size)
    num_classes = int(model_config.get("num_classes", len(train_set.label_to_id)))
    model_kwargs = {k: v for k, v in model_config.items() if k not in {"num_classes", "variant"}}
    variant = str(model_config.get("variant", "ca_hb2v_moe")).strip().lower()
    model = build_ablation_model(variant, num_classes=num_classes, **model_kwargs).to(device)
    parameter_summary = parameter_report(model)
    output_dir = Path(trainer_config.get("output_dir", "outputs/CA-HB2V-MoE"))
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = Path(args.log_file) if args.log_file else output_dir / str(trainer_config.get("log_filename", "train.log"))
    if not log_file.is_absolute():
        log_file = output_dir / log_file
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = _create_logger(output_dir, str(log_file.relative_to(output_dir)) if log_file.is_relative_to(output_dir) else str(log_file))
    progress_enabled = bool(trainer_config.get("progress_bar", True))
    logger.info(
        f"variant={variant} total_parameters={parameter_summary['total_parameters']} "
        f"trainable_parameters={parameter_summary['trainable_parameters']} "
        f"frozen_parameters={parameter_summary['frozen_parameters']}"
    )
    with (output_dir / "parameter_report.json").open("w", encoding="utf-8") as handle:
        json.dump({"variant": variant, **parameter_summary}, handle, indent=2)
    base_lr = float(trainer_config.get("lr", 1e-4))
    optimizer = AdamW(
        _optimizer_param_groups(
            model,
            lr=base_lr,
            backbone_lr=float(trainer_config.get("backbone_lr", base_lr * 0.1)),
            weight_decay=float(trainer_config.get("weight_decay", 1e-5)),
        ),
        lr=base_lr,
        betas=(0.9, 0.98),
        eps=1e-8,
    )
    epochs = int(trainer_config.get("epochs", 50))
    gradient_accumulation_steps = max(1, int(trainer_config.get("gradient_accumulation_steps", 1)))
    updates_per_epoch = max(1, (len(train_loader) + gradient_accumulation_steps - 1) // gradient_accumulation_steps)
    total_steps = max(1, epochs * updates_per_epoch)
    warmup_steps = int(total_steps * float(trainer_config.get("warmup_ratio", 0.1)))
    min_lr = float(trainer_config.get("min_lr", 1e-7))
    scheduler = _scheduler(optimizer, total_steps, warmup_steps, min_lr / float(trainer_config.get("lr", 1e-4)))
    precision = str(trainer_config.get("mixed_precision", "fp16")).strip().lower()
    if precision not in {"no", "fp16", "bf16"}:
        raise ValueError("trainer.mixed_precision must be one of: no, fp16, bf16.")
    amp_dtype = {"no": None, "fp16": torch.float16, "bf16": torch.bfloat16}[precision]
    scaler_enabled = device.type == "cuda" and amp_dtype == torch.float16
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)
    else:  # PyTorch 2.2 compatibility
        scaler = torch.cuda.amp.GradScaler(enabled=scaler_enabled)
    class_weights = train_set.get_class_weights(
        mode=str(trainer_config.get("class_weighting", "inverse_sqrt_freq"))
    )
    if class_weights is not None:
        class_weights = class_weights.to(device)
    label_smoothing = float(trainer_config.get("label_smoothing", 0.05))
    grad_clip = trainer_config.get("grad_clip", 1.0)
    requested_selection_metric = str(trainer_config.get("selection_metric", "ua")).strip().lower()
    if requested_selection_metric not in {"ua", "wa"}:
        raise ValueError("trainer.selection_metric must be 'ua' or 'wa'.")
    # Model selection is intentionally fixed to validation UA.  WA remains a
    # reported metric and is used only as a deterministic tie-breaker for
    # equal-UA epochs, matching the requested top-5 protocol.
    selection_metric = "ua"
    if requested_selection_metric != selection_metric:
        logger.warning("trainer.selection_metric=%s ignored; top-5 selection is fixed to validation UA", requested_selection_metric)
    best_val = -1.0
    patience = int(trainer_config.get("early_stop_patience", 10))
    min_delta = float(trainer_config.get("min_delta", 1e-4))
    epochs_without_improvement = 0
    history = []
    topk_dir = output_dir / "top5_checkpoints"
    topk_dir.mkdir(parents=True, exist_ok=True)
    topk: list[dict[str, Any]] = []
    for epoch in range(1, epochs + 1):
        train_loss, train_ua, train_wa = _run_epoch(
            model,
            train_loader,
            device,
            num_classes,
            optimizer=optimizer,
            scaler=scaler,
            amp_dtype=amp_dtype,
            # The two single-view baselines intentionally have no routing
            # regularizer.  Keep the trainer shared by treating their
            # missing attribute as zero.
            lb_weight=float(getattr(model, "load_balance_weight", 0.0)),
            scheduler=scheduler,
            class_weights=class_weights,
            label_smoothing=label_smoothing,
            grad_clip=grad_clip,
            gradient_accumulation_steps=gradient_accumulation_steps,
            progress_desc=f"Train {epoch}/{epochs}",
            progress_enabled=progress_enabled,
        )
        with torch.no_grad():
            val_loss, val_ua, val_wa = _run_epoch(
                model,
                val_loader,
                device,
                num_classes,
                amp_dtype=amp_dtype,
                lb_weight=float(getattr(model, "load_balance_weight", 0.0)),
                class_weights=class_weights,
                label_smoothing=label_smoothing,
                gradient_accumulation_steps=gradient_accumulation_steps,
                progress_desc=f"Val {epoch}/{epochs}",
                progress_enabled=progress_enabled,
            )
        monitored = val_ua
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_ua": train_ua,
            "train_wa": train_wa,
            "val_loss": val_loss,
            "val_ua": val_ua,
            "val_wa": val_wa,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        # Every epoch gets a self-contained checkpoint.  The final top-5
        # ranking is based strictly on validation UA (WA is retained as a
        # secondary diagnostic and deterministic tie-breaker).
        checkpoint_payload = {
            "model": model.state_dict(),
            "label_to_id": train_set.label_to_id,
            "epoch": epoch,
            "selection_metric": "ua",
            "best_validation_ua": val_ua,
            "best_validation_wa": val_wa,
            "config": config,
            "epoch_metrics": row,
        }
        candidate = {
            "epoch": epoch,
            "val_ua": float(val_ua),
            "val_wa": float(val_wa),
            "checkpoint": str((topk_dir / f"epoch_{epoch:03d}.pt").resolve()),
            "epoch_metrics": row,
        }
        # Avoid serializing a WavLM-large state dictionary for every epoch:
        # after five candidates, write only when the new validation UA can
        # enter the retained set.  The five files below are the complete
        # retained model set used for subsequent train/val/test evaluation.
        ranking_key = lambda item: (item["val_ua"], item["val_wa"], -item["epoch"])
        if len(topk) < 5 or ranking_key(candidate) > ranking_key(topk[-1]):
            torch.save(checkpoint_payload, candidate["checkpoint"])
            topk.append(candidate)
            topk.sort(key=ranking_key, reverse=True)
            for dropped in topk[5:]:
                Path(dropped["checkpoint"]).unlink(missing_ok=True)
            topk = topk[:5]
        logger.info(
            f"epoch={epoch:03d} train_loss={train_loss:.5f} "
            f"train_UA={train_ua:.4f} train_WA={train_wa:.4f} "
            f"val_loss={val_loss:.5f} val_UA={val_ua:.4f} val_WA={val_wa:.4f}"
        )
        if monitored > best_val + min_delta:
            best_val = monitored
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        with (output_dir / "history.json").open("w", encoding="utf-8") as handle:
            json.dump(history, handle, indent=2)
        if patience > 0 and epochs_without_improvement >= patience:
            logger.info(f"early stopping at epoch {epoch}; best val_{selection_metric}={best_val:.4f}")
            break

    # Persist the history before evaluating the selected checkpoints.
    with (output_dir / "history.json").open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)
    for rank, item in enumerate(topk, 1):
        item["rank"] = rank
    if topk:
        # Keep best.pt as a convenient alias for the highest validation-UA
        # model, which is also rank 1 in the top-5 table.
        shutil.copy2(topk[0]["checkpoint"], output_dir / "best.pt")
    with (output_dir / "top5_checkpoints.json").open("w", encoding="utf-8") as handle:
        json.dump(topk, handle, indent=2)
    with (output_dir / "top5_checkpoints.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["rank", "epoch", "val_ua", "val_wa", "checkpoint"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: item.get(field) for field in fields} for item in topk)

    # Rebuild deterministic loaders and evaluate all three splits for each
    # selected model.  This preserves complete per-sample predictions and
    # both Band-MoE/View-MoE gate vectors in top5_rank_*/<split>/records.jsonl.
    train_eval_set, train_eval_loader = _loader(
        dataset_config, "train", train_set.label_to_id, batch_size, evaluation=True
    )
    _, val_eval_loader = _loader(
        dataset_config, "val", train_set.label_to_id, batch_size, evaluation=True
    )
    test_eval_loader = None
    if dataset_config.get("test_manifest"):
        _, test_eval_loader = _loader(
            dataset_config, "test", train_set.label_to_id, batch_size, evaluation=True
        )
    top5_results: list[dict[str, Any]] = []
    for rank, item in enumerate(topk, 1):
        payload = torch.load(item["checkpoint"], map_location=device, weights_only=True)
        model.load_state_dict(payload["model"])
        rank_dir = output_dir / f"top5_rank_{rank}"
        split_results: dict[str, Any] = {}
        split_loaders = [("train", train_eval_loader), ("val", val_eval_loader)]
        if test_eval_loader is not None:
            split_loaders.append(("test", test_eval_loader))
        for split_name, split_loader in split_loaders:
            split_metrics = _evaluate_split(
                model,
                split_loader,
                device,
                num_classes,
                lb_weight=float(getattr(model, "load_balance_weight", 0.0)),
                class_weights=class_weights,
                label_smoothing=label_smoothing,
                amp_dtype=amp_dtype,
                progress_desc=f"{split_name} top-{rank}",
                progress_enabled=progress_enabled,
            )
            _write_split_results(rank_dir / split_name, split_metrics)
            split_results[split_name] = {
                key: value for key, value in split_metrics.items() if key != "records"
            }
            logger.info(
                f"top-{rank} epoch={item['epoch']} {split_name}: "
                f"loss={split_metrics.get('loss')} UA={split_metrics['ua']:.4f} WA={split_metrics['wa']:.4f}"
            )
        top5_results.append(
            {
                "rank": rank,
                "epoch": item["epoch"],
                "checkpoint": item["checkpoint"],
                "selection_val_ua": item["val_ua"],
                "selection_val_wa": item["val_wa"],
                "epoch_metrics": item.get("epoch_metrics", {}),
                "result_dir": str(rank_dir),
                "splits": split_results,
            }
        )
    with (output_dir / "top5_results.json").open("w", encoding="utf-8") as handle:
        json.dump(top5_results, handle, indent=2)
    for split_name in ("train", "val", "test"):
        rows = []
        for result in top5_results:
            split = result["splits"].get(split_name)
            if split is None:
                continue
            row = {
                "rank": result["rank"],
                "epoch": result["epoch"],
                "selection_val_ua": result["selection_val_ua"],
                "selection_val_wa": result["selection_val_wa"],
            }
            row.update(split)
            rows.append(row)
        if rows:
            with (output_dir / f"top5_{split_name}_metrics.json").open("w", encoding="utf-8") as handle:
                json.dump(rows, handle, indent=2)
            fields = list(rows[0].keys())
            with (output_dir / f"top5_{split_name}_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
    # Keep the legacy single-best test file populated from top-1 when a test
    # manifest exists, while top5_results.json remains the complete source.
    if top5_results and "test" in top5_results[0]["splits"]:
        test_metrics = dict(top5_results[0]["splits"]["test"])
        # Preserve the legacy test_metrics.json field names alongside the
        # split-generic ``loss``/``ua``/``wa`` keys used by top5_results.
        test_metrics.update(
            {
                "test_loss": test_metrics.get("loss"),
                "test_ua": test_metrics.get("ua"),
                "test_wa": test_metrics.get("wa"),
                "selected_epoch": top5_results[0]["epoch"],
                "selection_metric": "ua",
            }
        )
        with (output_dir / "test_metrics.json").open("w", encoding="utf-8") as handle:
            json.dump(test_metrics, handle, indent=2)
    logger.info(f"saved top-{len(top5_results)} complete evaluations to {output_dir}")

    logger.info(f"run complete; logs saved to {log_file}")


if __name__ == "__main__":
    main()
