"""Build and run one correctly paired IEMOCAP LOSO CA-HB2V-MoE case.

The outer test session, inner validation session, and fine-tuned Stage-1
checkpoint are resolved from the same fold/case. Test data is only evaluated
after the best validation-UA checkpoint has been selected.

Example::

    python CA-HB2V-MoE/train_loso.py --fold loso_Ses02 --val-session Ses05 --dry-run
    python CA-HB2V-MoE/train_loso.py --fold loso_Ses02 --val-session Ses05
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
SESSIONS = {f"Ses{i:02d}" for i in range(1, 6)}


def select_stage1_checkpoint(case_dir: Path) -> Path:
    """Pick the best existing checkpoint from this exact LOSO case only."""
    best = case_dir / "best_stage1.pt"
    if best.is_file():
        return best.resolve()

    metadata_path = case_dir / "stage1_topk_checkpoints.json"
    if metadata_path.is_file():
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        if isinstance(metadata, dict):
            metadata = metadata.get("checkpoints", metadata.get("items", []))
        for entry in sorted(
            metadata,
            key=lambda item: (
                int(item.get("rank", 10**9)),
                -float(item.get("selection_score", item.get("ua", -1.0))),
            ),
        ):
            checkpoint = Path(entry.get("checkpoint", ""))
            if not checkpoint.is_absolute():
                checkpoint = case_dir / checkpoint
            if checkpoint.is_file():
                return checkpoint.resolve()

    candidates = sorted((case_dir / "stage1_topk_checkpoints").glob("stage1_epoch*.pt"))
    if candidates:
        return candidates[-1].resolve()
    raise FileNotFoundError(
        f"No fine-tuned Stage-1 checkpoint exists for {case_dir}. "
        "Refusing to silently use the generic WavLM initialization."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", required=True, choices=[f"loso_Ses{i:02d}" for i in range(1, 6)])
    parser.add_argument("--val-session", required=True, choices=sorted(SESSIONS))
    parser.add_argument("--manifest-root", type=Path, default=ROOT / "manifests" / "loso")
    parser.add_argument(
        "--stage1-root",
        type=Path,
        default=ROOT / "outputs" / "1024" / "stage1_no_aug",
        help=(
            "Stage-1 fine-tuned checkpoint root. Pass either the supplied "
            "stage1_no_aug directory or its nested loso directory."
        ),
    )
    parser.add_argument(
        "--stage1-checkpoint",
        "--wavlm-checkpoint",
        dest="stage1_checkpoint",
        type=Path,
        default=None,
        help=(
            "Use this exact fine-tuned Stage-1 .pt file. When supplied, it "
            "takes precedence over --stage1-root checkpoint discovery."
        ),
    )
    parser.add_argument(
        "--wavlm-name",
        type=Path,
        default=ROOT.parent / "pretrained_model" / "wavlm-large",
        help="Base WavLM-large directory used to construct the architecture before loading the fold checkpoint.",
    )
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs" / "CA-HB2V-MoE" / "loso")
    parser.add_argument(
        "--variant",
        default="ca_hb2v_moe",
        choices=("wavlm_only", "wavlm_fullband_mfcc", "standard_hierarchical_moe", "ca_hb2v_moe"),
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if __import__("torch").cuda.is_available() else "cpu")
    parser.add_argument(
        "--freeze-backbone",
        action="store_true",
        help=(
            "Freeze every WavLM parameter, including feature_projection. "
            "By default the historical policy freezes only CNN and Transformer layers."
        ),
    )
    parser.add_argument("--dry-run", action="store_true", help="Write the resolved config and print the command without training.")
    args = parser.parse_args()

    if args.val_session == args.fold.removeprefix("loso_"):
        parser.error("The validation session must differ from the outer held-out test session.")

    # Accept the user-facing Stage-1 directory directly:
    #   outputs/1024/stage1_no_aug
    # while retaining compatibility with the already-nested form:
    #   outputs/1024/stage1_no_aug/loso
    case_name = f"val_session_{args.val_session}"
    manifest_case = args.manifest_root / args.fold / "cases" / case_name
    for split in ("train", "val", "test"):
        manifest = manifest_case / f"{split}.csv"
        if not manifest.is_file():
            raise FileNotFoundError(f"Missing {split} manifest: {manifest}")
    if args.stage1_checkpoint is not None:
        checkpoint = args.stage1_checkpoint.expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Fine-tuned Stage-1 checkpoint does not exist: {checkpoint}")
    else:
        stage1_root = args.stage1_root.expanduser().resolve()
        nested_loso_root = stage1_root / "loso"
        if not (stage1_root / args.fold).is_dir() and nested_loso_root.is_dir():
            stage1_root = nested_loso_root
        stage1_case = stage1_root / args.fold / "cases" / case_name
        checkpoint = select_stage1_checkpoint(stage1_case)
    output_dir = args.output_root / args.variant / args.fold / case_name
    output_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "seed": 42,
        "dataset": {
            "train_manifest": str((manifest_case / "train.csv").resolve()),
            "val_manifest": str((manifest_case / "val.csv").resolve()),
            "test_manifest": str((manifest_case / "test.csv").resolve()),
            "sample_rate": 16000,
            "max_duration_seconds": 6.0,
            "audio_path_column": "audio_path",
            "id_column": "sample_id",
            "label_column": "label",
            "normalize_waveform": True,
            "num_workers": args.num_workers,
            "pin_memory": True,
            "persistent_workers": True,
            # Match the Stage-1 class balancing policy.  The per-sample
            # sampler is only used for the training split; val/test remain
            # in their original order and distribution.
            "use_weighted_sampler": True,
            "sample_weighting": "inverse_freq",
            "augmentation": {"enable": False},
        },
        "model": {
            "variant": args.variant,
            "num_classes": 4,
            "wavlm_name": str(args.wavlm_name.resolve()),
            "wavlm_checkpoint": str(checkpoint),
            "d_model": 256,
            "mfcc_hidden_dim": 128,
            "n_mels": 20,
            "n_mfcc": 13,
            "sample_rate": 16000,
            "n_fft": 400,
            "hop_length": 160,
            "dropout": 0.2,
            "freeze_cnn": True,
            "freeze_transformer": True,
            "freeze_backbone": bool(args.freeze_backbone),
            "load_balance_weight": 0.01,
        },
        "trainer": {
            "output_dir": str(output_dir.resolve()),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": 1e-4,
            "backbone_lr": 1e-5,
            "min_lr": 1e-7,
            "weight_decay": 1e-5,
            "warmup_ratio": 0.1,
            "mixed_precision": "fp16",
            "selection_metric": "ua",
            "class_weighting": "inverse_sqrt_freq",
            "label_smoothing": 0.05,
            "grad_clip": 1.0,
            "progress_bar": True,
            "log_filename": "train.log",
            # Two micro-batches give a steadier update on 16 GB GPUs while
            # retaining the same effective batch size as Stage-1.
            "gradient_accumulation_steps": 2,
            "early_stop_patience": 10,
            "min_delta": 1e-4,
        },
    }
    config_path = output_dir / "resolved_config.yaml"
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)

    command = [sys.executable, str(Path(__file__).with_name("train.py")), "--config", str(config_path)]
    print(f"fold: {args.fold}")
    print(f"validation session: {args.val_session}")
    print(f"test session: {args.fold.removeprefix('loso_')}")
    print(f"fine-tuned Stage-1 checkpoint: {checkpoint}")
    print(f"resolved config: {config_path}")
    if args.dry_run:
        print("command:", " ".join(command))
    else:
        subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
