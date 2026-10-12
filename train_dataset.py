"""Dataset-specific CA-HB2V-MoE launcher for SAVEE and RAVDESS.

The launcher resolves the repository's speaker-independent manifests, writes a
fully explicit config for ``train.py`` (including UA/WA top-5 evaluation), and
then runs one case or all available train/validation/test cases.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]

DATASETS = {
    "savee": {
        "manifest_root": ROOT / "manifests" / "savee" / "speaker_independent",
        # Existing SAVEE Stage-1 checkpoints are under outputs/savee_stage1_no_aug.
        "default_stage1_root": ROOT / "outputs" / "savee_stage1_no_aug" / "speaker_independent",
        "default_output_root": ROOT / "outputs" / "CA-HB2V-MoE" / "savee" / "speaker_independent",
        "num_classes": 7,
        "fold_glob": "fold_*_test_*",
        "case_glob": "cases/val_*",
    },
    "ravdess": {
        "manifest_root": ROOT / "manifests" / "ravdess" / "ltspo",
        "default_stage1_root": ROOT / "outputs" / "1024" / "ravdess_stage1_no_aug" / "ltspo",
        "default_output_root": ROOT / "outputs" / "CA-HB2V-MoE" / "ravdess" / "ltspo",
        "num_classes": 8,
        "fold_glob": "fold_*_test_*",
        "case_glob": "cases/val_*",
    },
}


def _cases(spec: dict) -> list[tuple[str, str, Path]]:
    result = []
    for fold_dir in sorted(spec["manifest_root"].glob(spec["fold_glob"])):
        for case_dir in sorted(fold_dir.glob(spec["case_glob"])):
            if all((case_dir / f"{split}.csv").is_file() for split in ("train", "val", "test")):
                result.append((fold_dir.name, case_dir.name, case_dir))
    return result


def _find_checkpoint(root: Path, fold: str, case: str) -> Path | None:
    candidates = [
        root / fold / case / "best_stage1.pt",
        root / fold / "cases" / case / "best_stage1.pt",
        root / fold / case / "best.pt",
        root / fold / "cases" / case / "best.pt",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    case_dirs = [root / fold / "cases" / case, root / fold / case]
    # Older Stage-1 runs retained only top-k files and a metadata table.
    for case_dir in case_dirs:
        metadata = case_dir / "stage1_topk_checkpoints.json"
        if metadata.is_file():
            try:
                payload = __import__("json").loads(metadata.read_text(encoding="utf-8"))
                entries = payload.get("checkpoints", payload.get("items", payload)) if isinstance(payload, dict) else payload
                if isinstance(entries, list):
                    entries = sorted(entries, key=lambda e: (int(e.get("rank", 10**9)), -float(e.get("selection_score", e.get("ua", -1.0)))))
                    for entry in entries:
                        value = entry.get("checkpoint") or entry.get("path")
                        if value:
                            path = Path(value)
                            if not path.is_absolute():
                                path = case_dir / path
                            if path.is_file():
                                return path.resolve()
            except (OSError, ValueError, TypeError):
                pass
        topk = sorted((case_dir / "stage1_topk_checkpoints").glob("stage1_epoch*.pt"))
        if topk:
            return topk[-1].resolve()
    return None


def _resolve_checkpoint(args, spec: dict, fold: str, case: str) -> Path | None:
    if args.stage1_checkpoint is not None:
        checkpoint = args.stage1_checkpoint.expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Stage-1 checkpoint does not exist: {checkpoint}")
        return checkpoint
    root = (args.stage1_root or spec["default_stage1_root"]).expanduser().resolve()
    checkpoint = _find_checkpoint(root, fold, case)
    if checkpoint is None and root.name != "cases":
        # Also accept a root that already points at the dataset's stage1 tree
        # or at a generic nested ``loso``-style directory.
        checkpoint = _find_checkpoint(root / "cases", fold, case)
    return checkpoint


def _build_config(args, spec: dict, fold: str, case: str, case_dir: Path, output_dir: Path, checkpoint: Path | None) -> dict:
    model = {
        "variant": args.variant,
        "num_classes": spec["num_classes"],
        "wavlm_name": str(args.wavlm_name.expanduser().resolve()),
        "d_model": 256,
        "mfcc_hidden_dim": 128,
        "n_mels": 20,
        "n_mfcc": 13,
        "sample_rate": 16000,
        "n_fft": 400,
        "hop_length": 160,
        "dropout": args.dropout,
        "freeze_cnn": True,
        "freeze_transformer": True,
        "freeze_backbone": args.freeze_backbone,
        "load_balance_weight": 0.01,
    }
    if checkpoint is not None:
        model["wavlm_checkpoint"] = str(checkpoint)
    return {
        "seed": args.seed,
        "dataset": {
            "train_manifest": str((case_dir / "train.csv").resolve()),
            "val_manifest": str((case_dir / "val.csv").resolve()),
            "test_manifest": str((case_dir / "test.csv").resolve()),
            "sample_rate": 16000,
            "max_duration_seconds": args.max_duration,
            "audio_path_column": "audio_path",
            "id_column": "sample_id",
            "label_column": "label",
            "normalize_waveform": True,
            "num_workers": args.num_workers,
            "pin_memory": True,
            "persistent_workers": args.num_workers > 0,
            "use_weighted_sampler": True,
            "sample_weighting": "inverse_freq",
            "augmentation": {"enable": False},
        },
        "model": model,
        "trainer": {
            "output_dir": str(output_dir.resolve()),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "backbone_lr": args.backbone_lr,
            "min_lr": 1e-7,
            "weight_decay": 1e-5,
            "warmup_ratio": 0.1,
            "mixed_precision": args.mixed_precision,
            "selection_metric": "ua",
            "class_weighting": "inverse_sqrt_freq",
            "label_smoothing": 0.05,
            "grad_clip": 1.0,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "early_stop_patience": args.early_stop_patience,
            "progress_bar": not args.no_progress,
            "log_filename": "train.log",
        },
    }


def _run_case(args, spec: dict, fold: str, case: str, case_dir: Path) -> None:
    checkpoint = _resolve_checkpoint(args, spec, fold, case)
    if checkpoint is None and not args.allow_base_wavlm:
        default_root = (args.stage1_root or spec["default_stage1_root"]).expanduser().resolve()
        raise FileNotFoundError(
            f"No Stage-1 checkpoint found for {fold}/{case}. "
            f"Searched {default_root}. Pass --stage1-checkpoint, --stage1-root, "
            "or explicitly use --allow-base-wavlm."
        )
    output_root = (args.output_root or spec["default_output_root"]).expanduser().resolve()
    output_dir = output_root / args.variant / fold / case
    output_dir.mkdir(parents=True, exist_ok=True)
    config = _build_config(args, spec, fold, case, case_dir, output_dir, checkpoint)
    config_path = output_dir / "resolved_config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(f"dataset: {args.dataset}")
    print(f"fold: {fold}; case: {case}")
    print(f"Stage-1 checkpoint: {checkpoint or 'base WavLM (explicitly allowed)'}")
    print(f"resolved config: {config_path}")
    if args.dry_run:
        return
    subprocess.run([sys.executable, str(Path(__file__).with_name("train.py")), "--config", str(config_path)], cwd=ROOT, check=True)


def main(dataset: str | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train CA-HB2V-MoE on SAVEE or RAVDESS manifests.")
    parser.add_argument("--dataset", choices=sorted(DATASETS), default=dataset)
    parser.add_argument("--seed", type=int, default=42, help="Random seed written to the resolved trainer config.")
    parser.add_argument("--fold", default=None, help="Manifest fold directory, e.g. fold_01_test_DC")
    parser.add_argument("--case", dest="case_name", default=None, help="Case directory, e.g. val_JE or val_05-08")
    parser.add_argument("--all", action="store_true", help="Run every available fold/case")
    parser.add_argument("--manifest-root", type=Path, default=None)
    parser.add_argument("--stage1-root", type=Path, default=None)
    parser.add_argument("--stage1-checkpoint", "--wavlm-checkpoint", dest="stage1_checkpoint", type=Path, default=None)
    parser.add_argument("--wavlm-name", type=Path, default=ROOT.parent / "pretrained_model" / "wavlm-large")
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--variant", choices=("wavlm_only", "wavlm_fullband_mfcc", "standard_hierarchical_moe", "ca_hb2v_moe"), default="ca_hb2v_moe")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--early-stop-patience", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--backbone-lr", type=float, default=1e-5)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--max-duration", type=float, default=6.0)
    parser.add_argument("--mixed-precision", choices=("no", "fp16", "bf16"), default="fp16")
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--allow-base-wavlm", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.dataset is None:
        parser.error("--dataset is required when using train_dataset.py directly")
    spec = dict(DATASETS[args.dataset])
    if args.manifest_root is not None:
        spec["manifest_root"] = args.manifest_root.expanduser().resolve()
    available = _cases(spec)
    if not available:
        raise FileNotFoundError(f"No train/val/test cases found below {spec['manifest_root']}")
    selected = available
    if not args.all:
        if not args.fold or not args.case_name:
            parser.error("provide --fold and --case, or use --all")
        selected = [(fold, case, path) for fold, case, path in available if fold == args.fold and case == args.case_name]
        if not selected:
            choices = ", ".join(f"{fold}/{case}" for fold, case, _ in available)
            raise ValueError(f"Unknown fold/case {args.fold}/{args.case_name}; available: {choices}")
    if args.stage1_checkpoint is not None and args.all:
        print("using the explicitly supplied Stage-1 checkpoint for every selected case")
    for fold, case, case_dir in selected:
        _run_case(args, spec, fold, case, case_dir)


if __name__ == "__main__":
    main()
