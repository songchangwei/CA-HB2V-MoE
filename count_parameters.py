"""Count total/trainable parameters for all CA-HB2V-MoE ablations.

This uses the local WavLM-large ``config.json`` and does not load the 1.2 GB
base checkpoint or any optimizer state. Counts therefore match the model
architecture while remaining cheap to reproduce.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

from transformers import WavLMConfig

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ablations import build_ablation_model, parameter_report  # noqa: E402


VARIANTS = (
    "wavlm_only",
    "wavlm_fullband_mfcc",
    "standard_hierarchical_moe",
    "ca_hb2v_moe",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--wavlm-name",
        type=Path,
        default=ROOT.parent / "pretrained_model" / "wavlm-large",
        help="Local WavLM-large directory containing config.json.",
    )
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--num-classes", type=int, default=4)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    wavlm_config = WavLMConfig.from_pretrained(str(args.wavlm_name))
    reports = {}
    for variant in VARIANTS:
        model = build_ablation_model(
            variant,
            num_classes=args.num_classes,
            load_pretrained=False,
            wavlm_config=wavlm_config,
            d_model=args.d_model,
            dropout=args.dropout,
        )
        reports[variant] = parameter_report(model)
        del model
        gc.collect()

    print(json.dumps(reports, indent=2))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(reports, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
