"""Ablation models for CA-HB2V-MoE.

The four supported variants are:

``wavlm_only``
    WavLM utterance representation and a classifier.
``wavlm_fullband_mfcc``
    WavLM plus one full-band 39-D MFCC stream (static, delta and
    delta-delta), followed by a learned two-view fusion projection.
``standard_hierarchical_moe``
    The same three physical MFCC bands and shared causal encoder as the full
    model, but both routers are standard: Band-MoE sees only band embeddings
    and View-MoE sees only the two view embeddings and their product.
``ca_hb2v_moe``
    The complete condition-aware model from ``model.py``.

All variants use the requested WavLM policy by default: CNN feature extractor
and Transformer encoder are frozen; feature projection and the new modules
remain trainable.
"""

from __future__ import annotations

import math
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lcmr_tff_two_stage.models.backbone import WavLMBackbone  # noqa: E402
from model import (  # noqa: E402
    CAHB2VMoE,
    CAHB2VMoEOutput,
    EPS,
    MLP,
    SharedCausalMFCCEncoder,
    _masked_mean,
)


@dataclass
class AblationOutput:
    """Common output contract consumed by the training loop."""

    logits: torch.Tensor
    probabilities: torch.Tensor
    load_balance_loss: torch.Tensor
    wavlm_embedding: torch.Tensor
    acoustic_embedding: torch.Tensor
    band_gates: Optional[torch.Tensor] = None
    view_gates: Optional[torch.Tensor] = None


class _WavLMViewBase(nn.Module):
    """Shared WavLM construction, freezing and checkpoint loading."""

    def __init__(
        self,
        num_classes: int,
        wavlm_name: str = "microsoft/wavlm-large",
        wavlm_checkpoint: Optional[str | Path] = None,
        d_model: int = 256,
        dropout: float = 0.2,
        load_pretrained: bool = True,
        wavlm_config=None,
        freeze_cnn: bool = True,
        freeze_transformer: bool = True,
        freeze_backbone: bool = False,
        load_balance_weight: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.d_model = int(d_model)
        # Keep one common trainer interface across all ablations.  The
        # single-view baselines do not use a routing regularizer, while the
        # standard hierarchical variant may opt into it.
        self.load_balance_weight = float(load_balance_weight)
        self.freeze_backbone_during_stage2 = bool(freeze_backbone)
        self.backbone = WavLMBackbone(
            model_name_or_path=wavlm_name,
            load_pretrained=load_pretrained,
            wavlm_config=wavlm_config,
        )
        if self.freeze_backbone_during_stage2:
            self.backbone.freeze()
        elif freeze_cnn and freeze_transformer:
            self.backbone.configure_stage1_finetune_mode("freeze_cnn_and_transformer")
        elif freeze_cnn:
            self.backbone.configure_stage1_finetune_mode("freeze_cnn_only")
        elif freeze_transformer:
            for parameter in self.backbone.wavlm.encoder.parameters():
                parameter.requires_grad = False
            self.backbone.keep_encoder_eval = True
            self.backbone.wavlm.encoder.eval()

        self.wavlm_projection = nn.Sequential(
            nn.Linear(self.backbone.hidden_size, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
        )
        if wavlm_checkpoint is not None:
            self.load_wavlm_checkpoint(wavlm_checkpoint)

    def load_wavlm_checkpoint(self, checkpoint_path: str | Path, strict: bool = False):
        checkpoint_path = Path(checkpoint_path)
        try:
            payload = torch.load(checkpoint_path, map_location="cpu", mmap=True, weights_only=True)
        except (TypeError, RuntimeError, pickle.UnpicklingError):
            payload = torch.load(checkpoint_path, map_location="cpu", mmap=True, weights_only=False)
        state_dict = payload.get("model", payload.get("state_dict", payload))
        wavlm_state = self.backbone._extract_wavlm_state_dict(state_dict)
        result = self.backbone.wavlm.load_state_dict(wavlm_state, strict=strict)
        del wavlm_state, state_dict, payload
        if self.freeze_backbone_during_stage2:
            self.backbone.freeze()
        else:
            self.backbone.configure_stage1_finetune_mode("freeze_cnn_and_transformer")
        return result

    def _wavlm_view(self, waveform: torch.Tensor, attention_mask: torch.Tensor):
        backbone_output = self.backbone(
            input_values=waveform,
            attention_mask=attention_mask.to(torch.long),
            output_hidden_states=False,
        )
        hidden = backbone_output.last_hidden_state
        token_mask = self.backbone.wavlm._get_feature_vector_attention_mask(
            hidden.size(1), attention_mask.to(torch.long)
        )
        pooled = _masked_mean(hidden, token_mask)
        return self.wavlm_projection(pooled), backbone_output.feature_lengths


class WavLMOnlyBaseline(_WavLMViewBase):
    """Single-view WavLM baseline."""

    def __init__(
        self,
        *args,
        dropout: float = 0.2,
        # These options belong to acoustic variants.  Accepting them here
        # keeps a single YAML config usable for every ablation.
        sample_rate=16000,
        n_fft=400,
        hop_length=160,
        n_mels=20,
        n_mfcc=13,
        mfcc_hidden_dim=128,
        load_balance_weight=0.0,
        **kwargs,
    ):
        del sample_rate, n_fft, hop_length, n_mels, n_mfcc, mfcc_hidden_dim
        super().__init__(*args, dropout=dropout, load_balance_weight=load_balance_weight, **kwargs)
        self.classifier = nn.Linear(self.d_model, self.num_classes)

    def forward(self, waveform, attention_mask=None, waveform_lengths=None):
        del waveform_lengths
        if attention_mask is None:
            attention_mask = torch.ones_like(waveform, dtype=torch.long)
        wavlm, _ = self._wavlm_view(waveform, attention_mask)
        logits = self.classifier(wavlm)
        return AblationOutput(
            logits=logits,
            probabilities=F.softmax(logits, dim=-1),
            load_balance_loss=logits.new_zeros(()),
            wavlm_embedding=wavlm,
            acoustic_embedding=logits.new_zeros(logits.size(0), self.d_model),
        )


class FullBandMFCCExtractor(nn.Module):
    """Full-band 13 MFCC + delta + delta-delta extractor."""

    def __init__(self, sample_rate=16000, n_fft=400, hop_length=160, n_mels=40, n_mfcc=13):
        super().__init__()
        self.sample_rate = int(sample_rate)
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.n_mels = int(n_mels)
        self.n_mfcc = int(n_mfcc)
        self.register_buffer("window", torch.hann_window(self.n_fft), persistent=False)
        self.register_buffer("mel_filterbank", self._build_filterbank(), persistent=False)
        self.register_buffer("dct_matrix", self._build_dct(), persistent=False)

    @staticmethod
    def _hz_to_mel(hz):
        return torch.where(
            hz < 1000.0,
            hz * (3.0 / 200.0),
            15.0 + 27.0 * torch.log1p((hz - 1000.0) / 700.0),
        )

    @staticmethod
    def _mel_to_hz(mel):
        return torch.where(
            mel < 15.0,
            mel * (200.0 / 3.0),
            700.0 * (torch.exp((mel - 15.0) / 27.0) - 1.0) + 1000.0,
        )

    def _build_filterbank(self):
        frequencies = torch.fft.rfftfreq(self.n_fft, d=1.0 / self.sample_rate)
        edges = self._mel_to_hz(
            torch.linspace(
                self._hz_to_mel(torch.tensor(50.0)),
                self._hz_to_mel(torch.tensor(7600.0)),
                self.n_mels + 2,
            )
        )
        bank = torch.zeros(self.n_mels, frequencies.numel())
        for index in range(self.n_mels):
            left, center, right = edges[index : index + 3]
            up = ((frequencies - left) / (center - left).clamp_min(EPS)).clamp(0.0, 1.0)
            down = ((right - frequencies) / (right - center).clamp_min(EPS)).clamp(0.0, 1.0)
            bank[index] = torch.minimum(up, down) * (2.0 / (right - left).clamp_min(EPS))
        return bank

    def _build_dct(self):
        mel = torch.arange(self.n_mels, dtype=torch.float32).unsqueeze(0)
        coefficient = torch.arange(self.n_mfcc, dtype=torch.float32).unsqueeze(1)
        matrix = torch.cos(math.pi / self.n_mels * (mel + 0.5) * coefficient)
        matrix[0] *= math.sqrt(1.0 / self.n_mels)
        matrix[1:] *= math.sqrt(2.0 / self.n_mels)
        return matrix

    def forward(self, waveform: torch.Tensor, lengths: torch.Tensor):
        with torch.autocast(device_type=waveform.device.type, enabled=False):
            waveform = waveform.float()
            if waveform.size(1) < self.n_fft:
                waveform = F.pad(waveform, (0, self.n_fft - waveform.size(1)))
            spectrum = torch.stft(
                waveform,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.n_fft,
                window=self.window.to(waveform),
                center=False,
                return_complex=True,
            )
            power = spectrum.abs().square().transpose(1, 2)
            mel = torch.einsum("bfk,mk->bfm", power, self.mel_filterbank.to(power)).clamp_min(EPS)
            static = torch.einsum("bfm,cm->bfc", mel.log(), self.dct_matrix.to(mel))
            frame_lengths = ((lengths - self.n_fft).clamp_min(0) // self.hop_length + 1).clamp_min(1)
            frame_mask = torch.arange(static.size(1), device=lengths.device)[None] < frame_lengths[:, None]
            valid = frame_mask.to(static.dtype).unsqueeze(-1)
            count = valid.sum(dim=1).clamp_min(1.0)
            mean = (static * valid).sum(dim=1) / count
            std = (((static - mean[:, None]).square() * valid).sum(dim=1) / count + EPS).sqrt()
            static = ((static - mean[:, None]) / std[:, None]) * valid
            delta = torch.zeros_like(static)
            delta[:, 1:] = static[:, 1:] - static[:, :-1]
            delta = delta * valid
            delta_delta = torch.zeros_like(static)
            delta_delta[:, 1:] = delta[:, 1:] - delta[:, :-1]
            delta_delta = delta_delta * valid
            features = torch.cat([static, delta, delta_delta], dim=-1)
        return features, frame_mask


class WavLMFullBandMFCCBaseline(_WavLMViewBase):
    """WavLM + full-band 39-D MFCC baseline without band routing."""

    def __init__(
        self,
        *args,
        sample_rate=16000,
        n_fft=400,
        hop_length=160,
        n_mels=40,
        n_mfcc=13,
        mfcc_hidden_dim=128,
        dropout=0.2,
        load_balance_weight=0.0,
        **kwargs,
    ):
        # The full-band definition is fixed at 40 mel filters and 13 DCT
        # coefficients; retain the arguments for config compatibility.
        del n_mels, n_mfcc, mfcc_hidden_dim
        super().__init__(*args, dropout=dropout, load_balance_weight=load_balance_weight, **kwargs)
        self.fullband_mfcc = FullBandMFCCExtractor(sample_rate, n_fft, hop_length)
        self.acoustic_encoder = SharedCausalMFCCEncoder(
            input_dim=39, hidden_dim=128, output_dim=self.d_model, dropout=dropout
        )
        self.view_fusion = MLP(2 * self.d_model, self.d_model, dropout=dropout)
        self.classifier = nn.Linear(self.d_model, self.num_classes)

    def forward(self, waveform, attention_mask=None, waveform_lengths=None):
        if waveform_lengths is None:
            waveform_lengths = attention_mask.sum(dim=1) if attention_mask is not None else torch.full(
                (waveform.size(0),), waveform.size(1), device=waveform.device, dtype=torch.long
            )
        if attention_mask is None:
            attention_mask = torch.arange(waveform.size(1), device=waveform.device)[None] < waveform_lengths[:, None]
        wavlm, _ = self._wavlm_view(waveform, attention_mask)
        features, frame_mask = self.fullband_mfcc(waveform, waveform_lengths)
        acoustic = self.acoustic_encoder(features, frame_mask)
        fused = self.view_fusion(torch.cat([wavlm, acoustic], dim=-1))
        logits = self.classifier(fused)
        return AblationOutput(
            logits=logits,
            probabilities=F.softmax(logits, dim=-1),
            load_balance_loss=logits.new_zeros(()),
            wavlm_embedding=wavlm,
            acoustic_embedding=acoustic,
        )


class StandardBandMoE(nn.Module):
    """Band-MoE whose router does not receive condition descriptors."""

    def __init__(self, d_model, dropout=0.2):
        super().__init__()
        self.router = MLP(3 * d_model, 4, hidden_dim=max(128, d_model), dropout=dropout)
        self.low_expert = MLP(d_model, d_model, dropout=dropout)
        self.mid_expert = MLP(d_model, d_model, dropout=dropout)
        self.high_expert = MLP(d_model, d_model, dropout=dropout)
        self.cross_expert = MLP(3 * d_model, d_model, dropout=dropout)

    def forward(self, bands):
        gates = F.softmax(self.router(bands.flatten(1)), dim=-1)
        experts = torch.stack(
            [
                self.low_expert(bands[:, 0]),
                self.mid_expert(bands[:, 1]),
                self.high_expert(bands[:, 2]),
                self.cross_expert(bands.flatten(1)),
            ], dim=1,
        )
        return (experts * gates.unsqueeze(-1)).sum(dim=1), gates


class StandardViewMoE(nn.Module):
    """View-MoE whose router omits q and first-level alpha."""

    def __init__(self, d_model, dropout=0.2):
        super().__init__()
        self.router = MLP(3 * d_model, 3, hidden_dim=max(128, d_model), dropout=dropout)
        self.wavlm_expert = MLP(d_model, d_model, dropout=dropout)
        self.acoustic_expert = MLP(d_model, d_model, dropout=dropout)
        self.joint_expert = MLP(2 * d_model, d_model, dropout=dropout)

    def forward(self, wavlm, acoustic):
        gates = F.softmax(self.router(torch.cat([wavlm, acoustic, wavlm * acoustic], dim=-1)), dim=-1)
        experts = torch.stack(
            [
                self.wavlm_expert(wavlm),
                self.acoustic_expert(acoustic),
                self.joint_expert(torch.cat([wavlm, acoustic], dim=-1)),
            ], dim=1,
        )
        return (experts * gates.unsqueeze(-1)).sum(dim=1), gates


class StandardHierarchicalMoE(_WavLMViewBase):
    """Standard Band-MoE + Standard View-MoE ablation."""

    def __init__(
        self,
        *args,
        sample_rate=16000,
        n_fft=400,
        hop_length=160,
        n_mels=20,
        n_mfcc=13,
        mfcc_hidden_dim=128,
        dropout=0.2,
        load_balance_weight=0.0,
        **kwargs,
    ):
        super().__init__(*args, dropout=dropout, load_balance_weight=load_balance_weight, **kwargs)
        # Use the same feature definition as the complete model. Conditions
        # are computed by the extractor for a shared preprocessing path, but
        # are deliberately not passed to either standard router.
        self.sample_rate = int(sample_rate)
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.n_mels = int(n_mels)
        self.n_mfcc = int(n_mfcc)
        self.window = torch.hann_window(self.n_fft)
        self.register_buffer("mel_filterbanks", self._build_band_filterbanks(), persistent=False)
        self.register_buffer("dct_matrix", self._build_dct_matrix(), persistent=False)
        self.mfcc_encoder = SharedCausalMFCCEncoder(
            input_dim=n_mfcc, hidden_dim=mfcc_hidden_dim, output_dim=self.d_model, dropout=dropout
        )
        self.band_moe = StandardBandMoE(self.d_model, dropout=dropout)
        self.view_moe = StandardViewMoE(self.d_model, dropout=dropout)
        self.classifier = nn.Linear(self.d_model, self.num_classes)

    def _build_band_filterbanks(self):
        frequencies = torch.fft.rfftfreq(self.n_fft, d=1.0 / self.sample_rate)
        banks = []
        for low, high in ((50.0, 1000.0), (1000.0, 3000.0), (3000.0, 7600.0)):
            mel_edges = torch.linspace(
                self._hz_to_mel(torch.tensor(low)), self._hz_to_mel(torch.tensor(high)), self.n_mels + 2
            )
            edges = self._mel_to_hz(mel_edges)
            bank = torch.zeros(self.n_mels, frequencies.numel())
            for index in range(self.n_mels):
                left, center, right = edges[index : index + 3]
                up = ((frequencies - left) / (center - left).clamp_min(EPS)).clamp(0.0, 1.0)
                down = ((right - frequencies) / (right - center).clamp_min(EPS)).clamp(0.0, 1.0)
                bank[index] = torch.minimum(up, down) * (2.0 / (right - left).clamp_min(EPS))
            banks.append(bank)
        return torch.stack(banks, dim=0)

    @staticmethod
    def _hz_to_mel(hz):
        return torch.where(hz < 1000.0, hz * (3.0 / 200.0), 15.0 + 27.0 * torch.log1p((hz - 1000.0) / 700.0))

    @staticmethod
    def _mel_to_hz(mel):
        return torch.where(mel < 15.0, mel * (200.0 / 3.0), 700.0 * (torch.exp((mel - 15.0) / 27.0) - 1.0) + 1000.0)

    def _build_dct_matrix(self):
        mel = torch.arange(self.n_mels, dtype=torch.float32).unsqueeze(0)
        coefficient = torch.arange(self.n_mfcc, dtype=torch.float32).unsqueeze(1)
        matrix = torch.cos(math.pi / self.n_mels * (mel + 0.5) * coefficient)
        matrix[0] *= math.sqrt(1.0 / self.n_mels)
        matrix[1:] *= math.sqrt(2.0 / self.n_mels)
        return matrix

    def _mfcc_bands(self, waveform, lengths):
        with torch.autocast(device_type=waveform.device.type, enabled=False):
            waveform = waveform.float()
            if waveform.size(1) < self.n_fft:
                waveform = F.pad(waveform, (0, self.n_fft - waveform.size(1)))
            spectrum = torch.stft(
                waveform, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.n_fft,
                window=self.window.to(waveform), center=False, return_complex=True,
            )
            power = spectrum.abs().square().transpose(1, 2)
            frame_lengths = ((lengths - self.n_fft).clamp_min(0) // self.hop_length + 1).clamp_min(1)
            frame_mask = torch.arange(power.size(1), device=lengths.device)[None] < frame_lengths[:, None]
            valid = frame_mask.to(power.dtype)
            count = valid.sum(dim=1).clamp_min(1.0)
            bands = []
            for bank in self.mel_filterbanks:
                mel = torch.einsum("bfk,mk->bfm", power, bank.to(power)).clamp_min(EPS)
                coefficients = torch.einsum("bfm,cm->bfc", mel.log(), self.dct_matrix.to(mel))
                v = valid.unsqueeze(-1)
                mean = (coefficients * v).sum(dim=1) / count.unsqueeze(-1)
                std = (((coefficients - mean[:, None]).square() * v).sum(dim=1) / count.unsqueeze(-1) + EPS).sqrt()
                bands.append(((coefficients - mean[:, None]) / std[:, None]) * v)
            return torch.stack(bands, dim=1), frame_mask

    @staticmethod
    def _load_balance_loss(*gates):
        penalties = []
        for gate in gates:
            mean_gate = gate.mean(dim=0)
            penalties.append(gate.size(-1) * mean_gate.square().sum())
        return torch.stack(penalties).mean() - 1.0

    def forward(self, waveform, attention_mask=None, waveform_lengths=None):
        if waveform_lengths is None:
            waveform_lengths = attention_mask.sum(dim=1) if attention_mask is not None else torch.full(
                (waveform.size(0),), waveform.size(1), device=waveform.device, dtype=torch.long
            )
        if attention_mask is None:
            attention_mask = torch.arange(waveform.size(1), device=waveform.device)[None] < waveform_lengths[:, None]
        wavlm, _ = self._wavlm_view(waveform, attention_mask)
        mfcc, frame_mask = self._mfcc_bands(waveform, waveform_lengths)
        bands = torch.stack([self.mfcc_encoder(mfcc[:, index], frame_mask) for index in range(3)], dim=1)
        acoustic, band_gates = self.band_moe(bands)
        fused, view_gates = self.view_moe(wavlm, acoustic)
        logits = self.classifier(fused)
        return AblationOutput(
            logits=logits,
            probabilities=F.softmax(logits, dim=-1),
            load_balance_loss=self._load_balance_loss(band_gates, view_gates),
            wavlm_embedding=wavlm,
            acoustic_embedding=acoustic,
            band_gates=band_gates,
            view_gates=view_gates,
        )


VARIANT_ALIASES = {
    "wavlm_only": WavLMOnlyBaseline,
    "wavlm-only": WavLMOnlyBaseline,
    "fullband_mfcc": WavLMFullBandMFCCBaseline,
    "wavlm_fullband_mfcc": WavLMFullBandMFCCBaseline,
    "standard_hierarchical_moe": StandardHierarchicalMoE,
    "standard_moe": StandardHierarchicalMoE,
    "ca_hb2v_moe": CAHB2VMoE,
    "full": CAHB2VMoE,
}


def build_ablation_model(variant: str, **kwargs):
    key = str(variant).strip().lower()
    if key not in VARIANT_ALIASES:
        raise ValueError(f"Unknown CA-HB2V-MoE variant '{variant}'. Supported: {sorted(VARIANT_ALIASES)}")
    return VARIANT_ALIASES[key](**kwargs)


def parameter_report(model: nn.Module) -> dict[str, int | float]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    frozen = total - trainable
    return {
        "total_parameters": int(total),
        "trainable_parameters": int(trainable),
        "frozen_parameters": int(frozen),
        "trainable_percent": 100.0 * trainable / max(total, 1),
    }


__all__ = [
    "AblationOutput",
    "WavLMOnlyBaseline",
    "WavLMFullBandMFCCBaseline",
    "StandardHierarchicalMoE",
    "build_ablation_model",
    "parameter_report",
]
