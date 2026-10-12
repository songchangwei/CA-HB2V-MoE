"""CA-HB2V-MoE: condition-aware hierarchical band-to-view MoE.

The implementation follows ``combined_all_reviewers_clean.pdf``.  It is
deliberately self contained apart from the repository's ``WavLMBackbone``:
the three MFCC streams, condition statistics, causal encoder, Band-MoE and
View-MoE are all defined here.

The default WavLM policy is the one requested for this implementation:
``feature_extractor`` (CNN) and ``encoder`` (Transformer layers) are frozen,
while ``feature_projection`` and every module defined in this file remain
trainable.  A stage-1 checkpoint can be loaded either by passing its WavLM
directory as ``wavlm_name`` or with ``wavlm_checkpoint``.
"""

from __future__ import annotations

import math
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

# Allow ``python CA-HB2V-MoE/train.py`` from the repository root while still
# reusing the tested WavLM wrapper in the parent package.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from lcmr_tff_two_stage.models.backbone import WavLMBackbone  # noqa: E402


EPS = 1e-8


@dataclass
class CAHB2VMoEOutput:
    """Outputs and routing states exposed by :class:`CAHB2VMoE`."""

    logits: torch.Tensor
    probabilities: torch.Tensor
    wavlm_embedding: torch.Tensor
    band_embeddings: torch.Tensor
    acoustic_embedding: torch.Tensor
    final_embedding: torch.Tensor
    band_gates: torch.Tensor
    view_gates: torch.Tensor
    band_conditions: torch.Tensor
    load_balance_loss: torch.Tensor
    feature_lengths: torch.Tensor


def _masked_mean(x: torch.Tensor, mask: torch.Tensor, dim: int = 1) -> torch.Tensor:
    weights = mask.to(x.dtype)
    while weights.ndim < x.ndim:
        weights = weights.unsqueeze(-1)
    # Remove the reduced axis after summing so [B,F,C] becomes [B,C] and
    # [B,F] becomes [B].  This keeps the denominator aligned for every
    # feature width and batch size.
    denom = weights.sum(dim=dim, keepdim=True).clamp_min(1.0).squeeze(dim)
    return (x * weights).sum(dim=dim) / denom


def _masked_std(x: torch.Tensor, mask: torch.Tensor, dim: int = 1) -> torch.Tensor:
    mean = _masked_mean(x, mask, dim=dim)
    weights = mask.to(x.dtype)
    while weights.ndim < x.ndim:
        weights = weights.unsqueeze(-1)
    denom = weights.sum(dim=dim, keepdim=True).clamp_min(1.0).squeeze(dim)
    centered = x - mean.unsqueeze(dim)
    return ((centered.square() * weights).sum(dim=dim) / denom + EPS).sqrt()


class MLP(nn.Module):
    """Small projection used by routers and experts."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: Optional[int] = None, dropout: float = 0.1):
        super().__init__()
        hidden_dim = int(hidden_dim or max(output_dim, input_dim // 2))
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CausalDilatedBlock(nn.Module):
    """One left-padded causal Conv1d + LN + GELU residual block."""

    def __init__(self, channels: int, dilation: int, kernel_size: int = 3):
        super().__init__()
        self.dilation = int(dilation)
        self.kernel_size = int(kernel_size)
        self.conv = nn.Conv1d(channels, channels, kernel_size, dilation=self.dilation)
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor, frame_mask: torch.Tensor) -> torch.Tensor:
        # x is [B, F, C].  Explicit left padding prevents any look-ahead.
        y = x.transpose(1, 2)
        left = (self.kernel_size - 1) * self.dilation
        y = self.conv(F.pad(y, (left, 0))).transpose(1, 2)
        y = F.gelu(self.norm(y))
        y = (x + y) * frame_mask.unsqueeze(-1).to(y.dtype)
        return y


class SharedCausalMFCCEncoder(nn.Module):
    """Shared four-block encoder for all three MFCC bands."""

    def __init__(self, input_dim: int = 13, hidden_dim: int = 128, output_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [CausalDilatedBlock(hidden_dim, dilation=d) for d in (1, 2, 4, 8)]
        )
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim * 2, output_dim),
            nn.GELU(),
            nn.LayerNorm(output_dim),
            nn.Dropout(dropout),
        )

    def forward(self, sequence: torch.Tensor, frame_mask: torch.Tensor) -> torch.Tensor:
        hidden = self.input_projection(sequence)
        hidden = hidden * frame_mask.unsqueeze(-1).to(hidden.dtype)
        for block in self.blocks:
            hidden = block(hidden, frame_mask)
        pooled = torch.cat([_masked_mean(hidden, frame_mask), _masked_std(hidden, frame_mask)], dim=-1)
        return self.output_projection(pooled)


class ConditionAwareBandMoE(nn.Module):
    """Dense four-expert Band-MoE."""

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.router = MLP(3 * d_model + 15, 4, hidden_dim=max(128, d_model), dropout=dropout)
        self.low_expert = MLP(d_model, d_model, dropout=dropout)
        self.mid_expert = MLP(d_model, d_model, dropout=dropout)
        self.high_expert = MLP(d_model, d_model, dropout=dropout)
        self.cross_expert = MLP(3 * d_model, d_model, dropout=dropout)

    def forward(self, bands: torch.Tensor, conditions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        context = torch.cat([bands.flatten(1), conditions.flatten(1)], dim=-1)
        gates = F.softmax(self.router(context), dim=-1)
        experts = torch.stack(
            [
                self.low_expert(bands[:, 0]),
                self.mid_expert(bands[:, 1]),
                self.high_expert(bands[:, 2]),
                self.cross_expert(bands.flatten(1)),
            ],
            dim=1,
        )
        fused = (experts * gates.unsqueeze(-1)).sum(dim=1)
        return fused, gates, experts


class ConditionAwareViewMoE(nn.Module):
    """Dense three-expert View-MoE with first-level routing state."""

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.router = MLP(3 * d_model + 19, 3, hidden_dim=max(128, d_model), dropout=dropout)
        self.wavlm_expert = MLP(d_model, d_model, dropout=dropout)
        self.acoustic_expert = MLP(d_model, d_model, dropout=dropout)
        self.joint_expert = MLP(2 * d_model, d_model, dropout=dropout)

    def forward(
        self, wavlm: torch.Tensor, acoustic: torch.Tensor, context: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        route_context = torch.cat([wavlm, acoustic, wavlm * acoustic, context], dim=-1)
        gates = F.softmax(self.router(route_context), dim=-1)
        experts = torch.stack(
            [self.wavlm_expert(wavlm), self.acoustic_expert(acoustic), self.joint_expert(torch.cat([wavlm, acoustic], dim=-1))],
            dim=1,
        )
        fused = (experts * gates.unsqueeze(-1)).sum(dim=1)
        return fused, gates, experts


class CAHB2VMoE(nn.Module):
    """Condition-Aware Hierarchical Band-to-View Mixture of Experts.

    Parameters are intentionally explicit so this class can be used from a
    training script or imported in another project.  ``wavlm_checkpoint`` may
    be a stage-1 project checkpoint (``model``/``state_dict`` wrapper is
    accepted) or a Hugging Face WavLM directory can be supplied directly as
    ``wavlm_name``.
    """

    BAND_RANGES = ((50.0, 1000.0), (1000.0, 3000.0), (3000.0, 7600.0))

    def __init__(
        self,
        num_classes: int,
        wavlm_name: str = "microsoft/wavlm-large",
        wavlm_checkpoint: Optional[str | Path] = None,
        d_model: int = 256,
        mfcc_hidden_dim: int = 128,
        n_mels: int = 20,
        n_mfcc: int = 13,
        sample_rate: int = 16000,
        n_fft: int = 400,
        hop_length: int = 160,
        dropout: float = 0.1,
        load_pretrained: bool = True,
        wavlm_config=None,
        freeze_cnn: bool = True,
        freeze_transformer: bool = True,
        freeze_backbone: bool = False,
        load_balance_weight: float = 1e-2,
    ) -> None:
        super().__init__()
        if n_mels < n_mfcc:
            raise ValueError("n_mels must be >= n_mfcc")
        self.sample_rate = int(sample_rate)
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.n_mels = int(n_mels)
        self.n_mfcc = int(n_mfcc)
        self.d_model = int(d_model)
        self.load_balance_weight = float(load_balance_weight)
        # ``freeze_cnn``/``freeze_transformer`` retain the historical policy
        # (the feature projection remains trainable).  ``freeze_backbone``
        # opts into freezing every WavLM parameter, including that projection.
        # Keep this state so loading a Stage-1 checkpoint cannot accidentally
        # re-enable gradients on a frozen backbone.
        self.freeze_backbone_during_stage2 = bool(freeze_backbone)
        self._freeze_cnn_requested = bool(freeze_cnn)
        self._freeze_transformer_requested = bool(freeze_transformer)
        self.window = torch.hann_window(self.n_fft)
        self.register_buffer("mel_filterbanks", self._build_band_filterbanks(), persistent=False)
        self.register_buffer("dct_matrix", self._build_dct_matrix(), persistent=False)

        self.backbone = WavLMBackbone(
            model_name_or_path=wavlm_name,
            load_pretrained=load_pretrained,
            wavlm_config=wavlm_config,
        )
        self._apply_backbone_freeze_policy()

        self.wavlm_projection = nn.Sequential(
            nn.Linear(self.backbone.hidden_size, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
        )
        self.mfcc_encoder = SharedCausalMFCCEncoder(
            input_dim=n_mfcc,
            hidden_dim=mfcc_hidden_dim,
            output_dim=d_model,
            dropout=dropout,
        )
        self.band_moe = ConditionAwareBandMoE(d_model, dropout=dropout)
        self.view_moe = ConditionAwareViewMoE(d_model, dropout=dropout)
        self.classifier = nn.Linear(d_model, int(num_classes))

        if wavlm_checkpoint is not None:
            self.load_wavlm_checkpoint(wavlm_checkpoint)

    def _apply_backbone_freeze_policy(
        self,
        freeze_cnn: Optional[bool] = None,
        freeze_transformer: Optional[bool] = None,
    ) -> None:
        """Apply the configured WavLM freezing policy.

        The legacy CNN/Transformer policy leaves ``feature_projection``
        trainable.  ``freeze_backbone=True`` takes precedence and freezes the
        complete WavLM module, which is useful when the Stage-1 checkpoint is
        already task-adapted and this stage should train only the CA modules.
        """
        if freeze_cnn is None:
            freeze_cnn = self._freeze_cnn_requested
        if freeze_transformer is None:
            freeze_transformer = self._freeze_transformer_requested
        if self.freeze_backbone_during_stage2:
            self.backbone.freeze()
            return
        if freeze_cnn and freeze_transformer:
            self.backbone.configure_stage1_finetune_mode("freeze_cnn_and_transformer")
        elif freeze_cnn:
            self.backbone.configure_stage1_finetune_mode("freeze_cnn_only")
        elif freeze_transformer:
            # The repository wrapper intentionally has no inverse mode; set
            # the encoder explicitly for this less common configuration.
            for p in self.backbone.wavlm.encoder.parameters():
                p.requires_grad = False
            self.backbone.keep_encoder_eval = True
            self.backbone.wavlm.encoder.eval()

    @staticmethod
    def _hz_to_mel(hz: torch.Tensor) -> torch.Tensor:
        # Slaney's piecewise Mel scale, matching common filterbank code.
        f_sp = 200.0 / 3.0
        mels = (hz / f_sp).clamp_min(0.0)
        log_region = hz >= 1000.0
        mels = torch.where(log_region, 15.0 + 27.0 * torch.log(1.0 + (hz - 1000.0) / 700.0), mels)
        return mels

    @staticmethod
    def _mel_to_hz(mel: torch.Tensor) -> torch.Tensor:
        f_sp = 200.0 / 3.0
        return torch.where(mel < 15.0, mel * f_sp, 700.0 * (torch.exp((mel - 15.0) / 27.0) - 1.0) + 1000.0)

    def _build_band_filterbanks(self) -> torch.Tensor:
        freqs = torch.fft.rfftfreq(self.n_fft, d=1.0 / self.sample_rate)
        banks = []
        for low, high in self.BAND_RANGES:
            mel_edges = torch.linspace(
                self._hz_to_mel(torch.tensor(low)),
                self._hz_to_mel(torch.tensor(high)),
                self.n_mels + 2,
            )
            edges = self._mel_to_hz(mel_edges)
            bank = torch.zeros(self.n_mels, freqs.numel())
            for i in range(self.n_mels):
                left, center, right = edges[i : i + 3]
                up = ((freqs - left) / (center - left).clamp_min(EPS)).clamp(0.0, 1.0)
                down = ((right - freqs) / (right - center).clamp_min(EPS)).clamp(0.0, 1.0)
                # Slaney-normalized triangular filters (librosa's default
                # ``norm='slaney'``): scale each triangle by its area in Hz.
                bank[i] = torch.minimum(up, down) * (2.0 / (right - left).clamp_min(EPS))
            banks.append(bank)
        return torch.stack(banks, dim=0)

    def _build_dct_matrix(self) -> torch.Tensor:
        k = torch.arange(self.n_mels, dtype=torch.float32).unsqueeze(0)
        n = torch.arange(self.n_mfcc, dtype=torch.float32).unsqueeze(1)
        matrix = torch.cos(math.pi / self.n_mels * (k + 0.5) * n)
        matrix[0] *= math.sqrt(1.0 / self.n_mels)
        if self.n_mfcc > 1:
            matrix[1:] *= math.sqrt(2.0 / self.n_mels)
        return matrix

    def load_wavlm_checkpoint(self, checkpoint_path: str | Path, strict: bool = False):
        """Load WavLM weights from a stage-1 checkpoint and reapply freezing.

        Stage-1 files in this project also contain large AdamW states. Memory
        mapping avoids eagerly copying those unused tensors into host RAM.
        """
        checkpoint_path = Path(checkpoint_path)
        try:
            payload = torch.load(
                checkpoint_path,
                map_location="cpu",
                mmap=True,
                weights_only=True,
            )
        except (TypeError, RuntimeError, pickle.UnpicklingError):
            # Compatibility for older torch builds or checkpoints containing
            # metadata types not accepted by the restricted unpickler.
            try:
                payload = torch.load(
                    checkpoint_path,
                    map_location="cpu",
                    mmap=True,
                    weights_only=False,
                )
            except (TypeError, RuntimeError):
                payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = payload.get("model", payload.get("state_dict", payload))
        wavlm_state = self.backbone._extract_wavlm_state_dict(state_dict)
        result = self.backbone.wavlm.load_state_dict(wavlm_state, strict=strict)
        del wavlm_state, state_dict, payload
        # Loading a checkpoint must preserve the model's freezing policy.
        # In particular, avoid making feature_projection trainable again when
        # ``freeze_backbone=True`` was requested.
        self._apply_backbone_freeze_policy()
        return result

    def _frame_mask(self, lengths: torch.Tensor, num_frames: int) -> torch.Tensor:
        frame_lengths = ((lengths - self.n_fft).clamp_min(0) // self.hop_length + 1).clamp_min(1)
        indices = torch.arange(num_frames, device=lengths.device).unsqueeze(0)
        return indices < frame_lengths.unsqueeze(1)

    def _mfcc_bands(self, waveform: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return [B,3,F,13] CMVN MFCCs and [B,3,5] condition vectors."""
        waveform = waveform.to(torch.float32)
        if waveform.size(1) < self.n_fft:
            waveform = F.pad(waveform, (0, self.n_fft - waveform.size(1)))
        window = self.window.to(device=waveform.device, dtype=waveform.dtype)
        spectrum = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=window,
            center=False,
            return_complex=True,
        )
        power = spectrum.abs().square().transpose(1, 2)  # [B,F,K]
        frame_mask = self._frame_mask(lengths, power.size(1))
        all_mfcc, all_conditions = [], []
        for band in range(3):
            mel = torch.einsum("bfk,mk->bfm", power, self.mel_filterbanks[band].to(power))
            mel = mel.clamp_min(EPS)
            energy = mel.mean(dim=-1)
            valid = frame_mask.to(power.dtype)
            count = valid.sum(dim=1).clamp_min(1.0)
            mean_energy = (energy * valid).sum(dim=1) / count
            std_energy = (((energy - mean_energy[:, None]).square() * valid).sum(dim=1) / count + EPS).sqrt()
            mel_prob = mel / mel.sum(dim=-1, keepdim=True).clamp_min(EPS)
            entropy = -(mel_prob * mel_prob.clamp_min(EPS).log()).sum(dim=-1) / math.log(self.n_mels)
            entropy = (entropy * valid).sum(dim=1) / count
            if mel.size(1) > 1:
                adjacent = frame_mask[:, 1:] & frame_mask[:, :-1]
                delta = energy[:, 1:] - energy[:, :-1]
                dcount = adjacent.to(power.dtype).sum(dim=1).clamp_min(1.0)
                dmean = (delta * adjacent).sum(dim=1) / dcount
                delta_var = (((delta - dmean[:, None]).square() * adjacent).sum(dim=1) / dcount)
            else:
                delta_var = energy.new_zeros(energy.size(0))
            high_ratio = ((energy > (mean_energy + std_energy).unsqueeze(1)) * valid).sum(dim=1) / count
            conditions = torch.stack([mean_energy, std_energy, entropy, delta_var, high_ratio], dim=-1)
            coefficients = torch.einsum("bfm,cm->bfc", mel.log(), self.dct_matrix.to(mel))
            coefficients = coefficients * valid.unsqueeze(-1)
            cmvn_mean = coefficients.sum(dim=1) / count.unsqueeze(-1)
            cmvn_std = (((coefficients - cmvn_mean[:, None]).square() * valid.unsqueeze(-1)).sum(dim=1) / count.unsqueeze(-1) + EPS).sqrt()
            coefficients = ((coefficients - cmvn_mean[:, None]) / cmvn_std[:, None]) * valid.unsqueeze(-1)
            all_mfcc.append(coefficients)
            all_conditions.append(conditions)
        return torch.stack(all_mfcc, dim=1), torch.stack(all_conditions, dim=1)

    @staticmethod
    def _load_balance_loss(*gates: torch.Tensor) -> torch.Tensor:
        losses = []
        for gate in gates:
            mean_gate = gate.mean(dim=0)
            losses.append(gate.size(-1) * mean_gate.square().sum())
        return torch.stack(losses).mean() - 1.0

    def forward(
        self,
        waveform: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        waveform_lengths: Optional[torch.Tensor] = None,
    ) -> CAHB2VMoEOutput:
        if waveform.ndim != 2:
            raise ValueError(f"waveform must have shape [batch, samples], got {tuple(waveform.shape)}")
        if waveform_lengths is None:
            waveform_lengths = attention_mask.sum(dim=1) if attention_mask is not None else torch.full(
                (waveform.size(0),), waveform.size(1), dtype=torch.long, device=waveform.device
            )
        waveform_lengths = waveform_lengths.to(device=waveform.device, dtype=torch.long).clamp_min(1)
        if attention_mask is None:
            attention_mask = torch.arange(waveform.size(1), device=waveform.device)[None, :] < waveform_lengths[:, None]
        backbone_output = self.backbone(
            input_values=waveform,
            attention_mask=attention_mask.to(torch.long),
            output_hidden_states=False,
        )
        hidden = backbone_output.last_hidden_state
        token_mask = self.backbone.wavlm._get_feature_vector_attention_mask(hidden.size(1), attention_mask.to(torch.long))
        wavlm_utterance = _masked_mean(hidden, token_mask)
        wavlm_embedding = self.wavlm_projection(wavlm_utterance)

        # Slaney-normalized high-band filters can produce powers below the
        # fp16 normal range. Keep log-Mel, entropy, CMVN, and energy statistics
        # in fp32 so the EPS floor cannot underflow to zero inside AMP.
        with torch.autocast(device_type=waveform.device.type, enabled=False):
            mfcc, conditions = self._mfcc_bands(waveform.float(), waveform_lengths)
        frame_mask = self._frame_mask(waveform_lengths, mfcc.size(2))
        bands = torch.stack([self.mfcc_encoder(mfcc[:, i], frame_mask) for i in range(3)], dim=1)
        acoustic_embedding, band_gates, _ = self.band_moe(bands, conditions)
        context = torch.cat([conditions.flatten(1), band_gates], dim=-1)
        final_embedding, view_gates, _ = self.view_moe(wavlm_embedding, acoustic_embedding, context)
        logits = self.classifier(final_embedding)
        return CAHB2VMoEOutput(
            logits=logits,
            probabilities=F.softmax(logits, dim=-1),
            wavlm_embedding=wavlm_embedding,
            band_embeddings=bands,
            acoustic_embedding=acoustic_embedding,
            final_embedding=final_embedding,
            band_gates=band_gates,
            view_gates=view_gates,
            band_conditions=conditions,
            load_balance_loss=self._load_balance_loss(band_gates, view_gates),
            feature_lengths=backbone_output.feature_lengths,
        )


def cahb2v_loss(
    output: CAHB2VMoEOutput,
    targets: torch.Tensor,
    load_balance_weight: float = 1e-2,
    class_weights: Optional[torch.Tensor] = None,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """Cross-entropy plus the dense-routing balancing regularizer."""
    classification_loss = F.cross_entropy(
        output.logits,
        targets,
        weight=class_weights,
        label_smoothing=float(label_smoothing),
    )
    return classification_loss + float(load_balance_weight) * output.load_balance_loss


# Descriptive aliases make the implementation convenient to import from code
# that spells the acronym with separators or uses the common ``Model`` suffix.
CAHB2VMoEModel = CAHB2VMoE
CA_HB2V_MoE = CAHB2VMoE

__all__ = [
    "CAHB2VMoE",
    "CAHB2VMoEModel",
    "CA_HB2V_MoE",
    "CAHB2VMoEOutput",
    "cahb2v_loss",
]
