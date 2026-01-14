from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal, Optional, Tuple

import torch
from torch import nn

try:
    from einops import rearrange  # type: ignore
except Exception as e:  # pragma: no cover
    raise ImportError("Missing dependency: einops. Please install requirements-pip.txt.") from e


@dataclass
class CWTConfig:
    sfreq: float = 250.0
    f_min: float = 1.0
    f_max: float = 50.0
    n_freqs: int = 48
    freqs_scale: Literal["log", "linear"] = "log"
    morlet_w0: float = 6.0
    # output: magnitude or power
    out: Literal["magnitude", "power"] = "magnitude"


class CWTTransform(nn.Module):
    """
    Morlet CWT implemented via FFT-based correlation.

    Input:  x (B, C, T) real
    Output: X_tf (B, C, F, T) real, magnitude/power of complex CWT coefficients
    """

    def __init__(self, cfg: CWTConfig):
        super().__init__()
        self.cfg = cfg
        freqs = self._make_freqs(
            f_min=float(cfg.f_min),
            f_max=float(cfg.f_max),
            n=int(cfg.n_freqs),
            scale=str(cfg.freqs_scale),
        )
        # Stored for band masking and reproducibility
        self.register_buffer("freqs_hz", freqs, persistent=True)
        self._wavelet_cache: Dict[int, torch.Tensor] = {}

    @staticmethod
    def _make_freqs(f_min: float, f_max: float, n: int, scale: str) -> torch.Tensor:
        if n <= 0:
            raise ValueError(f"n_freqs must be > 0, got {n}")
        if f_min <= 0 or f_max <= 0 or f_max <= f_min:
            raise ValueError(f"Invalid freq range: f_min={f_min}, f_max={f_max}")
        if scale == "log":
            return torch.logspace(torch.log10(torch.tensor(f_min)), torch.log10(torch.tensor(f_max)), steps=n)
        if scale == "linear":
            return torch.linspace(f_min, f_max, steps=n)
        raise ValueError(f"Unknown freqs_scale={scale!r}")

    def _morlet_wavelets_fft(self, T: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """
        Returns wavelets in frequency domain for a given signal length T.
        Shape: (F, T) complex64/complex128
        """
        key = (int(T), int(device.index) if device.type == "cuda" else -1, 0 if dtype == torch.float32 else 1)
        cached = self._wavelet_cache.get(key, None)
        if cached is not None and cached.device == device and cached.dtype == (torch.complex64 if dtype == torch.float32 else torch.complex128):
            return cached

        sfreq = float(self.cfg.sfreq)
        w0 = float(self.cfg.morlet_w0)
        freqs = self.freqs_hz.to(device=device, dtype=dtype)

        # Morlet mother wavelet in time: pi^-0.25 * exp(1j*w0*t) * exp(-t^2/2)
        # Scale s chosen to target center frequency: f = (w0 / (2*pi)) / s  -> s = (w0 / (2*pi)) / f
        t = (torch.arange(T, device=device, dtype=dtype) - (T // 2)) / sfreq  # centered
        t = t.unsqueeze(0)  # (1, T)

        scales = (w0 / (2.0 * torch.pi)) / freqs  # (F,)
        scales = scales.clamp_min(1e-6).unsqueeze(1)  # (F,1)
        ts = t / scales  # (F,T)

        # Complex morlet
        norm = torch.pow(torch.pi, torch.tensor(-0.25, device=device, dtype=dtype))
        wavelet = norm * torch.exp(1j * w0 * ts) * torch.exp(-0.5 * ts * ts)  # (F,T) complex
        # Energy normalization across time (roughly scale-invariant)
        wavelet = wavelet / torch.sqrt(scales)

        # Correlation uses conj(wavelet)
        wavelet_fft = torch.fft.fft(torch.conj(wavelet), dim=-1)
        wavelet_fft = wavelet_fft.to(torch.complex64 if dtype == torch.float32 else torch.complex128)
        self._wavelet_cache[key] = wavelet_fft.detach()
        return wavelet_fft

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected x with shape (B,C,T), got {tuple(x.shape)}")
        B, C, T = x.shape
        device = x.device
        dtype = x.dtype if x.dtype in (torch.float32, torch.float64) else torch.float32
        x = x.to(dtype=dtype)

        wf = self._morlet_wavelets_fft(T=T, device=device, dtype=dtype)  # (F,T) complex
        xf = torch.fft.fft(x, dim=-1).to(wf.dtype)  # (B,C,T) complex
        # Broadcast multiply: (B,C,1,T) * (1,1,F,T) -> (B,C,F,T)
        coef = torch.fft.ifft(xf.unsqueeze(2) * wf.unsqueeze(0).unsqueeze(0), dim=-1)  # complex

        if self.cfg.out == "magnitude":
            out = torch.abs(coef)
        elif self.cfg.out == "power":
            out = torch.abs(coef) ** 2
        else:
            raise ValueError(f"Unknown CWT out={self.cfg.out!r}")
        return out.to(x.dtype)


def mask_frequency_band(
    X_tf: torch.Tensor,
    freqs_hz: torch.Tensor,
    band: Literal["none", "theta", "alpha", "beta"],
    mode: Literal["zero", "mean"] = "zero",
) -> torch.Tensor:
    """
    Mask a canonical EEG band on TF maps.

    X_tf: (B, C, F, T)
    freqs_hz: (F,)
    """
    if band == "none":
        return X_tf
    if X_tf.ndim != 4:
        raise ValueError(f"Expected X_tf (B,C,F,T), got {tuple(X_tf.shape)}")
    if freqs_hz.ndim != 1 or freqs_hz.numel() != X_tf.shape[2]:
        raise ValueError("freqs_hz must be shape (F,) aligned with X_tf frequency bins.")

    ranges = {
        "theta": (4.0, 8.0),
        "alpha": (8.0, 12.0),
        "beta": (13.0, 30.0),
    }
    lo, hi = ranges[band]
    mask = (freqs_hz >= lo) & (freqs_hz <= hi)  # (F,)
    if not bool(mask.any()):
        return X_tf

    X = X_tf.clone()
    if mode == "zero":
        X[:, :, mask, :] = 0.0
    elif mode == "mean":
        fill = X.mean(dim=2, keepdim=True)  # (B,C,1,T)
        X[:, :, mask, :] = fill.expand(-1, -1, int(mask.sum().item()), -1)
    else:
        raise ValueError(f"Unknown mask mode={mode!r}")
    return X


@dataclass
class STCNNConfig:
    d1: int = 32
    d2: int = 64
    kt: int = 15
    kf: int = 7
    pool_t: int = 4
    pool_f: int = 2
    dropout: float = 0.1
    act: Literal["gelu", "relu"] = "gelu"


class SpectroTemporalEncoder(nn.Module):
    """
    Temporal conv + spectral conv on TF maps.

    Input:  (B, C, F, T)
    Output: tokens (B, L, D) where L=Fp*Tp, D=d2
    """

    def __init__(self, in_ch: int, cfg: STCNNConfig):
        super().__init__()
        self.cfg = cfg
        act = nn.GELU() if cfg.act == "gelu" else nn.ReLU(inplace=True)

        self.temporal = nn.Sequential(
            nn.Conv2d(in_ch, int(cfg.d1), kernel_size=(1, int(cfg.kt)), padding=(0, int(cfg.kt) // 2), bias=False),
            nn.BatchNorm2d(int(cfg.d1)),
            act,
        )
        self.spectral = nn.Sequential(
            nn.Conv2d(int(cfg.d1), int(cfg.d2), kernel_size=(int(cfg.kf), 1), padding=(int(cfg.kf) // 2, 0), bias=False),
            nn.BatchNorm2d(int(cfg.d2)),
            act,
        )
        self.pool = nn.AvgPool2d(kernel_size=(int(cfg.pool_f), int(cfg.pool_t)))
        self.drop = nn.Dropout(float(cfg.dropout))

    def forward(self, x_tf: torch.Tensor) -> torch.Tensor:
        if x_tf.ndim != 4:
            raise ValueError(f"Expected x_tf (B,C,F,T), got {tuple(x_tf.shape)}")
        x = self.temporal(x_tf)
        x = self.spectral(x)
        x = self.pool(x)
        x = self.drop(x)
        # (B, D, Fp, Tp) -> (B, L, D)
        tokens = rearrange(x, "b d f t -> b (f t) d")
        return tokens


@dataclass
class AttnConfig:
    num_layers: int = 2
    num_heads: int = 4
    mlp_ratio: float = 4.0
    dropout: float = 0.1
    attn_dropout: float = 0.1


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, cfg: AttnConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=int(cfg.num_heads),
            dropout=float(cfg.attn_dropout),
            batch_first=True,
        )
        self.drop = nn.Dropout(float(cfg.dropout))
        self.ln2 = nn.LayerNorm(dim)
        hidden = int(round(dim * float(cfg.mlp_ratio)))
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(float(cfg.dropout)),
            nn.Linear(hidden, dim),
            nn.Dropout(float(cfg.dropout)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,L,D)
        h = self.ln1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + self.drop(attn_out)
        x = x + self.mlp(self.ln2(x))
        return x


class GlobalContextMHSA(nn.Module):
    """
    Stack of Transformer blocks for global context modeling.
    """

    def __init__(self, dim: int, cfg: AttnConfig):
        super().__init__()
        self.blocks = nn.ModuleList([TransformerBlock(dim=dim, cfg=cfg) for _ in range(int(cfg.num_layers))])

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = tokens
        for blk in self.blocks:
            x = blk(x)
        return x


@dataclass
class EarEmbedConfig:
    # Tasks
    num_valence_classes: int = 2
    num_load_classes: int = 3
    lambda_emo: float = 1.0
    lambda_load: float = 1.0

    # Ablations
    use_attention: bool = True
    mask_band: Literal["none", "theta", "alpha", "beta"] = "none"

    # Embedding
    emb_dim: int = 128
    l2_norm: bool = False

    # Sub-modules
    cwt: CWTConfig = CWTConfig()
    stcnn: STCNNConfig = STCNNConfig()
    attn: AttnConfig = AttnConfig()


class EarEmbedMultiTask(nn.Module):
    """
    Paper-matching Ear-Embed (TF + Spectro-Temporal CNN + optional MHSA + shared embedding + two heads).

    Input:  x (B,C,T)
    Output: dict(logits_valence, logits_load, embedding) depending on flags
    """

    def __init__(self, num_electrodes: int, cfg: EarEmbedConfig):
        super().__init__()
        if num_electrodes <= 0:
            raise ValueError(f"num_electrodes must be > 0, got {num_electrodes}")
        self.cfg = cfg

        self.tf = CWTTransform(cfg.cwt)
        self.encoder = SpectroTemporalEncoder(in_ch=int(num_electrodes), cfg=cfg.stcnn)

        dim_tokens = int(cfg.stcnn.d2)
        self.use_attention = bool(cfg.use_attention)
        self.mask_band = str(cfg.mask_band)
        self.attn = GlobalContextMHSA(dim=dim_tokens, cfg=cfg.attn) if self.use_attention else nn.Identity()

        self.proj = nn.Linear(dim_tokens, int(cfg.emb_dim))
        self.ln_emb = nn.LayerNorm(int(cfg.emb_dim))

        self.head_valence = nn.Linear(int(cfg.emb_dim), int(cfg.num_valence_classes))
        self.head_load = nn.Linear(int(cfg.emb_dim), int(cfg.num_load_classes))

    def forward(
        self,
        x: torch.Tensor,
        return_embedding: bool = False,
        return_tf: bool = False,
    ) -> Dict[str, torch.Tensor]:
        X_tf = self.tf(x)  # (B,C,F,T)
        X_tf = mask_frequency_band(X_tf, freqs_hz=self.tf.freqs_hz, band=self.cfg.mask_band, mode="zero")

        tokens = self.encoder(X_tf)  # (B,L,D)
        tokens = self.attn(tokens)  # (B,L,D) (or Identity)

        pooled = tokens.mean(dim=1)  # (B,D)
        z = self.ln_emb(self.proj(pooled))  # (B,emb)
        if bool(self.cfg.l2_norm):
            z = torch.nn.functional.normalize(z, p=2, dim=-1)

        out: Dict[str, torch.Tensor] = {
            "valence": self.head_valence(z),
            "load": self.head_load(z),
        }
        if return_embedding:
            out["embedding"] = z
        if return_tf:
            out["tf"] = X_tf
        return out

    def loss(self, logits: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> torch.Tensor:
        ce = nn.CrossEntropyLoss()
        loss_emo = ce(logits["valence"], targets["valence"].long())
        loss_load = ce(logits["load"], targets["load"].long())
        return float(self.cfg.lambda_emo) * loss_emo + float(self.cfg.lambda_load) * loss_load


