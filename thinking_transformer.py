"""
thinking_transformer.py
=======================
Python wrapper around transformer.dll / transformer.so

The C library exposes:
  transformer_configure(vocab, embed, heads, ff, layers, seq, think, mem) -> int  [NEW]
  transformer_init(seed)
  transformer_save(path)
  transformer_load(path)
  transformer_forward(tokens, seq_len, logits_out)
  transformer_generate(prompt, prompt_len, out_tokens, max_new_tokens) -> int
  transformer_info(buf, buf_len)
  transformer_vocab_size() -> int
  transformer_embed_dim()  -> int
  transformer_max_seq()    -> int
  transformer_is_ready()   -> int
  transformer_param_count() -> int                    [NEW]
  transformer_adam_step()  -> long long
  transformer_cross_entropy_loss(tokens, seq_len, targets) -> float
  transformer_zero_grad()
  transformer_backward(tokens, seq_len, targets, loss_out)   [Full BPTT]
  transformer_step(lr)

Full BPTT
---------
  All think-steps × all transformer layers are now differentiable.
  Activations are cached during forward; backward unrolls through the
  full computation graph (think_steps × layers × attention + FF).

Usage
-----
  from thinking_transformer import ThinkingTransformer, TransformerConfig

  # Custom architecture
  cfg = TransformerConfig(
      vocab_size=128, embed_dim=64, num_heads=4,
      ff_dim=128, num_layers=3, max_seq_len=48,
      think_steps=2, memory_slots=8,
  )
  model = ThinkingTransformer(config=cfg)
  model.init(seed=42)

  # ── Training ──────────────────────────────────────────────
  loss = model.train_step(
      tokens=[5, 12, 3, 22],
      targets=[12, 3, 22, 7],
      lr=1e-3,
  )

  GPL V3.
"""

import ctypes
import os
import platform
import sys
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple


# ── Configuration dataclass ──────────────────────────────────────────────────

@dataclass
class TransformerConfig:
    """
    Runtime configuration for the Thinking Transformer.
    All parameters can be tuned before model.init().
    """
    vocab_size:    int = 64
    embed_dim:     int = 32
    num_heads:     int = 4
    ff_dim:        int = 64
    num_layers:    int = 2
    max_seq_len:   int = 32
    think_steps:   int = 3
    memory_slots:  int = 8

    def validate(self):
        if self.embed_dim % self.num_heads != 0:
            raise ValueError(
                f"embed_dim ({self.embed_dim}) must be divisible by "
                f"num_heads ({self.num_heads})"
            )
        assert 4 <= self.vocab_size   <= 256,  "vocab_size out of range [4, 256]"
        assert 4 <= self.embed_dim    <= 128,  "embed_dim out of range [4, 128]"
        assert 1 <= self.num_heads    <= 8,    "num_heads out of range [1, 8]"
        assert 4 <= self.ff_dim       <= 512,  "ff_dim out of range [4, 512]"
        assert 1 <= self.num_layers   <= 6,    "num_layers out of range [1, 6]"
        assert 4 <= self.max_seq_len  <= 64,   "max_seq_len out of range [4, 64]"
        assert 1 <= self.think_steps  <= 8,    "think_steps out of range [1, 8]"
        assert 1 <= self.memory_slots <= 16,   "memory_slots out of range [1, 16]"

    @property
    def head_dim(self) -> int:
        return self.embed_dim // self.num_heads

    def param_count(self) -> int:
        """Rough estimate of total float parameters."""
        E, V, L, F, S, MS = (self.embed_dim, self.vocab_size, self.num_layers,
                              self.ff_dim, self.max_seq_len, self.memory_slots)
        p  = V * E + S * E                 # embeddings
        p += L * (4 * E * E + 2 * E)      # attn weights + LN
        p += L * (E * F + F + F * E + E + 2 * E)  # FF + LN
        p += E * E + E + 2 * E            # reasoning + LN
        p += MS * E + E * MS + MS * E     # memory
        p += E * V + V                    # output
        return p


# ── Locate the shared library ─────────────────────────────────────────────────

def _find_library(name: str) -> str:
    here = Path(__file__).parent.resolve()
    cwd  = Path.cwd()

    if platform.system() == "Windows":
        exts, prefixes = [".dll"], ["", "lib"]
    elif platform.system() == "Darwin":
        exts, prefixes = [".dylib", ".so"], ["lib", ""]
    else:
        exts, prefixes = [".so"], ["lib", ""]

    candidates = []
    for base in [here, cwd, here / "build", cwd / "build"]:
        for prefix in prefixes:
            for ext in exts:
                candidates.append(base / f"{prefix}{name}{ext}")

    for p in candidates:
        if p.exists():
            return str(p)

    raise FileNotFoundError(
        f"Cannot find '{name}' shared library.\n"
        f"Searched:\n" + "\n".join(f"  {c}" for c in candidates) + "\n\n"
        f"Build it first:\n"
        f"  Linux/macOS : gcc -O2 -shared -fPIC -o transformer.so transformer.c -lm\n"
        f"  Windows     : gcc -O2 -shared -fPIC -o transformer.dll transformer.c -lm\n"
    )


# ── ctypes bridge ─────────────────────────────────────────────────────────────

class _CLib:
    def __init__(self, lib_path: str):
        self._lib = ctypes.CDLL(lib_path)
        self._setup_signatures()

    def _setup_signatures(self):
        L = self._lib

        # ── Configuration (NEW) ───────────────────────────────────────────
        L.transformer_configure.argtypes = [
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ]
        L.transformer_configure.restype = ctypes.c_int

        # ── Core ──────────────────────────────────────────────────────────
        L.transformer_init.argtypes = [ctypes.c_uint]
        L.transformer_init.restype  = None

        L.transformer_save.argtypes = [ctypes.c_char_p]
        L.transformer_save.restype  = ctypes.c_int

        L.transformer_load.argtypes = [ctypes.c_char_p]
        L.transformer_load.restype  = ctypes.c_int

        L.transformer_forward.argtypes = [
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_float),
        ]
        L.transformer_forward.restype = None

        L.transformer_generate.argtypes = [
            ctypes.POINTER(ctypes.c_int), ctypes.c_int,
            ctypes.POINTER(ctypes.c_int), ctypes.c_int,
        ]
        L.transformer_generate.restype = ctypes.c_int

        L.transformer_info.argtypes = [ctypes.c_char_p, ctypes.c_int]
        L.transformer_info.restype  = None

        L.transformer_vocab_size.argtypes = []
        L.transformer_vocab_size.restype  = ctypes.c_int
        L.transformer_embed_dim.argtypes  = []
        L.transformer_embed_dim.restype   = ctypes.c_int
        L.transformer_max_seq.argtypes    = []
        L.transformer_max_seq.restype     = ctypes.c_int
        L.transformer_is_ready.argtypes   = []
        L.transformer_is_ready.restype    = ctypes.c_int
        L.transformer_param_count.argtypes= []
        L.transformer_param_count.restype = ctypes.c_int

        # ── Training ──────────────────────────────────────────────────────
        L.transformer_adam_step.argtypes  = []
        L.transformer_adam_step.restype   = ctypes.c_longlong

        L.transformer_cross_entropy_loss.argtypes = [
            ctypes.POINTER(ctypes.c_int), ctypes.c_int,
            ctypes.POINTER(ctypes.c_int),
        ]
        L.transformer_cross_entropy_loss.restype = ctypes.c_float

        L.transformer_zero_grad.argtypes  = []
        L.transformer_zero_grad.restype   = None

        L.transformer_backward.argtypes = [
            ctypes.POINTER(ctypes.c_int), ctypes.c_int,
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_float),
        ]
        L.transformer_backward.restype = None

        L.transformer_step.argtypes = [ctypes.c_float]
        L.transformer_step.restype  = None


# ── High-level Python class ───────────────────────────────────────────────────

class ThinkingTransformer:
    """
    High-level Python interface to the Thinking Transformer C backend.

    Supports runtime configuration of all hyperparameters, full BPTT
    training, and byte-level tokenisation for quick experiments.

    Parameters
    ----------
    config    : TransformerConfig  – architecture specification
    lib_path  : str | None         – explicit path to shared library

    Example
    -------
    >>> from thinking_transformer import ThinkingTransformer, TransformerConfig
    >>> cfg = TransformerConfig(vocab_size=128, embed_dim=64, num_heads=4,
    ...                         ff_dim=128, num_layers=2, max_seq_len=32)
    >>> model = ThinkingTransformer(config=cfg)
    >>> model.init(seed=0)
    >>> loss = model.train_step([4,5,6], [5,6,7], lr=1e-3)
    """

    # Special token constants
    TOK_PAD    = 0
    TOK_THINK  = 1
    TOK_PLAN   = 2
    TOK_VERIFY = 3

    def __init__(
        self,
        config:   Optional[TransformerConfig] = None,
        lib_path: Optional[str] = None,
    ):
        self.config = config or TransformerConfig()
        self.config.validate()

        path = lib_path or _find_library("transformer")
        print(f"[ThinkingTransformer] Loading library: {path}")
        self._clib = _CLib(path)

        # Apply configuration to C backend
        rc = self._clib._lib.transformer_configure(
            self.config.vocab_size,
            self.config.embed_dim,
            self.config.num_heads,
            self.config.ff_dim,
            self.config.num_layers,
            self.config.max_seq_len,
            self.config.think_steps,
            self.config.memory_slots,
        )
        if rc != 0:
            raise ValueError(
                "transformer_configure() rejected the configuration. "
                "Check that embed_dim is divisible by num_heads and all "
                "values are within supported ranges."
            )

        # Sync Python-side properties from C backend
        self._sync_properties()

    def _sync_properties(self):
        L = self._clib._lib
        self.vocab_size  = L.transformer_vocab_size()
        self.embed_dim   = L.transformer_embed_dim()
        self.max_seq_len = L.transformer_max_seq()

    # ── Lifecycle ──────────────────────────────────────────────────────────

    def init(self, seed: int = 42) -> "ThinkingTransformer":
        """Randomly initialise all weights (also resets Adam state)."""
        self._clib._lib.transformer_init(ctypes.c_uint(seed))
        self._sync_properties()
        print(f"[ThinkingTransformer] Weights initialised (seed={seed})")
        return self

    def configure(self, **kwargs) -> "ThinkingTransformer":
        """
        Update architecture configuration.
        Must be called BEFORE init() / after load().

        Keyword args mirror TransformerConfig fields.
        """
        for k, v in kwargs.items():
            if not hasattr(self.config, k):
                raise ValueError(f"Unknown config key: {k!r}")
            setattr(self.config, k, v)
        self.config.validate()
        rc = self._clib._lib.transformer_configure(
            self.config.vocab_size, self.config.embed_dim,
            self.config.num_heads,  self.config.ff_dim,
            self.config.num_layers, self.config.max_seq_len,
            self.config.think_steps, self.config.memory_slots,
        )
        if rc != 0:
            raise ValueError("transformer_configure() rejected updated config.")
        return self

    def save(self, path: str) -> None:
        rc = self._clib._lib.transformer_save(path.encode())
        if rc != 0:
            raise IOError(f"transformer_save failed (rc={rc})")
        print(f"[ThinkingTransformer] Weights saved → {path}")

    def load(self, path: str) -> None:
        rc = self._clib._lib.transformer_load(path.encode())
        if rc != 0:
            raise IOError(f"transformer_load failed (rc={rc})")
        self._sync_properties()
        print(f"[ThinkingTransformer] Weights loaded ← {path}")

    def is_ready(self) -> bool:
        return bool(self._clib._lib.transformer_is_ready())

    def info(self) -> str:
        buf = ctypes.create_string_buffer(1024)
        self._clib._lib.transformer_info(buf, 1024)
        return buf.value.decode()

    def param_count(self) -> int:
        """Return the number of trainable float parameters."""
        return int(self._clib._lib.transformer_param_count())

    def adam_step_count(self) -> int:
        return int(self._clib._lib.transformer_adam_step())

    # ── Forward pass ───────────────────────────────────────────────────────

    def forward(self, tokens: List[int]) -> np.ndarray:
        """
        Run one full forward pass (with iterative thinking loop).
        Returns logits of shape (seq_len, vocab_size).
        Also caches all activations for subsequent backward().
        """
        self._assert_ready()
        seq_len = len(tokens)
        if seq_len > self.max_seq_len:
            raise ValueError(f"Sequence too long ({seq_len} > {self.max_seq_len})")
        c_tokens = (ctypes.c_int   * seq_len)(*tokens)
        c_logits = (ctypes.c_float * (seq_len * self.vocab_size))()
        self._clib._lib.transformer_forward(c_tokens, seq_len, c_logits)
        return np.frombuffer(c_logits, dtype=np.float32).reshape(
            seq_len, self.vocab_size).copy()

    def logprobs(self, tokens: List[int]) -> np.ndarray:
        logits = self.forward(tokens)
        logits -= logits.max(axis=-1, keepdims=True)
        probs   = np.exp(logits)
        probs  /= probs.sum(axis=-1, keepdims=True)
        return np.log(probs + 1e-9)

    # ── Training ───────────────────────────────────────────────────────────

    def compute_loss(self, tokens: List[int], targets: List[int]) -> float:
        """Compute cross-entropy loss without updating weights."""
        self._assert_ready()
        assert len(tokens) == len(targets)
        T = len(tokens)
        c_tok = (ctypes.c_int * T)(*tokens)
        c_tgt = (ctypes.c_int * T)(*targets)
        return float(self._clib._lib.transformer_cross_entropy_loss(c_tok, T, c_tgt))

    def zero_grad(self) -> None:
        self._clib._lib.transformer_zero_grad()

    def backward(self, tokens: List[int], targets: List[int]) -> float:
        """
        Run full forward + BPTT backward pass.
        Gradients are accumulated into the C gradient buffer.
        Call zero_grad() before and step() after.

        Returns the mean cross-entropy loss.
        """
        self._assert_ready()
        assert len(tokens) == len(targets)
        T = len(tokens)
        c_tok  = (ctypes.c_int   * T)(*tokens)
        c_tgt  = (ctypes.c_int   * T)(*targets)
        c_loss = ctypes.c_float(0.0)
        self._clib._lib.transformer_backward(c_tok, T, c_tgt, ctypes.byref(c_loss))
        return float(c_loss.value)

    def step(self, lr: float = 1e-3) -> None:
        """Apply one Adam optimizer step, then zero gradients."""
        self._clib._lib.transformer_step(ctypes.c_float(lr))

    def train_step(
        self,
        tokens:  List[int],
        targets: List[int],
        lr:      float = 1e-3,
    ) -> float:
        """
        Convenience method: zero_grad → full BPTT backward → Adam step.
        Returns mean cross-entropy loss.
        """
        self.zero_grad()
        loss = self.backward(tokens, targets)
        self.step(lr)
        return loss

    def train_batch(
        self,
        batch:   List[Tuple[List[int], List[int]]],
        lr:      float = 1e-3,
    ) -> float:
        """
        Mini-batch training: accumulate gradients over all samples,
        then take one Adam step. Returns mean loss.

        Parameters
        ----------
        batch : list of (tokens, targets) pairs
        lr    : learning rate
        """
        self._assert_ready()
        self.zero_grad()
        total_loss = 0.0
        for tokens, targets in batch:
            assert len(tokens) == len(targets)
            T = len(tokens)
            c_tok  = (ctypes.c_int   * T)(*tokens)
            c_tgt  = (ctypes.c_int   * T)(*targets)
            c_loss = ctypes.c_float(0.0)
            self._clib._lib.transformer_backward(c_tok, T, c_tgt, ctypes.byref(c_loss))
            total_loss += float(c_loss.value)
        self.step(lr)
        return total_loss / max(len(batch), 1)

    # ── Generation ─────────────────────────────────────────────────────────

    def generate(self, prompt: List[int], max_new_tokens: int = 16) -> List[int]:
        self._assert_ready()
        prompt_len = len(prompt)
        c_prompt   = (ctypes.c_int * prompt_len)(*prompt)
        c_out      = (ctypes.c_int * max_new_tokens)()
        n = self._clib._lib.transformer_generate(
            c_prompt, prompt_len, c_out, max_new_tokens)
        return list(c_out[:n])

    def generate_with_thinking(
        self,
        prompt:         List[int],
        max_new_tokens: int  = 16,
        verbose:        bool = False,
    ) -> dict:
        """
        Wrap prompt with THINK/PLAN tokens and expose reasoning structure.

        Returns dict with keys:
          input_with_reasoning, output_tokens, logits,
          think_token_idx, plan_token_idx
        """
        think_prompt = [self.TOK_THINK] + list(prompt) + [self.TOK_PLAN]
        if verbose:
            print(f"[ThinkingTransformer] Reasoning prompt: {think_prompt}")
        output = self.generate(think_prompt, max_new_tokens=max_new_tokens)
        logits = self.forward(think_prompt)
        if verbose:
            print(f"[ThinkingTransformer] Generated tokens: {output}")
        return {
            "input_with_reasoning": think_prompt,
            "output_tokens":        output,
            "logits":               logits,
            "think_token_idx":      0,
            "plan_token_idx":       len(think_prompt) - 1,
        }

    # ── Token utilities ────────────────────────────────────────────────────

    def text_to_tokens(self, text: str) -> List[int]:
        """Byte-level tokeniser — maps bytes mod vocab_size."""
        return [b % self.vocab_size for b in text.encode("utf-8")]

    def tokens_to_text(self, tokens: List[int]) -> str:
        return bytes(t % 256 for t in tokens).decode("utf-8", errors="replace")

    @staticmethod
    def make_targets(tokens: List[int], pad: int = 0) -> List[int]:
        """Shift tokens left by one for next-token prediction."""
        return tokens[1:] + [pad]

    # ── Internals ──────────────────────────────────────────────────────────

    def _assert_ready(self):
        if not self.is_ready():
            raise RuntimeError(
                "Model not initialised. Call .init() or .load() first."
            )

    def __repr__(self) -> str:
        return (
            f"ThinkingTransformer("
            f"vocab={self.vocab_size}, embed={self.embed_dim}, "
            f"heads={self.config.num_heads}, ff={self.config.ff_dim}, "
            f"layers={self.config.num_layers}, "
            f"think_steps={self.config.think_steps}, "
            f"max_seq={self.max_seq_len}, "
            f"params={self.param_count() if self.is_ready() else '?'}, "
            f"adam_steps={self.adam_step_count()}"
            f")"
        )


# ── Quick smoke-test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  Thinking Transformer — Python Demo (Full BPTT)")
    print("=" * 60)

    cfg = TransformerConfig(
        vocab_size=64, embed_dim=32, num_heads=4,
        ff_dim=64, num_layers=2, max_seq_len=32,
        think_steps=2, memory_slots=8,
    )
    model = ThinkingTransformer(config=cfg)
    model.init(seed=1337)
    print(f"Model : {model}")
    print(f"Info  : {model.info()}")
    print()

    tokens  = [5, 12, 3, 22, 7]
    targets = model.make_targets(tokens)

    loss_before = model.compute_loss(tokens, targets)
    print(f"Loss before training : {loss_before:.4f}")

    for step in range(30):
        loss = model.train_step(tokens, targets, lr=1e-3)
        if step % 5 == 0:
            print(f"  step {step:3d}  loss={loss:.4f}")

    loss_after = model.compute_loss(tokens, targets)
    print(f"Loss after  training : {loss_after:.4f}")

    generated = model.generate(tokens[:3], max_new_tokens=5)
    print(f"Generated : {generated}")

    result = model.generate_with_thinking(tokens[:3], max_new_tokens=5, verbose=True)
    print(f"Output    : {result['output_tokens']}")
    print()
    print("Done ✓")
