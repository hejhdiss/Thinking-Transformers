"""
thinking_transformer.py
=======================
Python wrapper around transformer.dll / transformer.so

The C library exposes:
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
  transformer_adam_step()  -> long long          [NEW]
  transformer_cross_entropy_loss(tokens, seq_len, targets) -> float  [NEW]
  transformer_zero_grad()                        [NEW]
  transformer_backward(tokens, seq_len, targets, loss_out)  [NEW]
  transformer_step(lr)                           [NEW]

Usage
-----
  from thinking_transformer import ThinkingTransformer

  model = ThinkingTransformer()
  model.init(seed=42)

  # ── Forward / generate (unchanged) ───────────────────────
  logits   = model.forward([5, 12, 3])
  tokens   = model.generate([5, 12], max_new_tokens=8)
  result   = model.generate_with_thinking([5, 12])

  # ── Training API (new) ───────────────────────────────────
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
from pathlib import Path
from typing import List, Optional, Tuple


# ── Locate the shared library ───────────────────────────────────────────────

def _find_library(name: str) -> str:
    """Search for the compiled shared library next to this script or in CWD."""
    here = Path(__file__).parent.resolve()
    cwd  = Path.cwd()

    if platform.system() == "Windows":
        exts = [".dll"]
        prefixes = ["", "lib"]
    elif platform.system() == "Darwin":
        exts = [".dylib", ".so"]
        prefixes = ["lib", ""]
    else:
        exts = [".so"]
        prefixes = ["lib", ""]

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
        f"Searched:\n" + "\n".join(f"  {c}" for c in candidates) + "\n"
        f"\nBuild it first:\n"
        f"  Linux/macOS : gcc -O2 -shared -fPIC -o transformer.so transformer.c -lm\n"
        f"  Windows     : gcc -O2 -shared -fPIC -o transformer.dll transformer.c -lm\n"
    )


# ── ctypes bridge ────────────────────────────────────────────────────────────

class _CLib:
    """Low-level ctypes interface to the C transformer."""

    def __init__(self, lib_path: str):
        self._lib = ctypes.CDLL(lib_path)
        self._setup_signatures()

    def _setup_signatures(self):
        L = self._lib

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
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
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

        # ── Training (new) ────────────────────────────────────────────────
        L.transformer_adam_step.argtypes  = []
        L.transformer_adam_step.restype   = ctypes.c_longlong

        # float transformer_cross_entropy_loss(tokens, seq_len, targets)
        L.transformer_cross_entropy_loss.argtypes = [
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int),
        ]
        L.transformer_cross_entropy_loss.restype = ctypes.c_float

        # void transformer_zero_grad(void)
        L.transformer_zero_grad.argtypes  = []
        L.transformer_zero_grad.restype   = None

        # void transformer_backward(tokens, seq_len, targets, loss_out*)
        L.transformer_backward.argtypes = [
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_float),
        ]
        L.transformer_backward.restype = None

        # void transformer_step(float lr)
        L.transformer_step.argtypes = [ctypes.c_float]
        L.transformer_step.restype  = None


# ── High-level Python class ──────────────────────────────────────────────────

class ThinkingTransformer:
    """
    High-level Python interface to the Thinking Transformer C backend.

    Inference attributes
    --------------------
    vocab_size  : int   – vocabulary size
    embed_dim   : int   – embedding / hidden dimension
    max_seq_len : int   – maximum sequence length

    Training
    --------
    train_step(tokens, targets, lr)   – zero_grad + backward + step, returns loss
    compute_loss(tokens, targets)     – forward-only CE loss (no grad update)
    """

    # Special token constants (mirror transformer.c)
    TOK_PAD    = 0
    TOK_THINK  = 1
    TOK_PLAN   = 2
    TOK_VERIFY = 3

    def __init__(self, lib_path: Optional[str] = None):
        path = lib_path or _find_library("transformer")
        print(f"[ThinkingTransformer] Loading library: {path}")
        self._clib = _CLib(path)

        self.vocab_size  = self._clib._lib.transformer_vocab_size()
        self.embed_dim   = self._clib._lib.transformer_embed_dim()
        self.max_seq_len = self._clib._lib.transformer_max_seq()

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def init(self, seed: int = 42) -> "ThinkingTransformer":
        """Randomly initialise all weights (also resets Adam state)."""
        self._clib._lib.transformer_init(ctypes.c_uint(seed))
        print(f"[ThinkingTransformer] Weights initialised (seed={seed})")
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
        print(f"[ThinkingTransformer] Weights loaded ← {path}")

    def is_ready(self) -> bool:
        return bool(self._clib._lib.transformer_is_ready())

    def info(self) -> str:
        buf = ctypes.create_string_buffer(512)
        self._clib._lib.transformer_info(buf, 512)
        return buf.value.decode()

    def adam_step_count(self) -> int:
        """Return the number of completed Adam optimizer steps."""
        return int(self._clib._lib.transformer_adam_step())

    # ── Forward pass ──────────────────────────────────────────────────────

    def forward(self, tokens: List[int]) -> np.ndarray:
        """
        Run one full forward pass (with iterative thinking loop).

        Returns logits of shape (seq_len, vocab_size).
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
        """Like forward() but returns log-softmax probabilities."""
        logits = self.forward(tokens)
        logits -= logits.max(axis=-1, keepdims=True)
        probs   = np.exp(logits)
        probs  /= probs.sum(axis=-1, keepdims=True)
        return np.log(probs + 1e-9)

    # ── Training ──────────────────────────────────────────────────────────

    def compute_loss(
        self,
        tokens:  List[int],
        targets: List[int],
    ) -> float:
        """
        Compute cross-entropy loss without updating weights.

        Parameters
        ----------
        tokens  : input token ids  (length T)
        targets : target token ids (length T), typically tokens shifted by 1

        Returns
        -------
        float – mean cross-entropy loss
        """
        self._assert_ready()
        assert len(tokens) == len(targets), "tokens and targets must have the same length"
        T = len(tokens)
        c_tok = (ctypes.c_int * T)(*tokens)
        c_tgt = (ctypes.c_int * T)(*targets)
        return float(self._clib._lib.transformer_cross_entropy_loss(c_tok, T, c_tgt))

    def zero_grad(self) -> None:
        """Zero all accumulated gradients."""
        self._clib._lib.transformer_zero_grad()

    def backward(
        self,
        tokens:  List[int],
        targets: List[int],
    ) -> float:
        """
        Run forward + backward pass; accumulate gradients into C buffer.

        Returns the mean cross-entropy loss for this batch.
        Call zero_grad() before and step() after.
        """
        self._assert_ready()
        assert len(tokens) == len(targets), "tokens and targets must have the same length"
        T = len(tokens)
        c_tok   = (ctypes.c_int   * T)(*tokens)
        c_tgt   = (ctypes.c_int   * T)(*targets)
        c_loss  = ctypes.c_float(0.0)
        self._clib._lib.transformer_backward(c_tok, T, c_tgt,
                                              ctypes.byref(c_loss))
        return float(c_loss.value)

    def step(self, lr: float = 1e-3) -> None:
        """
        Apply one Adam optimizer step using accumulated gradients,
        then zero the gradient buffer.
        """
        self._clib._lib.transformer_step(ctypes.c_float(lr))

    def train_step(
        self,
        tokens:  List[int],
        targets: List[int],
        lr:      float = 1e-3,
    ) -> float:
        """
        Convenience method: zero_grad + backward + step.

        Returns
        -------
        float – mean cross-entropy loss
        """
        self.zero_grad()
        loss = self.backward(tokens, targets)
        self.step(lr)
        return loss

    # ── Generation ────────────────────────────────────────────────────────

    def generate(
        self,
        prompt: List[int],
        max_new_tokens: int = 16,
    ) -> List[int]:
        """Greedy auto-regressive generation."""
        self._assert_ready()
        prompt_len = len(prompt)
        c_prompt   = (ctypes.c_int * prompt_len)(*prompt)
        c_out      = (ctypes.c_int * max_new_tokens)()
        n = self._clib._lib.transformer_generate(
            c_prompt, prompt_len, c_out, max_new_tokens)
        return list(c_out[:n])

    def generate_with_thinking(
        self,
        prompt: List[int],
        max_new_tokens: int = 16,
        verbose: bool = False,
    ) -> dict:
        """
        Generate tokens and expose the think/plan/verify structure.

        Wraps prompt as: [THINK] + prompt + [PLAN]

        Returns
        -------
        dict:
          'input_with_reasoning' : full token list fed to the model
          'output_tokens'        : generated tokens
          'logits'               : raw logits for the input sequence
          'think_token_idx'      : index of <THINK> token (always 0)
          'plan_token_idx'       : index of <PLAN> token
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

    # ── Token utilities ───────────────────────────────────────────────────

    @staticmethod
    def text_to_tokens(text: str, vocab_size: int = 64) -> List[int]:
        """Trivial byte-level tokeniser (mod vocab_size)."""
        return [b % vocab_size for b in text.encode("utf-8")]

    @staticmethod
    def tokens_to_text(tokens: List[int]) -> str:
        """Best-effort bytes → string (demo only, not lossless)."""
        return bytes(t % 256 for t in tokens).decode("utf-8", errors="replace")

    # ── Internals ─────────────────────────────────────────────────────────

    def _assert_ready(self):
        if not self.is_ready():
            raise RuntimeError(
                "Model not initialised. Call .init() or .load() first."
            )

    def __repr__(self) -> str:
        return (
            f"ThinkingTransformer("
            f"vocab={self.vocab_size}, "
            f"embed={self.embed_dim}, "
            f"max_seq={self.max_seq_len}, "
            f"ready={self.is_ready()}, "
            f"adam_steps={self.adam_step_count()}"
            f")"
        )


# ── Quick smoke-test ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  Thinking Transformer — Python Demo (with Training)")
    print("=" * 60)

    model = ThinkingTransformer()
    model.init(seed=1337)
    print(f"Model : {model}")
    print(f"Info  : {model.info()}")
    print()

    tokens = [5, 12, 3, 22, 7]
    targets = tokens[1:] + [0]   # next-token targets

    # Before training
    loss_before = model.compute_loss(tokens, targets)
    print(f"Loss before training : {loss_before:.4f}")

    # 20 training steps
    for step in range(20):
        loss = model.train_step(tokens, targets, lr=1e-3)
        if step % 5 == 0:
            print(f"  step {step:3d}  loss={loss:.4f}")

    loss_after = model.compute_loss(tokens, targets)
    print(f"Loss after  training : {loss_after:.4f}")
    print()

    # Generate
    generated = model.generate(tokens[:3], max_new_tokens=5)
    print(f"Generated : {generated}")

    # Thinking generation
    result = model.generate_with_thinking(tokens[:3], max_new_tokens=5, verbose=True)
    print(f"Output    : {result['output_tokens']}")
    print()
    print("Done ✓")
