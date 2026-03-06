# Thinking Transformer

A from-scratch transformer implementation in C with a Python API — designed around *iterative reasoning*: the model runs multiple internal "think steps" before producing output, with full backpropagation through all of them.

Licensed under **GNU GPL v3**.

---

## What it is

Most transformer implementations do one forward pass and output a prediction. This one does `N` think steps first — re-running its own hidden states through the full transformer stack repeatedly before generating any tokens. Each step is differentiable, so training uses **full Backpropagation Through Time (BPTT)** across all think steps × all layers.

The model also carries a small **memory bank** (key-value slots) that is gated and read at each think step, giving it a lightweight persistent state across reasoning iterations.

The core is written in plain C (no dependencies beyond `libm`) and compiled into a shared library (`transformer.so` / `transformer.dll`). The Python wrapper loads this library via `ctypes` and exposes a clean object-oriented API.

**Architecture summary:**

- Multi-head self-attention + feed-forward layers (standard transformer blocks)
- Iterative reasoning loop: `think_steps` × full transformer stack per forward pass
- Gated memory read/write at each think step
- Special tokens: `PAD`, `THINK`, `PLAN`, `VERIFY`
- Adam optimizer with full BPTT gradients
- All dimensions configurable at runtime before initialisation

**Default parameter ranges** (hard limits baked into the C):

| Parameter | Default | Max |
|---|---|---|
| `vocab_size` | 64 | 256 |
| `embed_dim` | 32 | 128 |
| `num_heads` | 4 | 8 |
| `ff_dim` | 64 | 512 |
| `num_layers` | 2 | 6 |
| `max_seq_len` | 32 | 64 |
| `think_steps` | 3 | 8 |
| `memory_slots` | 8 | 16 |

---

## Building the C library

**Linux / macOS:**
```bash
gcc -O2 -shared -fPIC -o transformer.so transformer.c -lm
```

**Windows (MSYS2 / MinGW):**
```bash
gcc -O2 -shared -fPIC -o transformer.dll transformer.c -lm
```

Place the resulting `.so` / `.dll` in the same directory as `thinking_transformer.py`. The Python wrapper searches there (and `./build/`) automatically.

---

## Python API

### Setup

```python
from thinking_transformer import ThinkingTransformer, TransformerConfig

cfg = TransformerConfig(
    vocab_size=128, embed_dim=64, num_heads=4,
    ff_dim=128, num_layers=3, max_seq_len=48,
    think_steps=2, memory_slots=8,
)
model = ThinkingTransformer(config=cfg)
model.init(seed=42)

print(model)       # ThinkingTransformer(vocab=128, embed=64, ...)
print(model.info()) # detailed string from C layer
```

### Training

**Single step** — the most common case:
```python
tokens  = [5, 12, 3, 22, 7]
targets = ThinkingTransformer.make_targets(tokens)  # shift-left by 1

loss = model.train_step(tokens, targets, lr=1e-3)
```
`train_step` is a convenience wrapper for `zero_grad → backward → step`.

**Mini-batch** — accumulates gradients before a single Adam update:
```python
batch = [
    ([5, 12, 3], [12, 3, 0]),
    ([7, 2, 9],  [2, 9, 0]),
]
loss = model.train_batch(batch, lr=1e-3)
```

**Manual control** if you need to inspect or clip gradients between passes:
```python
model.zero_grad()
loss = model.backward(tokens, targets)   # full BPTT, returns loss
model.step(lr=1e-3)                      # Adam update
```

**Loss only** (no gradient):
```python
loss = model.compute_loss(tokens, targets)
```

### Inference

**Greedy generation:**
```python
output = model.generate(prompt=[5, 12, 3], max_new_tokens=10)
# returns list of token ids
```

**Generation with explicit reasoning tokens** (wraps prompt in THINK…PLAN and exposes the reasoning structure):
```python
result = model.generate_with_thinking(
    prompt=[5, 12, 3],
    max_new_tokens=10,
    verbose=True,
)
# result keys: input_with_reasoning, output_tokens, logits,
#              think_token_idx, plan_token_idx
```

**Raw logits** (shape `[seq_len, vocab_size]`):
```python
logits = model.forward(tokens)
```

**Log-probabilities:**
```python
logprobs = model.logprobs(tokens)
```

### Tokenisation helpers

The built-in tokeniser is byte-level — each byte of a UTF-8 string maps to `byte % vocab_size`:

```python
tokens = model.text_to_tokens("hello")
text   = model.tokens_to_text(tokens)
```

This is intentionally minimal. For anything real you'll want to bring your own tokeniser and just use the model as an integer-sequence processor.

### Save / load

```python
model.save("checkpoint.bin")   # returns 0 on success
model.load("checkpoint.bin")   # returns 0 on success
```

### Introspection

```python
model.vocab_size      # int
model.embed_dim       # int
model.max_seq_len     # int
model.is_ready()      # bool
model.param_count()   # total float parameters in C buffer
model.adam_step_count()  # how many optimizer steps so far
```

---

## Quick example — training loop

```python
from thinking_transformer import ThinkingTransformer, TransformerConfig

model = ThinkingTransformer(config=TransformerConfig())
model.init(seed=0)

corpus = [5, 12, 3, 22, 7, 18, 4, 9]
tokens  = corpus[:-1]
targets = corpus[1:]

for step in range(100):
    loss = model.train_step(tokens, targets, lr=5e-4)
    if step % 10 == 0:
        print(f"step {step:3d}  loss={loss:.4f}")

output = model.generate(corpus[:3], max_new_tokens=5)
print("generated:", output)
```

---

## License

This project is licensed under the **GNU General Public License v3.0**. See [https://www.gnu.org/licenses/gpl-3.0.html](https://www.gnu.org/licenses/gpl-3.0.html) for the full text.
