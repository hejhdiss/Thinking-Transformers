# Thinking Transformer

A lightweight, educational Transformer implementation in C with Python bindings, featuring **iterative reasoning capabilities** and **training support**.

## Overview

This project implements a Transformer architecture with a unique "thinking" mechanism - the model performs multiple internal reasoning steps before producing output, mimicking human-like deliberation. It includes a full training pipeline with automatic differentiation and Adam optimization, all implemented in pure C for performance.

## Architecture
Input Tokens → Embedding → [Transformer + Reasoning Loop × T] → Output


### Key Features

- **Iterative Reasoning**: Configurable number of internal "thinking" steps (`THINK_STEPS = 3`)
- **Memory Module**: External memory slots that the model can read from and write to during reasoning
- **Special Tokens**: Built-in tokens for structured reasoning
  - `<PAD>` (0) - Padding token
  - `<THINK>` (1) - Begin reasoning phase
  - `<PLAN>` (2) - Planning phase marker
  - `<VERIFY>` (3) - Verification phase marker

### Model Specifications

Edit C code for another dimensions 
Example:
```
#define VOCAB_SIZE      256     // Was 64
#define EMBED_DIM       128     // Was 32
#define NUM_HEADS        8      // Was 4 (128/8=16, so HEAD_DIM=16)
#define FF_DIM          512     // Was 64
#define MAX_SEQ_LEN     128     // Was 32
#define NUM_LAYERS       4      // Was 2
#define THINK_STEPS      4      // Was 3
#define MEMORY_SLOTS    16      // Was 8
```

| Parameter | Value |
|-----------|-------|
| Vocabulary Size | 64 |
| Embedding Dimension | 32 |
| Number of Heads | 4 |
| Head Dimension | 8 |
| Feed-forward Dimension | 64 |
| Maximum Sequence Length | 32 |
| Number of Layers | 2 |
| Thinking Steps | 3 |
| Memory Slots | 8 |

## Project Structure

```
.
├── transformer.c          # Core C implementation (inference + training)
├── thinking_transformer.py # Python wrapper with ctypes
├── transformer.so         # Compiled shared library (Linux/macOS)
└── transformer.dll        # Compiled shared library (Windows)
```

## Compilation

### Linux / macOS

```bash
gcc -O2 -shared -fPIC -o transformer.so transformer.c -lm
```

### Windows (MSYS2 / MinGW)

```bash 
gcc -O2 -shared -fPIC -o transformer.dll transformer.c -lm
```

## Python API

### Quick Start

```
from thinking_transformer import ThinkingTransformer

# Initialize model
model = ThinkingTransformer()
model.init(seed=42)

# Check model info
print(model.info())
# Output: ThinkingTransformer | vocab=64 embed=32 heads=4 ...
```

### Inference

```
# Simple forward pass
tokens = [5, 12, 3, 22, 7]
logits = model.forward(tokens)  # Shape: (seq_len, vocab_size)

# Greedy generation
generated = model.generate(tokens[:3], max_new_tokens=5)

# Generation with reasoning structure
result = model.generate_with_thinking(
    prompt=tokens[:3],
    max_new_tokens=5,
    verbose=True
)
# Returns: {
#   'input_with_reasoning': [1, 5, 12, 3, 2],  # [THINK] + prompt + [PLAN]
#   'output_tokens': [...],
#   'logits': array(...),
#   'think_token_idx': 0,
#   'plan_token_idx': 4
# }
```

### Training 

```
# Prepare training data
tokens = [5, 12, 3, 22, 7]
targets = tokens[1:] + [0]  # Next-token prediction

# Compute loss (no gradient)
loss = model.compute_loss(tokens, targets)

# Single training step (zero_grad + backward + optimizer_step)
loss = model.train_step(tokens, targets, lr=1e-3)

# Or manually control the training loop:
model.zero_grad()
loss = model.backward(tokens, targets)  # Accumulates gradients
model.step(lr=1e-3)                      # Adam optimizer step

# Check training progress
print(f"Adam steps completed: {model.adam_step_count()}")
```

## License

GPL V3.
