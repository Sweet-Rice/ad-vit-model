# ad-vit-model

A model that reads 1,024 bytes of machine code in an unknown (scrambled) RISC-V-style encoding
and recovers its instruction layout: where each instruction starts, how long it is, which bits
form each field, and each field's type (fixed, register, immediate, other variable, not code).

## Contents

| File | What it is |
|---|---|
| `model.pt` | Weights (float32, 26.6 MB) and the model's config. 6.64M parameters. |
| `model.py` | The network (ViT encoder plus output heads). Needs only PyTorch. |
| `metrics.py` | `decode_fields`, which turns the grouping output into fields. Needs only numpy. The rest of the file is evaluation code. |
| `predict.py` | Command-line tool and library functions: bytes in, instructions and fields out. |

## Setup

Python 3.12 or newer, plus PyTorch (CPU, CUDA or ROCm) and numpy:

```
pip install torch numpy
```

## Use

```
python predict.py firmware.bin --offset 0x0 --device cpu
```

The tool reads 1,024 bytes from the file at `--offset` and zero-pads them if the file is shorter.
The offset must be 2-byte aligned. It prints one line per instruction: the byte address, the
16-bit parcels, then the fields from high bit to low. One window takes about 4 seconds on CPU,
mostly to load the model.

```
0x000000  00a7 00b3   [31:29]fixed  [28:24]register  [23:19]register  [18:14]fixed  [13:2]immediate  [1:0]fixed
0x000008  00d7 0021   [18:14]fixed  [31:7]immediate*  [6:2]register  [1:0]fixed
```

- Bits are numbered from the first parcel of the instruction: bit `16*k + j` is bit `j` of parcel
  `k`.
- `*` marks a field whose bits are not contiguous. It is printed as its lowest and highest bit,
  so the range also covers bits of other fields.

As a library, for many windows at once:

```python
import numpy as np
from predict import load_model, predict

model = load_model(device="cuda")            # or "cpu"
windows = np.frombuffer(data, np.uint8).reshape(-1, 1024)   # B x 1024 bytes
for instr in predict(model, windows)[0]:     # instructions of the first window
    print(instr["start"], instr["length"], instr["fields"])  # fields: (low bit, high bit, type, contiguous)
```

How the output is formed:
- **Starts and lengths:** a parcel starts an instruction when its start logit is above 0. The
  instruction's length (1 to 4 parcels) is the length head's top choice.
- **Fields:** inside an instruction, two bits whose grouping vectors are closer than
  `--threshold` (default 1.5) are joined into one field.
- **Field type:** the most common predicted type among the field's bits.

## How good it is

Scores are from the project's fixed eval set, `heldout_both`: programs and encodings never seen
in training, 2,000 windows per level.

- **Levels:** L0 is real RISC-V. L1 to L4 are progressively harder scrambles; at L4, fields are
  split into scattered bits.
- **Exact-format:** for each distinct instruction format in a window, whether the most common
  predicted layout matches it exactly (length, field bit sets, and types).

| Level | L0 | L1 | L2 | L3 | L4 | All |
|---|---|---|---|---|---|---|
| Exact-format, threshold 1.0 | 0.991 | 0.793 | 0.624 | 0.440 | 0.117 | 0.561 |
| Exact-format, threshold 1.5 (the default here) | | | | | | 0.589 |

Other scores, all at threshold 1.0:
- Field-type macro-F1: 0.978.
- Field F1: 0.848.

**The model's own boundaries:** the scores above use the true instruction boundaries. With the
boundaries the model predicts itself, 99.9% of instructions get exactly the right start and
length (64 held-out windows per level). So the scores should hold with no boundaries given.

## Limits

- **Input size:** windows are exactly 1,024 bytes. For a longer binary, run overlapping windows.
  Instructions cut off at a window's edge may be wrong.
- **Hard scrambles:** L3 and L4 are often wrong. Treat field boundaries there as guesses.
- **Training data:** RISC-V (rv32i, rv32imc) code built with GCC and Clang. Code from very
  different compilers or ISAs has not been tested.
