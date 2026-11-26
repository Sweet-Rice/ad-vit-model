"""Run the instruction-format model on raw bytes.

    python predict.py FILE [--offset N] [--device cpu|cuda] [--threshold 1.5]

Reads 1,024 bytes (512 little-endian 16-bit parcels) from FILE at --offset (shorter input is
zero-padded), runs the model, and prints one line per predicted instruction: its start parcel,
length in parcels, and its fields (bit ranges with a type). Use as a library through
load_model() and predict().
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from metrics import FIELD_TYPE_NAMES, decode_fields
from model import ViTConfig, ViTEncoder

WINDOW_BYTES = 1024
HERE = Path(__file__).resolve().parent


def load_model(path: str | Path = HERE / "model.pt", device: str = "cpu") -> ViTEncoder:
    """Build the model and load its weights."""
    blob = torch.load(path, map_location="cpu", weights_only=False)
    model = ViTEncoder(ViTConfig(**blob["config"]), attention="sdpa")
    model.load_state_dict(blob["state"])
    return model.to(device).eval()


def bytes_to_bits(windows: np.ndarray) -> torch.Tensor:
    """B x 1024 uint8 -> B x 512 x 4 x 4 float of +1/-1 (bit 4c+k of parcel r at [r, c, k])."""
    bits = np.unpackbits(windows, axis=1, bitorder="little").reshape(len(windows), -1, 4, 4)
    return torch.from_numpy(bits.astype(np.float32) * 2 - 1)


def boundaries(start_logit: np.ndarray, length_logits: np.ndarray) -> np.ndarray:
    """B x 512 instr_row (start row of the instruction owning each row, -1 for none), from the
    start and length heads: a row starts an instruction when its start logit > 0; it owns
    its predicted length in rows, cut short at the next start."""
    b, rows = start_logit.shape
    is_start = start_logit > 0
    length = length_logits.argmax(-1) + 1
    out = np.full((b, rows), -1, dtype=np.int64)
    for w in range(b):
        for r in np.flatnonzero(is_start[w]):
            for k in range(length[w, r]):
                if r + k >= rows or (k and is_start[w, r + k]):
                    break
                out[w, r + k] = r
    return out


@torch.no_grad()
def predict(model: ViTEncoder, windows: np.ndarray, threshold: float = 1.5) -> list[list[dict]]:
    """windows: B x 1024 uint8. Returns per window a list of instructions:
    {"start": parcel index, "length": parcels, "fields": [(first_bit, last_bit, type), ...]}
    with bits numbered from the instruction's first parcel (bit 16*k + j = parcel k, bit j)."""
    dev = next(model.parameters()).device
    out = model(bytes_to_bits(windows).to(dev), heads=("field_type", "grouping", "start_length"))
    ftype = out["field_type"].argmax(-1).cpu().numpy()          # B x 512 x 4 x 4
    instr_row = boundaries(out["start"].float().cpu().numpy(), out["length"].float().cpu().numpy())
    field = decode_fields(out["grouping"].float().cpu().numpy(), instr_row, threshold)
    result = []
    for w in range(len(windows)):
        instrs = []
        for s in np.unique(instr_row[w][instr_row[w] >= 0]):
            rows = np.flatnonzero(instr_row[w] == s)
            lab = field[w, rows].reshape(-1)                    # bit order: 16*k + 4c + j
            typ = ftype[w, rows].reshape(-1)
            fields = []
            for f in np.unique(lab):
                idx = np.flatnonzero(lab == f)
                t = np.bincount(typ[idx], minlength=5).argmax()   # majority type
                fields.append((int(idx.min()), int(idx.max()), FIELD_TYPE_NAMES[t],
                               len(idx) == idx.max() - idx.min() + 1))
            instrs.append({"start": int(s), "length": len(rows),
                           "fields": sorted(fields, key=lambda x: -x[0])})
        result.append(instrs)
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("file")
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--threshold", type=float, default=1.5)
    p.add_argument("--weights", default=str(HERE / "model.pt"))
    a = p.parse_args()
    data = Path(a.file).read_bytes()[a.offset:a.offset + WINDOW_BYTES]
    win = np.zeros((1, WINDOW_BYTES), dtype=np.uint8)
    win[0, :len(data)] = np.frombuffer(data, dtype=np.uint8)
    model = load_model(a.weights, a.device)
    for ins in predict(model, win, a.threshold)[0]:
        rows = range(ins["start"], ins["start"] + ins["length"])
        hexw = " ".join(f"{win[0, 2*r] | win[0, 2*r+1] << 8:04x}" for r in rows)
        f = "  ".join(f"[{hi}:{lo}]{t}" + ("" if contig else "*") for lo, hi, t, contig in ins["fields"])
        print(f"{a.offset + 2*ins['start']:#08x}  {hexw:<19}  {f}")


if __name__ == "__main__":
    main()
