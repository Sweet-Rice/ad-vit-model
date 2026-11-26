"""Evaluation metrics (evaluation.md), scored against a model batch and accumulated per level.

Model-agnostic: every function takes predictions as plain arrays or tensors (numpy or torch, any
device), so the ViT and the B0 baseline are scored by the same code. The batch is the one
``vit.data.make_batch`` returns (B windows, R = 512 rows, per-bit arrays B×R×4×4 with bit
b of patch (r, c) = parcel bit 4c+b).

Predictions:

=================  =====================  ===================================================
argument           shape                  meaning
=================  =====================  ===================================================
``pred_type``      B×R×4×4 int            per-bit field type 0..4 (or B×R×4×4×5 logits)
``pred_field``     B×R×4×4 int            per-bit field label; only equality within one
                                          instruction matters (``decode_fields`` makes it
                                          from grouping vectors, B0 gives it directly)
``pred_length``    B×R int                length in parcels 1..4 at start rows (or B×R×4
                                          logits); optional
``pred_cluster``   B×R int                cluster id of the instruction starting at each row
                                          (``clusters_from_embeddings`` makes it); optional
=================  =====================  ===================================================

Rules the evaluation doc leaves open, as implemented here:

- **Oracle boundaries.** There is no segmentation decoder yet, so every instruction-level metric
  (field F1, exact-format rate, length accuracy, clustering) uses the TRUE starts and lengths.
  Boundary F1 and shift consistency are not computed.
- **Cut instructions** (``cut`` set on their start row; head rows own no instruction) keep their
  bits in the per-bit field-type metrics and are left out of every instruction-level metric.
- **Field-type macro-F1** averages F1 over the 4 classes that occur (fixed, register, immediate,
  not-code). ``macro_f1_code`` also reports the 3 code classes alone. A class with no true and no
  predicted bits in the scored set is left out of the average.
- **Field decoding** (architecture.md, "closer than 1.0 form one field"): within one instruction,
  a field is a connected component of the graph joining two bits whose grouping vectors are at
  Euclidean distance < ``threshold``. So a chain A-B, B-C close with A-C far is one field. A
  field's type is the majority of its bits' predicted types; ties go to the lowest type index.
- **Field F1** is micro-averaged over all (non-cut) instructions: TP = predicted fields whose bit
  set equals a true field's bit set and whose type equals that field's type.
- **True fields** are the groups of bits sharing a ``field_id`` within an instruction, as the
  labels have them. The labels give separate ids to opcode, funct and fixed-mask bits, so an
  instruction usually has 2-5 fixed fields. ``*_fixed_merged`` variants also score with all
  fixed-typed bits of an instruction merged into one field on both sides (true: type 0 bits;
  predicted: bits of fields whose majority type is fixed).
- **Formats** for the exact-format rate: a window's (non-cut) instructions are grouped by their
  TRUE layout, i.e. length plus the partition of bit positions into typed fields; each distinct
  true layout in a window is one format. Per format, the predicted layouts of its instructions
  are counted; the format is correct when the most common predicted layout is strictly more
  common than any other and equals the true layout (a tie counts as incorrect). Formats with one
  instruction are included (their only prediction decides); ``rate_multi`` reports formats with
  at least 2 instructions. ``exact_format_by_class`` groups by ``instr_class`` instead, with the
  class's true layout taken as its most common one.
- **Length accuracy** is over non-cut true start rows.
- **Clustering** is the adjusted Rand index (``adjusted_rand_index``) between predicted clusters
  and ``instr_class`` over a window's non-cut instructions, averaged over windows with at least
  2 such instructions. ``threshold_clusters`` is the default embeddings -> clusters step:
  connected components of cosine similarity >= ``min_cos`` (0.5); any other function can be
  passed to ``clusters_from_embeddings``.

Native core: the optional C++ extension ``vit._native_metrics`` (built by setup.py, see
NATIVE_EVAL.md) runs field decoding, the instruction-level part of ``score_batch``, its ARI loop
and ``clusters_from_embeddings`` with ``threshold_clusters``. Each native call either gives
exactly this module's numpy result or declines (whole call, or single instructions / windows whose
threshold comparisons are too close to decide), and the numpy code below runs instead.
``HARNESS_VIT_METRICS_NATIVE=0`` (or ``set_native(False)``) switches it off.
"""
from __future__ import annotations

import os
from typing import Any, Callable, Mapping

import numpy as np

try:
    from . import _native_metrics as _native_mod  # type: ignore[attr-defined]
except ImportError:
    _native_mod = None

_use_native = os.environ.get("HARNESS_VIT_METRICS_NATIVE", "1").strip().lower() not in (
    "0", "false", "no", "off", "python")
# Times a native call declined (whole call) or handed items back to numpy; for checks.
native_fallbacks = {"decode_call": 0, "decode_instr": 0, "score_call": 0, "ari_call": 0,
                    "cluster_call": 0, "cluster_window": 0}


def native_available() -> bool:
    """Whether the C++ extension is built and importable."""
    return _native_mod is not None


def set_native(on: bool) -> None:
    """Use (True) or bypass (False) the C++ extension in this process."""
    global _use_native
    _use_native = bool(on)


def _native() -> Any:
    return _native_mod if _use_native else None

__all__ = [
    "CODE_TYPES", "FIELD_TYPE_NAMES", "MAX_BITS", "SCORED_TYPES", "MetricAccumulator",
    "adjusted_rand_index", "clusters_from_embeddings", "connected_components", "decode_fields",
    "render_markdown", "score_batch", "threshold_clusters",
]

TYPE_CLASSES = 5
FIXED, NOT_CODE = 0, 4
FIELD_TYPE_NAMES = ("fixed", "register", "immediate", "other_variable", "not_code")
SCORED_TYPES = (0, 1, 2, 4)
CODE_TYPES = (0, 1, 2)
PARCEL_BITS = 16
MAX_PARCELS = 4
MAX_BITS = PARCEL_BITS * MAX_PARCELS
_NONE = MAX_BITS          # canonical label of a bit outside the instruction
_CHUNK = 2048             # instructions per vectorized block


def _np(x: Any) -> np.ndarray:
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _labels(x: Any, classes: int, ndim: int) -> np.ndarray:
    """Class indices from indices (``ndim`` dims) or logits (one more trailing dim of ``classes``)."""
    a = _np(x)
    if a.ndim == ndim + 1:
        if a.shape[-1] != classes:
            raise ValueError(f"logits last dim {a.shape[-1]}, expected {classes}")
        return a.argmax(axis=-1).astype(np.int64)
    if a.ndim != ndim:
        raise ValueError(f"expected {ndim} dims (or {ndim + 1} with logits), got shape {a.shape}")
    return a.astype(np.int64)


# --------------------------------------------------------------------------- components

def connected_components(adj: np.ndarray) -> np.ndarray:
    """Connected components of graphs given as ``...×n×n`` boolean adjacency (made symmetric).

    Returns ``...×n`` int64: each node's label is the smallest node index in its component.
    """
    a = np.asarray(adj, dtype=bool)
    n = a.shape[-1]
    a = a | np.swapaxes(a, -1, -2) | np.eye(n, dtype=bool)
    label = np.broadcast_to(np.arange(n, dtype=np.int64), a.shape[:-1]).copy()
    while True:
        new = np.where(a, label[..., None, :], n).min(axis=-1)
        if np.array_equal(new, label):
            return label
        label = new


# --------------------------------------------------------------------------- instructions

def _instruction_rows(instr_row: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(window, start row, visible rows) of every instruction in an ``instr_row`` assignment."""
    b, rows = instr_row.shape
    w, r = np.nonzero(instr_row >= 0)
    key = w * rows + instr_row[w, r]
    uniq, counts = np.unique(key, return_counts=True)
    return uniq // rows, uniq % rows, counts


def _gather(per_bit: np.ndarray, w: np.ndarray, s: np.ndarray, nrows: np.ndarray,
            fill: int) -> tuple[np.ndarray, np.ndarray]:
    """n×64 values of ``per_bit`` (B×R×16 or B×R×16×k) over rows s..s+nrows-1, ``fill`` past them."""
    rows = per_bit.shape[1]
    k = np.arange(MAX_PARCELS)
    r = np.minimum(s[:, None] + k, rows - 1)
    valid_row = k < nrows[:, None]
    vals = per_bit[w[:, None], r]                                       # n×4×16(×k)
    valid = np.repeat(valid_row, PARCEL_BITS, axis=1)                   # n×64
    vals = vals.reshape(len(w), MAX_BITS, *per_bit.shape[3:])
    mask = valid.reshape(valid.shape + (1,) * (vals.ndim - 2))
    return np.where(mask, vals, fill), valid


def decode_fields(grouping: Any, instr_row: Any, threshold: float = 1.0) -> np.ndarray:
    """Per-bit field labels from grouping vectors, within each instruction of ``instr_row``.

    Args:
        grouping: B×R×4×4×G grouping vectors.
        instr_row: B×R start row of the instruction owning each row, -1 for none (the batch's
            ``instr_row`` for oracle boundaries, or a decoded one).
        threshold: Bits closer than this (Euclidean) are joined; fields are the connected
            components of that graph within one instruction.

    Returns:
        B×R×4×4 int64: the smallest in-instruction bit index (16 * row offset + parcel bit) of
        each bit's field, -1 on rows owned by no instruction.
    """
    g = _np(grouping).astype(np.float32)
    ir = _np(instr_row).astype(np.int64)
    b, rows = ir.shape
    out = np.full((b, rows, PARCEL_BITS), -1, dtype=np.int64)
    w, s, nrows = _instruction_rows(ir)
    flat = g.reshape(b, rows, PARCEL_BITS, -1)
    labels = _decode_native(flat, ir, w, s, nrows, threshold)
    for lo in range(0, len(w), _CHUNK):
        sl = slice(lo, lo + _CHUNK)
        lab = (_decode_chunk(flat, w[sl], s[sl], nrows[sl], threshold) if labels is None
               else labels[sl])
        for k in range(MAX_PARCELS):
            ok = k < nrows[sl]
            out[w[sl][ok], s[sl][ok] + k] = lab[ok, PARCEL_BITS * k:PARCEL_BITS * (k + 1)]
    return out.reshape(b, rows, 4, 4)


def _decode_chunk(flat: np.ndarray, w: np.ndarray, s: np.ndarray, nrows: np.ndarray,
                  threshold: float) -> np.ndarray:
    """n×64 component labels of instructions (at most ``_CHUNK``), the numpy reference."""
    vec, valid = _gather(flat, w, s, nrows, 0)
    sq = (vec * vec).sum(-1)
    d2 = sq[:, :, None] + sq[:, None, :] - 2 * np.einsum("nig,njg->nij", vec, vec)
    adj = (d2 < threshold * threshold) & valid[:, :, None] & valid[:, None, :]
    return connected_components(adj)


def _decode_native(flat: np.ndarray, ir: np.ndarray, w: np.ndarray, s: np.ndarray,
                   nrows: np.ndarray, threshold: float) -> np.ndarray | None:
    """n×64 labels as ``_decode_chunk`` gives them, from the C++ core; None to decline."""
    nat = _native()
    if nat is None or len(w) == 0:
        return None
    rows = ir.shape[1]
    # Inputs the reference handles in its own way (aliasing keys, IndexError): leave to it.
    if ir.max() >= rows or (s + np.minimum(nrows, MAX_PARCELS) > rows).any():
        native_fallbacks["decode_call"] += 1
        return None
    lab = np.empty((len(w), MAX_BITS), dtype=np.int64)
    amb = np.zeros(len(w), dtype=bool)
    if nat.decode_labels(np.ascontiguousarray(flat), np.ascontiguousarray(w),
                         np.ascontiguousarray(s), np.ascontiguousarray(nrows), float(threshold),
                         lab, amb) is None:
        native_fallbacks["decode_call"] += 1
        return None
    redo = np.flatnonzero(amb)
    if len(redo):
        native_fallbacks["decode_instr"] += len(redo)
        for lo in range(0, len(redo), _CHUNK):
            r = redo[lo:lo + _CHUNK]
            lab[r] = _decode_chunk(flat, w[r], s[r], nrows[r], threshold)
    return lab


def _canon(lab: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """n×64: index of the first bit carrying the same label (``_NONE`` outside the instruction)."""
    eq = (lab[:, :, None] == lab[:, None, :]) & valid[:, None, :]
    first = eq.argmax(axis=2)
    return np.where(valid, first, _NONE)


def _majority(canon: np.ndarray, ptype: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """n×64: the majority predicted type of each bit's field (ties -> lowest type)."""
    n = len(canon)
    i = np.repeat(np.arange(n), MAX_BITS).reshape(n, MAX_BITS)
    idx = ((i * (MAX_BITS + 1) + canon) * TYPE_CLASSES + ptype)[valid]
    counts = np.bincount(idx, minlength=n * (MAX_BITS + 1) * TYPE_CLASSES)
    maj = counts.reshape(n, MAX_BITS + 1, TYPE_CLASSES).argmax(axis=2)
    return np.where(valid, np.take_along_axis(maj, canon, axis=1), NOT_CODE)


def _merge_fixed(canon: np.ndarray, ftype: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Labels with every fixed-typed bit of an instruction moved into one field."""
    return np.where(valid & (ftype == FIXED), -1, canon)


def _field_scores(ct: np.ndarray, tt: np.ndarray, cp: np.ndarray, pt: np.ndarray
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per instruction: (true-positive fields, predicted fields, true fields).

    ``ct``/``cp`` canonical labels, ``tt`` true types, ``pt`` predicted field types (per bit).
    """
    pos = np.arange(MAX_BITS)
    t_start = ct == pos
    p_start = cp == pos
    in_t = ct[:, None, :] == pos[None, :, None]                          # n×c×j
    in_p = cp[:, None, :] == pos[None, :, None]
    same = (in_t == in_p).all(axis=2)
    match = t_start & same & (tt == pt)
    return match.sum(1), p_start.sum(1), t_start.sum(1)


def _layout_keys(length: np.ndarray, canon: np.ndarray, ftype: np.ndarray) -> np.ndarray:
    """Hashable (void) row per instruction: length, canonical partition, per-bit types."""
    arr = np.concatenate([length[:, None], canon, ftype], axis=1).astype(np.uint8)
    arr = np.ascontiguousarray(arr)
    return arr.view(np.dtype((np.void, arr.shape[1]))).ravel()


def _votes(group: np.ndarray, true_id: np.ndarray, pred_id: np.ndarray
           ) -> tuple[int, int, int, int, int]:
    """Exact-format votes. Returns (groups, correct, groups >= 2, correct >= 2, layout conflicts).

    Per group: true layout = most common true id; predicted = most common predicted id, which must
    be strictly more common than any other.
    """
    if len(group) == 0:
        return 0, 0, 0, 0, 0
    _, sizes = np.unique(group, return_counts=True)
    _, t_mode, _ = _mode(group, true_id)
    _, p_mode, p_strict = _mode(group, pred_id)
    correct = p_strict & (p_mode == t_mode)
    multi = sizes >= 2
    conflicts = int(len(np.unique(np.stack([group, true_id], 1), axis=0)) - len(sizes))
    return (len(sizes), int(correct.sum()), int(multi.sum()), int((correct & multi).sum()),
            conflicts)


def _mode(group: np.ndarray, ids: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per group (ascending): (group, most common id, whether it is strictly the most common).

    Among tied ids the smallest is returned, with ``strict`` False.
    """
    pairs, counts = np.unique(np.stack([group, ids], 1), axis=0, return_counts=True)
    order = np.lexsort((pairs[:, 1], -counts, pairs[:, 0]))
    pairs, counts = pairs[order], counts[order]
    first = np.ones(len(pairs), dtype=bool)
    first[1:] = pairs[1:, 0] != pairs[:-1, 0]
    top = np.flatnonzero(first)
    nxt = np.minimum(top + 1, len(pairs) - 1)
    tie = (top + 1 < len(pairs)) & ~first[nxt] & (counts[nxt] == counts[top])
    return pairs[top, 0], pairs[top, 1], ~tie


# --------------------------------------------------------------------------- clustering

def adjusted_rand_index(a: Any, b: Any) -> float:
    """Adjusted Rand index of two labelings of the same items (1 = identical, 0 = chance).

    Two identical trivial partitions (both one cluster, or both all singletons) give 1.0; fewer
    than 2 items give 1.0.
    """
    x, y = _np(a).ravel(), _np(b).ravel()
    if len(x) != len(y):
        raise ValueError("labelings differ in length")
    n = len(x)
    if n < 2:
        return 1.0
    _, xi = np.unique(x, return_inverse=True)
    _, yi = np.unique(y, return_inverse=True)
    _, nij = np.unique(xi.astype(np.int64) * (yi.max() + 1) + yi, return_counts=True)

    def c2(v: np.ndarray | int) -> float:
        v = np.asarray(v, dtype=np.float64)
        return float((v * (v - 1) / 2).sum())

    index = c2(nij)
    sa, sb = c2(np.bincount(xi)), c2(np.bincount(yi))
    expected = sa * sb / c2(n)
    top = (sa + sb) / 2
    if top == expected:
        return 1.0
    return (index - expected) / (top - expected)


def threshold_clusters(emb: Any, min_cos: float = 0.5) -> np.ndarray:
    """Clusters of unit vectors (N×D): connected components of cosine similarity >= ``min_cos``.

    Deterministic and label-free. Returns N int64 labels (smallest member index).
    """
    z = _np(emb).astype(np.float64)
    if len(z) == 0:
        return np.zeros(0, dtype=np.int64)
    z = z / np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-12)
    return connected_components((z @ z.T) >= min_cos)


def clusters_from_embeddings(emb: Any, win: Any, row: Any, shape: tuple[int, int],
                             cluster_fn: Callable[[np.ndarray], np.ndarray] = threshold_clusters
                             ) -> np.ndarray:
    """B×R cluster ids from per-instruction embeddings, clustered within each window.

    ``emb``, ``win``, ``row``: the model's ``instr_embed``, ``instr_batch``, ``instr_start``.
    ``cluster_fn`` maps one window's N×D embeddings to N labels. Rows without an embedding get -1.
    """
    z, wi, ri = _np(emb), _np(win).astype(np.int64), _np(row).astype(np.int64)
    out = np.full(shape, -1, dtype=np.int64)
    windows = np.unique(wi)
    nat = _native()
    if (nat is not None and cluster_fn is threshold_clusters and z.ndim == 2
            and z.dtype in (np.float32, np.float64) and len(wi) == len(z) == len(ri)):
        left = nat.cluster_windows(np.ascontiguousarray(z), np.ascontiguousarray(wi),
                                   np.ascontiguousarray(ri), out, 0.5)
        if left is None:
            native_fallbacks["cluster_call"] += 1
        else:
            native_fallbacks["cluster_window"] += len(left)
            windows = np.asarray(left, dtype=np.int64)
    for w in windows:
        sel = wi == w
        out[w, ri[sel]] = np.asarray(cluster_fn(z[sel]), dtype=np.int64)
    return out


# --------------------------------------------------------------------------- scoring

def _type_counts(conf: np.ndarray) -> dict[str, Any]:
    out: dict[str, Any] = {"per_class": {}}
    total = conf.sum()
    out["accuracy"] = float(np.trace(conf) / total) if total else None
    f1s = {}
    for k in SCORED_TYPES:
        tp = conf[k, k]
        pred, true = conf[:, k].sum(), conf[k].sum()
        p = tp / pred if pred else 0.0
        r = tp / true if true else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        f1s[k] = f
        out["per_class"][FIELD_TYPE_NAMES[k]] = {
            "precision": float(p), "recall": float(r), "f1": float(f), "support": int(true),
            "predicted": int(pred)}
    # A class with no true and no predicted bits has no F1; it is left out of the average.
    present = {k for k in SCORED_TYPES if conf[k].sum() or conf[:, k].sum()}
    for name, classes in (("macro_f1", SCORED_TYPES), ("macro_f1_code", CODE_TYPES)):
        vals = [f1s[k] for k in classes if k in present]
        out[name] = float(np.mean(vals)) if vals else None
    out["bits"] = int(total)
    return out


def score_batch(batch: Mapping[str, Any], pred_type: Any, pred_field: Any,
                pred_length: Any | None = None, pred_cluster: Any | None = None
                ) -> dict[str, Any]:
    """Raw counts for one batch (all windows pooled). ``MetricAccumulator`` turns them into rates.

    Keys: ``confusion`` (5×5, true × predicted, all bits), ``field`` / ``field_fixed_merged``
    (tp, pred, true), ``instr_exact`` (instructions whose whole layout is right), ``exact_format``,
    ``exact_format_fixed_merged``, ``exact_format_by_class`` (formats, correct, formats_multi,
    correct_multi, conflicts), ``length`` (correct, n) or None, ``ari`` (sum, windows, skipped)
    or None, ``instructions``, ``cut_excluded``, ``windows``.
    """
    ft = _np(batch["field_type"]).astype(np.int64)
    b, rows = ft.shape[:2]
    ft = ft.reshape(b, rows, PARCEL_BITS)
    pt = _labels(pred_type, TYPE_CLASSES, 4).reshape(b, rows, PARCEL_BITS)
    fid = _np(batch["field_id"]).astype(np.int64).reshape(b, rows, PARCEL_BITS)
    pf = _np(pred_field).astype(np.int64).reshape(b, rows, PARCEL_BITS)
    start = _np(batch["start"]).astype(np.int64)
    length = _np(batch["length"]).astype(np.int64)
    cut = _np(batch["cut"]).astype(bool)
    klass = _np(batch["instr_class"]).astype(np.int64)

    conf = np.bincount((ft * TYPE_CLASSES + pt).ravel(),
                       minlength=TYPE_CLASSES * TYPE_CLASSES).reshape(TYPE_CLASSES, TYPE_CLASSES)

    is_start = start == 1
    w, s = np.nonzero(is_start & ~cut)
    n_cut = int((is_start & cut).sum())
    L = length[w, s]
    out: dict[str, Any] = {"confusion": conf, "windows": b, "instructions": len(w),
                           "cut_excluded": n_cut}

    nat = _native()
    res = None
    if nat is not None and len(w):
        res = nat.score_instructions(np.ascontiguousarray(ft), np.ascontiguousarray(fid),
                                     np.ascontiguousarray(pt), np.ascontiguousarray(pf),
                                     np.ascontiguousarray(w), np.ascontiguousarray(s),
                                     np.ascontiguousarray(L), np.ascontiguousarray(klass[w, s]))
        if res is None:
            native_fallbacks["score_call"] += 1
    if res is not None:
        (tp0, tp1), (np0, np1), (nt0, nt1), instr_exact, ef, efm, efc = res
        out["field"] = {"tp": tp0, "pred": np0, "true": nt0}
        out["field_fixed_merged"] = {"tp": tp1, "pred": np1, "true": nt1}
        out["instr_exact"] = instr_exact
        out["exact_format"] = ef
        out["exact_format_by_class"] = efc
        out["exact_format_fixed_merged"] = efm
    else:
        _score_instructions_py(out, ft, fid, pt, pf, w, s, L, klass)

    if pred_length is None:
        out["length"] = None
    else:
        pl = _labels(pred_length, MAX_PARCELS, 2)
        if pl.ndim == 2 and _np(pred_length).ndim == 3:
            pl = pl + 1                                   # logits index 0..3 -> 1..4 parcels
        out["length"] = (int((pl[w, s] == L).sum()), len(w))

    if pred_cluster is None:
        out["ari"] = None
    else:
        pc = _np(pred_cluster).astype(np.int64)
        ari = None
        if nat is not None:
            ari = nat.ari_sum(np.ascontiguousarray(pc[w, s]), np.ascontiguousarray(klass[w, s]),
                              np.ascontiguousarray(w), b)
            if ari is None:
                native_fallbacks["ari_call"] += 1
        if ari is None:
            total, used, skipped = 0.0, 0, 0
            for win in range(b):
                sel = w == win
                if sel.sum() < 2:
                    skipped += 1
                    continue
                total += adjusted_rand_index(pc[win, s[sel]], klass[win, s[sel]])
                used += 1
            ari = (total, used, skipped)
        out["ari"] = ari
    return {k: out[k] for k in _SCORE_ORDER}


_SCORE_ORDER = ("confusion", "windows", "instructions", "cut_excluded", "field",
                "field_fixed_merged", "instr_exact", "exact_format", "exact_format_by_class",
                "exact_format_fixed_merged", "length", "ari")


def _score_instructions_py(out: dict[str, Any], ft: np.ndarray, fid: np.ndarray, pt: np.ndarray,
                           pf: np.ndarray, w: np.ndarray, s: np.ndarray, L: np.ndarray,
                           klass: np.ndarray) -> None:
    """The instruction-level counts of ``score_batch`` (numpy reference), into ``out``."""
    tp = np.zeros(2, dtype=np.int64)
    npred = np.zeros(2, dtype=np.int64)
    ntrue = np.zeros(2, dtype=np.int64)
    instr_exact = 0
    keys_t: list[list[np.ndarray]] = [[], []]
    keys_p: list[list[np.ndarray]] = [[], []]
    for lo in range(0, len(w), _CHUNK):
        sl = slice(lo, lo + _CHUNK)
        tt, valid = _gather(ft, w[sl], s[sl], L[sl], NOT_CODE)
        tid, _ = _gather(fid, w[sl], s[sl], L[sl], -1)
        pty, _ = _gather(pt, w[sl], s[sl], L[sl], NOT_CODE)
        pfl, _ = _gather(pf, w[sl], s[sl], L[sl], -1)
        ct = _canon(tid, valid)
        cp = _canon(pfl, valid)
        ptf = _majority(cp, pty, valid)
        for v in range(2):
            if v == 1:
                ct = _canon(_merge_fixed(ct, tt, valid), valid)
                cp = _canon(_merge_fixed(cp, ptf, valid), valid)
            t, p, n = _field_scores(ct, tt, cp, ptf)
            tp[v] += t.sum()
            npred[v] += p.sum()
            ntrue[v] += n.sum()
            if v == 0:
                instr_exact += int(((t == n) & (p == n)).sum())
            keys_t[v].append(_layout_keys(L[sl], ct, tt))
            keys_p[v].append(_layout_keys(L[sl], cp, ptf))

    for v, name in ((0, "field"), (1, "field_fixed_merged")):
        out[name] = {"tp": int(tp[v]), "pred": int(npred[v]), "true": int(ntrue[v])}
    out["instr_exact"] = instr_exact

    for v, name in ((0, "exact_format"), (1, "exact_format_fixed_merged")):
        if len(w):
            kt, kp = np.concatenate(keys_t[v]), np.concatenate(keys_p[v])
            _, ids = np.unique(np.concatenate([kt, kp]), return_inverse=True)
            t_id, p_id = ids[:len(w)], ids[len(w):]
        else:
            t_id = p_id = np.zeros(0, dtype=np.int64)
        out[name] = _votes(w * (1 << 20) + t_id, t_id, p_id)
        if v == 0:
            out["exact_format_by_class"] = _votes(w * (1 << 20) + klass[w, s], t_id, p_id)


# --------------------------------------------------------------------------- accumulation

def _prf(tp: int, pred: int, true: int) -> dict[str, Any]:
    p = tp / pred if pred else 0.0
    r = tp / true if true else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": p, "recall": r, "f1": f, "tp": tp, "pred": pred, "true": true}


def _fmt(t: tuple[int, ...]) -> dict[str, Any]:
    formats, correct, multi, correct_multi, conflicts = t
    return {"rate": correct / formats if formats else None, "formats": formats, "correct": correct,
            "rate_multi": correct_multi / multi if multi else None, "formats_multi": multi,
            "correct_multi": correct_multi, "true_layout_conflicts": conflicts}


class MetricAccumulator:
    """Collects ``score_batch`` counts per ladder level over many batches.

    ``add`` splits a batch by ``batch["level"]``. ``result()`` returns
    ``{"L0": {metric: {...values and counts}}, ...}`` for the levels seen, plus ``"boundaries":
    "oracle"`` in each level (instruction-level metrics use true starts and lengths).
    """

    def __init__(self) -> None:
        self.levels: dict[int, dict[str, Any]] = {}

    def add(self, batch: Mapping[str, Any], pred_type: Any, pred_field: Any,
            pred_length: Any | None = None, pred_cluster: Any | None = None) -> None:
        lv = _np(batch["level"]).astype(np.int64).ravel()
        for level in np.unique(lv):
            ix = np.flatnonzero(lv == level)
            sub = {k: _np(v)[ix] for k, v in batch.items()}
            pick = lambda x: None if x is None else _np(x)[ix]  # noqa: E731
            self.add_counts(int(level), score_batch(sub, pick(pred_type), pick(pred_field),
                                                    pick(pred_length), pick(pred_cluster)))

    def add_counts(self, level: int, counts: Mapping[str, Any]) -> None:
        """Merge raw ``score_batch`` counts (e.g. from a worker process) into ``level``."""
        acc = self.levels.get(level)
        if acc is None:
            self.levels[level] = {k: (v.copy() if isinstance(v, np.ndarray) else
                                      (dict(v) if isinstance(v, dict) else v))
                                  for k, v in counts.items()}
            return
        for k, v in counts.items():
            if v is None or acc[k] is None:
                acc[k] = None                    # a metric missing from any batch is n/a
            elif isinstance(v, dict):
                acc[k] = {kk: acc[k][kk] + vv for kk, vv in v.items()}
            elif isinstance(v, tuple):
                acc[k] = tuple(a + c for a, c in zip(acc[k], v))
            else:
                acc[k] = acc[k] + v

    def result(self) -> dict[str, dict[str, Any]]:
        out = {}
        for level in sorted(self.levels):
            a = self.levels[level]
            r: dict[str, Any] = {"boundaries": "oracle", "windows": int(a["windows"]),
                                 "instructions": int(a["instructions"]),
                                 "cut_excluded": int(a["cut_excluded"])}
            r["field_type"] = _type_counts(a["confusion"])
            r["field_type"]["confusion"] = a["confusion"].tolist()
            r["field"] = _prf(**a["field"])
            r["field_fixed_merged"] = _prf(**a["field_fixed_merged"])
            n = a["instructions"]
            r["instr_exact"] = {"rate": a["instr_exact"] / n if n else None,
                                "correct": int(a["instr_exact"]), "n": int(n)}
            r["exact_format"] = _fmt(a["exact_format"])
            r["exact_format_fixed_merged"] = _fmt(a["exact_format_fixed_merged"])
            r["exact_format_by_class"] = _fmt(a["exact_format_by_class"])
            if a["length"] is None:
                r["length"] = None
            else:
                c, m = a["length"]
                r["length"] = {"accuracy": c / m if m else None, "correct": int(c), "n": int(m)}
            if a["ari"] is None:
                r["clustering"] = None
            else:
                total, used, skipped = a["ari"]
                r["clustering"] = {"ari": total / used if used else None, "windows": int(used),
                                   "skipped_windows": int(skipped)}
            out[f"L{level}"] = r
        return out


# --------------------------------------------------------------------------- report

def _cell(x: Any, digits: int = 3) -> str:
    return "n/a" if x is None else f"{x:.{digits}f}"


TABLE_COLUMNS = (
    ("Level", None), ("Windows", None), ("Type acc", ("field_type", "accuracy")),
    ("Type macro-F1 (4)", ("field_type", "macro_f1")),
    ("Type macro-F1 (code 3)", ("field_type", "macro_f1_code")),
    ("Field P", ("field", "precision")), ("Field R", ("field", "recall")),
    ("Field F1", ("field", "f1")), ("Exact-format", ("exact_format", "rate")),
    ("Exact-format (≥2)", ("exact_format", "rate_multi")),
    ("Exact-format by class", ("exact_format_by_class", "rate")),
    ("Field F1 (fixed merged)", ("field_fixed_merged", "f1")),
    ("Exact-format (fixed merged)", ("exact_format_fixed_merged", "rate")),
    ("Length acc", ("length", "accuracy")), ("ARI", ("clustering", "ari")),
)


def render_markdown(results: Mapping[str, Mapping[str, Any]], title: str | None = None,
                    notes: Mapping[str, str] | None = None) -> str:
    """Markdown table, one row per level of ``MetricAccumulator.result()``.

    ``notes`` maps a level key ("L0") to a note appended to its Level cell (e.g. the source split).
    Missing metrics show "n/a".
    """
    lines = []
    if title:
        lines += [f"### {title}", ""]
    lines.append("Instruction-level metrics (field, exact-format, length, ARI) use oracle (true) "
                 "starts and lengths; boundary F1 and shift consistency are not computed.")
    lines.append("")
    lines.append("| " + " | ".join(name for name, _ in TABLE_COLUMNS) + " |")
    lines.append("|" + "|".join("---" if path is None and name == "Level" else "---:"
                                for name, path in TABLE_COLUMNS) + "|")
    for level, r in results.items():
        cells = []
        for name, path in TABLE_COLUMNS:
            if name == "Level":
                note = (notes or {}).get(level)
                cells.append(f"{level} ({note})" if note else level)
            elif name == "Windows":
                cells.append(str(r["windows"]))
            else:
                section = r.get(path[0])
                cells.append(_cell(None if section is None else section.get(path[1])))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"
