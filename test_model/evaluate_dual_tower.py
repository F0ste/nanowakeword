"""
Dual-Tower Confidence Analysis
===============================
Compares the lite (gate) model against the full (verifier) model on
positive and negative samples, then evaluates cascade-mode metrics:

  - Per-model miss rate / false-alarm rate (over a sweep of thresholds)
  - Gate filter efficiency (what % of frames are blocked by the gate)
  - Cascade combined error rates
  - Score distribution overlap between positive and negative samples
"""

import os
import sys
import numpy as np
from tqdm import tqdm
from tabulate import tabulate
import onnxruntime as ort

# ── Config ──────────────────────────────────────────────────────────
LITE_MODEL    = "trained_models/hi_hotel_multilang_10k/model/hi_hotel_multilang_10k_lite.onnx"
FULL_MODEL    = "trained_models/hi_hotel_multilang_10k/model/hi_hotel_multilang_10k.onnx"
GATE_THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]  # thresholds to sweep
DETECT_THRESHOLD = 0.5  # fixed verifier threshold for detection

POSITIVE_NPY = "trained_models/hi_hotel_multilang_10k/features/positive_features.npy"
NEGATIVE_NPY = "trained_models/hi_hotel_multilang_10k/features/negative_features.npy"
HARD_NEG_NPY = "trained_models/hi_hotel_multilang_10k/features/hard_negative_features.npy"

BATCH_SIZE  = 256
MAX_SAMPLES = None  # set to an int to limit for quick iteration
# ────────────────────────────────────────────────────────────────────


def load_serial(path, limit=None):
    data = np.load(path)
    if limit:
        data = data[:limit]
    return data.astype(np.float32)


def predict_all(session, data, batch_size):
    """Run a stateless ONNX model over all samples, return 1-D score array."""
    input_name = session.get_inputs()[0].name
    scores_list = []
    for i in tqdm(range(0, len(data), batch_size),
                  desc="  Scoring", leave=False):
        batch = data[i : i + batch_size]
        out = session.run(None, {input_name: batch})[0]
        scores_list.append(out.squeeze())
    return np.concatenate(scores_list)


def load_session(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found: {path}")
    return ort.InferenceSession(path, providers=["CPUExecutionProvider"])


def compute_metrics(scores, labels, threshold):
    """labels: 1 = positive, 0 = negative. Returns (misses, false_alarms, total_pos, total_neg)."""
    pos_mask = labels == 1
    neg_mask = labels == 0
    misses      = int(np.sum(scores[pos_mask] < threshold))
    false_alarms = int(np.sum(scores[neg_mask] > threshold))
    return misses, false_alarms, int(pos_mask.sum()), int(neg_mask.sum())


# ── Load data ───────────────────────────────────────────────────────
print("=" * 60)
print("  Dual-Tower Confidence Analysis")
print("=" * 60)

print("\n[1] Loading feature data...")
pos_data = load_serial(POSITIVE_NPY, MAX_SAMPLES)
neg_data = load_serial(NEGATIVE_NPY, MAX_SAMPLES)
hard_neg_data = load_serial(HARD_NEG_NPY, MAX_SAMPLES)
print(f"  Positives     : {len(pos_data):,}")
print(f"  Negatives     : {len(neg_data):,}")
print(f"  Hard negatives: {len(hard_neg_data):,}")

# ── Load models ─────────────────────────────────────────────────────
print("\n[2] Loading models...")
lite_sess = load_session(LITE_MODEL)
full_sess = load_session(FULL_MODEL)
print(f"  Lite: {os.path.basename(LITE_MODEL)}")
print(f"  Full: {os.path.basename(FULL_MODEL)}")

# ── Score all samples ───────────────────────────────────────────────
print("\n[3] Scoring all samples (this may take a moment)...")

# Positive scores
print("  Lite on positives...")
lite_pos = predict_all(lite_sess, pos_data, BATCH_SIZE)
print("  Full on positives...")
full_pos = predict_all(full_sess, pos_data, BATCH_SIZE)

# Negative scores (speech negatives only)
print("  Lite on negatives...")
lite_neg = predict_all(lite_sess, neg_data, BATCH_SIZE)
print("  Full on negatives...")
full_neg = predict_all(full_sess, neg_data, BATCH_SIZE)

# Hard negative scores
print("  Lite on hard negatives...")
lite_hard = predict_all(lite_sess, hard_neg_data, BATCH_SIZE)
print("  Full on hard negatives...")
full_hard = predict_all(full_sess, hard_neg_data, BATCH_SIZE)

# ── Per-model evaluation at various thresholds ──────────────────────
print("\n[4] Per-model evaluation (varying detection threshold)...")

results = []
for thr in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    # Pos + regular neg
    lite_m, lite_fa, *_ = compute_metrics(
        np.concatenate([lite_pos, lite_neg]),
        np.concatenate([np.ones(len(lite_pos)), np.zeros(len(lite_neg))]),
        thr,
    )
    full_m, full_fa, tp, tn = compute_metrics(
        np.concatenate([full_pos, full_neg]),
        np.concatenate([np.ones(len(full_pos)), np.zeros(len(full_neg))]),
        thr,
    )
    # Pos + hard neg
    lite_hm, lite_hfa, *_ = compute_metrics(
        np.concatenate([lite_pos, lite_hard]),
        np.concatenate([np.ones(len(lite_pos)), np.zeros(len(lite_hard))]),
        thr,
    )
    full_hm, full_hfa, *_ = compute_metrics(
        np.concatenate([full_pos, full_hard]),
        np.concatenate([np.ones(len(full_pos)), np.zeros(len(full_hard))]),
        thr,
    )

    results.append([
        thr,
        f"{lite_m}/{tp} ({100*lite_m/tp:.1f}%)",
        f"{full_m}/{tp} ({100*full_m/tp:.1f}%)",
        f"{lite_fa}/{tn} ({100*lite_fa/tn:.1f}%)",
        f"{full_fa}/{tn} ({100*full_fa/tn:.1f}%)",
        f"{lite_hfa}/{len(lite_hard)} ({100*lite_hfa/len(lite_hard):.1f}%)",
        f"{full_hfa}/{len(full_hard)} ({100*full_hfa/len(full_hard):.1f}%)",
    ])

print(tabulate(results, headers=[
    "Thr", "Lite Miss", "Full Miss",
    "Lite FA (speech)", "Full FA (speech)",
    "Lite FA (hard)", "Full FA (hard)",
], tablefmt="pretty", stralign="center"))

# ── Cascade analysis ────────────────────────────────────────────────
print("\n[5] Cascade analysis (gate filter + verifier re-check)...")

cascade_table = []
for gate_thr in GATE_THRESHOLDS:
    # How many frames pass the gate?
    pos_pass = (lite_pos >= gate_thr).sum()
    neg_pass = (lite_neg >= gate_thr).sum()
    hard_pass = (lite_hard >= gate_thr).sum()

    # Among passed frames, how many does the verifier correctly classify?
    # Verifier uses DETECT_THRESHOLD for final decision
    pos_pass_mask = lite_pos >= gate_thr
    neg_pass_mask = lite_neg >= gate_thr
    hard_pass_mask = lite_hard >= gate_thr

    # Positives that pass gate AND are detected by verifier
    cascade_detected = int((full_pos[pos_pass_mask] >= DETECT_THRESHOLD).sum())
    cascade_missed   = int(pos_pass_mask.sum()) - cascade_detected + int((~pos_pass_mask).sum())

    # Negatives that pass gate AND false-trigger verifier
    cascade_fa = int((full_neg[neg_pass_mask] >= DETECT_THRESHOLD).sum())

    # Hard negatives
    cascade_hfa = int((full_hard[hard_pass_mask] >= DETECT_THRESHOLD).sum())

    # Verifier compute saved (negatives that never reach verifier)
    verifier_saved_pct = 100.0 * (len(lite_neg) - neg_pass) / len(lite_neg)

    cascade_table.append([
        gate_thr,
        f"{pos_pass}/{len(lite_pos)} ({100*pos_pass/len(lite_pos):.1f}%)",
        f"{neg_pass}/{len(lite_neg)} ({100*neg_pass/len(lite_neg):.0f}%)",
        f"{hard_pass}/{len(lite_hard)} ({100*hard_pass/len(lite_hard):.0f}%)",
        f"{cascade_missed}/{len(lite_pos)} ({100*cascade_missed/len(lite_pos):.1f}%)",
        f"{cascade_fa}/{len(lite_neg)} ({100*cascade_fa/len(lite_neg):.1f}%)",
        f"{cascade_hfa}/{len(lite_hard)} ({100*cascade_hfa/len(lite_hard):.1f}%)",
        f"{verifier_saved_pct:.1f}%",
    ])

print(tabulate(cascade_table, headers=[
    "Gate Thr", "Pos Pass Gate", "Neg Pass Gate", "Hard Pass Gate",
    "Cascade Miss", "Cascade FA", "Cascade H-FA",
    "Verifier Saved",
], tablefmt="pretty", stralign="center"))

# ── Distribution summary ────────────────────────────────────────────
print("\n[6] Score distribution summary...")

def describe(arr, label):
    return [
        label,
        f"{arr.min():.4f}",
        f"{np.percentile(arr, 25):.4f}",
        f"{np.median(arr):.4f}",
        f"{np.percentile(arr, 75):.4f}",
        f"{arr.max():.4f}",
        f"{arr.mean():.4f}",
        f"{arr.std():.4f}",
    ]

dist_table = [
    describe(lite_pos, "Lite-Pos"),
    describe(full_pos, "Full-Pos"),
    describe(lite_neg, "Lite-Neg"),
    describe(full_neg, "Full-Neg"),
    describe(lite_hard, "Lite-Hard"),
    describe(full_hard, "Full-Hard"),
]
print(tabulate(dist_table, headers=[
    "Group", "Min", "P25", "P50", "P75", "Max", "Mean", "Std",
], tablefmt="pretty", stralign="center"))

# ── Score gap (separability) ────────────────────────────────────────
print("\n[7] Score separability (higher = better separation)...")

def separation(pos_scores, neg_scores):
    """Bhattacharyya-style rough separability: mean_diff / (std_pos + std_neg)."""
    diff = pos_scores.mean() - neg_scores.mean()
    denom = pos_scores.std() + neg_scores.std()
    return diff / denom if denom > 0 else 0.0

lite_sep_speech = separation(lite_pos, lite_neg)
full_sep_speech = separation(full_pos, full_neg)
lite_sep_hard   = separation(lite_pos, lite_hard)
full_sep_hard   = separation(full_pos, full_hard)

print(f"  Lite vs speech neg : {lite_sep_speech:.4f}")
print(f"  Full vs speech neg : {full_sep_speech:.4f}")
print(f"  Lite vs hard neg   : {lite_sep_hard:.4f}")
print(f"  Full vs hard neg   : {full_sep_hard:.4f}")

# ── Overall recommendation ──────────────────────────────────────────
print("\n" + "=" * 60)
print("  RECOMMENDATION")
print("=" * 60)
rec_results = cascade_table  # reuse from section 5
best = min(rec_results, key=lambda r: float(r[5].split("(")[0].split("/")[0]) + float(r[4].split("(")[0].split("/")[0]))
print(f"  Suggested gate_threshold: {best[0]}")
print(f"    -> Verifier compute saved on negatives: {best[-1]}")
print(f"    -> Cascade miss rate:  {best[4]}")
print(f"    -> Cascade FA rate:   {best[5]}")
print("=" * 60)
