"""
Experiment 3b: LM-Head Projection & Recency Control

Test 1: Does the LM head actually USE the path-dependent information?
  - The linear probe finds path info at 100% accuracy throughout all layers.
  - But does that information survive projection through the unembedding matrix?
  - Compute projection_ratio = ||W_U @ dh|| / (||W_U||_F * ||dh||) at every layer
  - Scrub the path direction from layer 32 hidden states and measure Fisher-Rao
    distance between original and scrubbed logits

Test 2: Is the divergence just a recency effect?
  - Insert ~200 tokens of identical neutral text after the washout, before target
  - If divergence persists, it is not just "which topic was most recent"
"""

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import json
import time
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import roc_auc_score

from bank_experiment import (
    FINANCIAL_PARAGRAPHS, NATURE_PARAGRAPHS, WASHOUTS, TARGET, build_context
)

# ---- Configuration ----------------------------------------------------------

MODEL_NAME = "microsoft/phi-2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = Path("/home/anders/consciousness-geometry/results")
PLOT_DIR = OUTPUT_DIR / "bank_experiment_plots"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# ~200 tokens of bland neutral technical text.
# Must NOT mention: banks, finance, rivers, nature, water, money.
COMMON_SUFFIX = (
    "Data serialization converts structured objects into a format suitable for "
    "storage or network transmission. The three most common formats in contemporary "
    "software engineering are JSON, XML, and protocol buffers. Each format offers "
    "distinct tradeoffs between human readability, parsing speed, and payload size. "
    "JSON has become the de facto standard for REST APIs due to its simplicity and "
    "broad language support, though it lacks a native schema definition mechanism.\n\n"
    "Schema evolution presents significant challenges in distributed systems. When "
    "a producer updates its message schema, all downstream consumers must remain "
    "compatible with both old and new formats. Versioned schemas with optional fields "
    "provide one approach to backward compatibility. Another strategy uses tagged "
    "union types with explicit discriminator fields. Protocol buffers handle this "
    "through field numbering, where new fields receive higher numbers and old fields "
    "are never reused after deletion.\n\n"
    "Compression algorithms can be applied after serialization to reduce bandwidth "
    "requirements. Common approaches include gzip, which uses the DEFLATE algorithm "
    "combining LZ77 and Huffman coding, and zstd, which offers better compression "
    "ratios at comparable speeds. The choice of compression algorithm depends on "
    "the specific use case: real-time streaming applications favor speed, while "
    "archival storage prioritizes compression ratio.\n\n"
    "Configuration management in distributed systems requires careful attention to "
    "consistency and propagation delays. Centralized configuration servers maintain "
    "a single source of truth, pushing updates to registered clients through "
    "long-polling or server-sent events. Feature flags enable gradual rollouts "
    "and instant rollbacks without code deployment. Structured logging with "
    "correlation identifiers enables tracing requests across service boundaries."
)


# ---- Utility Functions ------------------------------------------------------

def _find_anchor_pos(input_ids, anchor_ids):
    """Find position of last token of anchor sequence at end of input."""
    ids = input_ids[0].tolist()
    n = len(anchor_ids)
    if ids[-n:] != anchor_ids:
        raise RuntimeError(
            f"Input does not end with expected anchor tokens. "
            f"Expected {anchor_ids}, got {ids[-n:]}"
        )
    return len(ids) - 1


def fisher_rao_distance(p, q):
    """Geodesic distance on probability simplex."""
    bc = torch.sum(torch.sqrt(p.clamp(min=1e-30) * q.clamp(min=1e-30)))
    bc = bc.clamp(min=0.0, max=1.0)
    return 2.0 * torch.arccos(bc).item()


def build_context_with_suffix(para_first, para_second, washout, suffix, target):
    """Build context: para1 + para2 + washout + suffix + target."""
    return f"{para_first}\n\n{para_second}\n\n{washout}\n\n{suffix}\n\n{target}"


def extract_all_hidden_states(model, tokenizer, text, device, anchor_ids):
    """Forward pass, return hidden state vectors at anchor position for all layers.

    Returns list of CPU float32 tensors, one per layer (including embedding layer 0).
    Uses anchor-based position extraction, NOT position -1.
    """
    inputs = tokenizer(text, return_tensors="pt").to(device)
    pos = _find_anchor_pos(inputs.input_ids, anchor_ids)

    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True, use_cache=False)

    hidden_states = []
    for layer_idx in range(len(out.hidden_states)):
        h = out.hidden_states[layer_idx][0, pos, :].float().cpu()
        hidden_states.append(h)

    del out, inputs
    return hidden_states


def run_probe_cv(X, y, groups, c_val=0.1):
    """Run leave-one-group-out CV for logistic regression probe.

    Returns (accuracy, auc).
    """
    logo = LeaveOneGroupOut()
    correct = 0
    total = 0
    all_probs = []
    all_true = []

    for train_idx, test_idx in logo.split(X, y, groups):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        mean = X_train.mean(axis=0)
        std = X_train.std(axis=0) + 1e-8
        X_train_std = (X_train - mean) / std
        X_test_std = (X_test - mean) / std

        clf = LogisticRegression(C=c_val, max_iter=2000, solver="lbfgs")
        clf.fit(X_train_std, y_train)
        preds = clf.predict(X_test_std)
        probs = clf.predict_proba(X_test_std)[:, 1]

        correct += (preds == y_test).sum()
        total += len(y_test)
        all_probs.extend(probs.tolist())
        all_true.extend(y_test.tolist())

    acc = correct / total
    try:
        auc = roc_auc_score(all_true, all_probs)
    except ValueError:
        auc = 0.5
    return acc, auc


# ---- Main -------------------------------------------------------------------

def main():
    print("=" * 70)
    print("EXPERIMENT 3b: LM-Head Projection & Recency Control")
    print("=" * 70)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"\nLoading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if DEVICE == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, torch_dtype=torch.float16,
            device_map="auto", trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, torch_dtype=torch.float32,
            trust_remote_code=True,
        ).to(DEVICE)
    model.eval()
    print("Model loaded.")

    # Anchor tokens for TARGET = "The bank"
    anchor_ids = tokenizer.encode(TARGET, add_special_tokens=False)
    print(f"  Anchor: '{TARGET}' -> {anchor_ids} "
          f"= {[tokenizer.decode([t]) for t in anchor_ids]}")

    # Check COMMON_SUFFIX token count
    suffix_token_count = len(tokenizer.encode(COMMON_SUFFIX, add_special_tokens=False))
    print(f"  COMMON_SUFFIX length: {suffix_token_count} tokens")

    # Get LM head weights
    W_U = model.lm_head.weight.float()  # (vocab_size, hidden_dim)
    has_bias = model.lm_head.bias is not None
    lm_bias = model.lm_head.bias.float() if has_bias else None
    W_U_frobenius = torch.norm(W_U, p='fro').item()
    print(f"  W_U shape: {W_U.shape}, Frobenius norm: {W_U_frobenius:.2f}")
    print(f"  LM head bias: {has_bias}")

    # Copy W_U (and bias) to CPU so we can free GPU memory later
    W_U_cpu = W_U.cpu()
    lm_bias_cpu = lm_bias.cpu() if lm_bias is not None else None
    del W_U, lm_bias

    washout = WASHOUTS["formatting"]
    max_pos_emb = getattr(model.config, "max_position_embeddings", 2048)

    t_start = time.time()

    # ================================================================
    # TEST 1: LM-HEAD PROJECTION
    # ================================================================

    print("\n" + "=" * 70)
    print("TEST 1: LM-Head Projection Analysis")
    print("=" * 70)

    # Extract hidden states for all 16 bank pairs at all layers
    print("\n--- Extracting hidden states for all bank pairs ---")

    all_h_A = []  # all_h_A[pair_idx][layer_idx] -> cpu float32 tensor (hidden_dim,)
    all_h_B = []
    pair_ids = []

    for fi, fin_para in enumerate(FINANCIAL_PARAGRAPHS):
        for ni, nat_para in enumerate(NATURE_PARAGRAPHS):
            ctx_A = build_context(fin_para, nat_para, washout, TARGET)
            ctx_B = build_context(nat_para, fin_para, washout, TARGET)

            h_A = extract_all_hidden_states(model, tokenizer, ctx_A, DEVICE, anchor_ids)
            h_B = extract_all_hidden_states(model, tokenizer, ctx_B, DEVICE, anchor_ids)

            all_h_A.append(h_A)
            all_h_B.append(h_B)
            pair_ids.append(fi * 4 + ni)

            if DEVICE == "cuda":
                torch.cuda.empty_cache()

            print(f"  Pair {fi * 4 + ni}: done ({len(h_A)} layers)")

    n_layers = len(all_h_A[0])
    n_pairs = len(pair_ids)
    last_layer = n_layers - 1  # layer 32 for Phi-2
    print(f"  {n_pairs} pairs, {n_layers} layers (last = {last_layer})")

    # --- Projection ratio at every layer for every pair ---
    print("\n--- Computing projection ratios per layer ---")

    projection_rows = []

    for layer_idx in range(n_layers):
        for pair_idx in range(n_pairs):
            h_A = all_h_A[pair_idx][layer_idx]
            h_B = all_h_B[pair_idx][layer_idx]

            delta_h = h_A - h_B
            delta_h_norm = torch.norm(delta_h).item()

            # ||W_U @ delta_h||
            projected = W_U_cpu @ delta_h  # (vocab_size,)
            projected_norm = torch.norm(projected).item()

            # ratio = ||W_U @ delta_h|| / (||W_U||_F * ||delta_h||)
            denominator = W_U_frobenius * delta_h_norm
            ratio = projected_norm / denominator if denominator > 1e-12 else 0.0

            projection_rows.append({
                "layer": layer_idx,
                "pair_id": pair_ids[pair_idx],
                "delta_h_norm": delta_h_norm,
                "projected_norm": projected_norm,
                "ratio": ratio,
                "d_fr_scrubbed": float("nan"),  # filled in for layer 32 below
            })

        if layer_idx % 8 == 0 or layer_idx == last_layer:
            layer_ratios = [r["ratio"] for r in projection_rows
                            if r["layer"] == layer_idx]
            print(f"  Layer {layer_idx:2d}: mean ratio = {np.mean(layer_ratios):.6f}")

    # --- Linear probe at every layer (for overlay plot) ---
    print("\n--- Linear probe at every layer (LOO-CV) ---")

    probe_accuracies = []
    for layer_idx in range(n_layers):
        X = []
        y = []
        groups = []
        for pair_idx in range(n_pairs):
            X.append(all_h_A[pair_idx][layer_idx].numpy())
            y.append(0)  # A = finance-first
            groups.append(pair_idx)
            X.append(all_h_B[pair_idx][layer_idx].numpy())
            y.append(1)  # B = nature-first
            groups.append(pair_idx)

        X = np.array(X)
        y = np.array(y)
        groups = np.array(groups)

        acc_lo, _ = run_probe_cv(X, y, groups, c_val=0.01)
        acc_hi, _ = run_probe_cv(X, y, groups, c_val=0.1)
        best_acc = max(acc_lo, acc_hi)
        probe_accuracies.append(best_acc)

        if layer_idx % 8 == 0 or layer_idx == last_layer:
            print(f"  Layer {layer_idx:2d}: acc = {best_acc:.3f}")

    probe_accuracies = np.array(probe_accuracies)

    # --- Scrubbing test at output layer ---
    print(f"\n--- Scrubbing test at layer {last_layer} ---")

    # Train probe on ALL data at the output layer to get weight vector w
    X_all = []
    y_all = []
    for pair_idx in range(n_pairs):
        X_all.append(all_h_A[pair_idx][last_layer].numpy())
        y_all.append(0)
        X_all.append(all_h_B[pair_idx][last_layer].numpy())
        y_all.append(1)
    X_all = np.array(X_all)
    y_all = np.array(y_all)

    clf_scrub = LogisticRegression(C=0.1, max_iter=2000, solver="lbfgs")
    clf_scrub.fit(X_all, y_all)
    w = torch.tensor(clf_scrub.coef_[0], dtype=torch.float32)  # (hidden_dim,)
    w_hat = w / torch.norm(w)

    print(f"  Probe weight norm: {torch.norm(w).item():.6f}")
    print(f"  Probe train accuracy: {clf_scrub.score(X_all, y_all):.3f}")

    # Scrub and compute Fisher-Rao distances
    scrub_results = []
    for pair_idx in range(n_pairs):
        h_A = all_h_A[pair_idx][last_layer]
        h_B = all_h_B[pair_idx][last_layer]

        # Project out path direction
        h_A_scrubbed = h_A - (h_A @ w_hat) * w_hat
        h_B_scrubbed = h_B - (h_B @ w_hat) * w_hat

        # Original logits
        logits_A_orig = W_U_cpu @ h_A
        logits_B_orig = W_U_cpu @ h_B
        if lm_bias_cpu is not None:
            logits_A_orig = logits_A_orig + lm_bias_cpu
            logits_B_orig = logits_B_orig + lm_bias_cpu

        # Scrubbed logits
        logits_A_scrub = W_U_cpu @ h_A_scrubbed
        logits_B_scrub = W_U_cpu @ h_B_scrubbed
        if lm_bias_cpu is not None:
            logits_A_scrub = logits_A_scrub + lm_bias_cpu
            logits_B_scrub = logits_B_scrub + lm_bias_cpu

        # Softmax distributions
        p_A_orig = torch.softmax(logits_A_orig, dim=0)
        p_A_scrub = torch.softmax(logits_A_scrub, dim=0)
        p_B_orig = torch.softmax(logits_B_orig, dim=0)
        p_B_scrub = torch.softmax(logits_B_scrub, dim=0)

        # Fisher-Rao: how much does scrubbing change the output distribution?
        d_fr_A = fisher_rao_distance(p_A_orig, p_A_scrub)
        d_fr_B = fisher_rao_distance(p_B_orig, p_B_scrub)
        d_fr_mean = 0.5 * (d_fr_A + d_fr_B)

        scrub_results.append({
            "pair_id": pair_ids[pair_idx],
            "d_fr_A_scrubbed": d_fr_A,
            "d_fr_B_scrubbed": d_fr_B,
            "d_fr_mean_scrubbed": d_fr_mean,
        })

        print(f"  Pair {pair_ids[pair_idx]:2d}: "
              f"d_FR(A, scrubbed_A) = {d_fr_A:.6f}, "
              f"d_FR(B, scrubbed_B) = {d_fr_B:.6f}")

    # Merge scrub results into projection_rows for the output layer
    scrub_lookup = {r["pair_id"]: r["d_fr_mean_scrubbed"] for r in scrub_results}
    for row in projection_rows:
        if row["layer"] == last_layer and row["pair_id"] in scrub_lookup:
            row["d_fr_scrubbed"] = scrub_lookup[row["pair_id"]]

    df_proj = pd.DataFrame(projection_rows)

    mean_d_fr_scrub = np.mean([r["d_fr_mean_scrubbed"] for r in scrub_results])
    print(f"\n  Mean d_FR(original, scrubbed) across all pairs: {mean_d_fr_scrub:.6f}")

    # ================================================================
    # TEST 2: RECENCY-EQUALIZED CONTROL
    # ================================================================

    print("\n" + "=" * 70)
    print("TEST 2: Recency-Equalized Control")
    print("=" * 70)

    # Build contexts with COMMON_SUFFIX inserted between washout and target.
    # A: fin_para + nat_para + washout + COMMON_SUFFIX + TARGET
    # B: nat_para + fin_para + washout + COMMON_SUFFIX + TARGET
    # The last ~200+ tokens before "The bank" are now identical.

    print("\n--- Extracting hidden states for recency-equalized pairs ---")

    # Check context length
    sample_ctx = build_context_with_suffix(
        FINANCIAL_PARAGRAPHS[0], NATURE_PARAGRAPHS[0],
        washout, COMMON_SUFFIX, TARGET
    )
    sample_len = len(tokenizer.encode(sample_ctx))
    print(f"  Sample context: {sample_len} tokens (max: {max_pos_emb})")
    if sample_len > max_pos_emb:
        print("  WARNING: Context exceeds max_position_embeddings!")

    rec_h_A = []
    rec_h_B = []
    rec_pair_ids = []

    for fi, fin_para in enumerate(FINANCIAL_PARAGRAPHS):
        for ni, nat_para in enumerate(NATURE_PARAGRAPHS):
            ctx_A = build_context_with_suffix(
                fin_para, nat_para, washout, COMMON_SUFFIX, TARGET
            )
            ctx_B = build_context_with_suffix(
                nat_para, fin_para, washout, COMMON_SUFFIX, TARGET
            )

            h_A = extract_all_hidden_states(model, tokenizer, ctx_A, DEVICE, anchor_ids)
            h_B = extract_all_hidden_states(model, tokenizer, ctx_B, DEVICE, anchor_ids)

            rec_h_A.append(h_A)
            rec_h_B.append(h_B)
            rec_pair_ids.append(fi * 4 + ni)

            if DEVICE == "cuda":
                torch.cuda.empty_cache()

            print(f"  Pair {fi * 4 + ni}: done")

    # Free model -- no more forward passes needed
    del model
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    print("  Model freed.")

    # --- Layer divergence profile (cosine distance per layer, all 16 pairs) ---
    print("\n--- Layer divergence profile (recency-equalized) ---")

    recency_rows = []
    for pair_idx in range(n_pairs):
        for layer_idx in range(n_layers):
            h_A = rec_h_A[pair_idx][layer_idx]
            h_B = rec_h_B[pair_idx][layer_idx]

            cos_sim = F.cosine_similarity(
                h_A.unsqueeze(0), h_B.unsqueeze(0)
            ).clamp(-1.0, 1.0).item()
            cos_dist = 1.0 - cos_sim

            recency_rows.append({
                "pair_id": rec_pair_ids[pair_idx],
                "layer": layer_idx,
                "cosine_dist": cos_dist,
            })

    df_recency = pd.DataFrame(recency_rows)
    rec_layer_mean = df_recency.groupby("layer")["cosine_dist"].mean().values

    print("  Layer profile (mean cosine dist):")
    for l in range(0, n_layers, 4):
        print(f"    Layer {l:2d}: {rec_layer_mean[l]:.6f}")
    print(f"    Layer {last_layer:2d}: {rec_layer_mean[last_layer]:.6f}")

    # --- Linear probe at every 8th layer (0, 8, 16, 24, 32) ---
    print("\n--- Linear probe at selected layers (recency-equalized, LOO-CV) ---")

    rec_probe_layers = [l for l in [0, 8, 16, 24, 32] if l < n_layers]
    rec_probe_results = {}

    for layer_idx in rec_probe_layers:
        X = []
        y = []
        groups = []
        for pair_idx in range(n_pairs):
            X.append(rec_h_A[pair_idx][layer_idx].numpy())
            y.append(0)
            groups.append(pair_idx)
            X.append(rec_h_B[pair_idx][layer_idx].numpy())
            y.append(1)
            groups.append(pair_idx)

        X = np.array(X)
        y = np.array(y)
        groups = np.array(groups)

        acc_lo, auc_lo = run_probe_cv(X, y, groups, c_val=0.01)
        acc_hi, auc_hi = run_probe_cv(X, y, groups, c_val=0.1)
        best_acc = max(acc_lo, acc_hi)
        best_auc = max(auc_lo, auc_hi)

        rec_probe_results[layer_idx] = {"acc": best_acc, "auc": best_auc}
        print(f"  Layer {layer_idx:2d}: acc = {best_acc:.3f}, AUC = {best_auc:.3f}")

    # --- LM-head projection at output layer (recency-equalized) ---
    print(f"\n--- LM-head projection at layer {last_layer} (recency-equalized) ---")

    rec_proj_ratios = []
    for pair_idx in range(n_pairs):
        h_A = rec_h_A[pair_idx][last_layer]
        h_B = rec_h_B[pair_idx][last_layer]
        delta_h = h_A - h_B
        delta_h_norm = torch.norm(delta_h).item()
        projected = W_U_cpu @ delta_h
        projected_norm = torch.norm(projected).item()
        denominator = W_U_frobenius * delta_h_norm
        ratio = projected_norm / denominator if denominator > 1e-12 else 0.0
        rec_proj_ratios.append(ratio)

    rec_proj_mean = np.mean(rec_proj_ratios)
    orig_proj_at_last = df_proj[df_proj["layer"] == last_layer]["ratio"].values
    orig_proj_mean = np.mean(orig_proj_at_last)
    print(f"  Original bank ratio at layer {last_layer}: {orig_proj_mean:.6f}")
    print(f"  Recency-equalized ratio at layer {last_layer}: {rec_proj_mean:.6f}")

    elapsed = time.time() - t_start
    print(f"\n  Total elapsed: {elapsed:.1f}s")

    # ================================================================
    # ANALYSIS & PLOTS
    # ================================================================

    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    # Compute original bank cosine distance profile for comparison
    orig_cosine_per_layer = []
    for layer_idx in range(n_layers):
        dists = []
        for pair_idx in range(n_pairs):
            h_A = all_h_A[pair_idx][layer_idx]
            h_B = all_h_B[pair_idx][layer_idx]
            cos_sim = F.cosine_similarity(
                h_A.unsqueeze(0), h_B.unsqueeze(0)
            ).clamp(-1.0, 1.0).item()
            dists.append(1.0 - cos_sim)
        orig_cosine_per_layer.append(np.mean(dists))
    orig_cosine_per_layer = np.array(orig_cosine_per_layer)

    # Projection ratio per layer (mean across pairs)
    proj_ratio_per_layer = df_proj.groupby("layer")["ratio"].mean().values

    print("\n  Projection ratio profile:")
    for l in range(0, n_layers, 4):
        print(f"    Layer {l:2d}: proj_ratio={proj_ratio_per_layer[l]:.6f}, "
              f"probe_acc={probe_accuracies[l]:.3f}")
    print(f"    Layer {last_layer:2d}: proj_ratio={proj_ratio_per_layer[last_layer]:.6f}, "
          f"probe_acc={probe_accuracies[last_layer]:.3f}")

    print(f"\n  Scrubbing test (layer {last_layer}):")
    print(f"    Mean d_FR(orig, scrubbed): {mean_d_fr_scrub:.6f}")
    if mean_d_fr_scrub < 0.01:
        scrub_interpretation = "NEGLIGIBLE effect on output logits"
    elif mean_d_fr_scrub < 0.1:
        scrub_interpretation = "SMALL but measurable effect on logits"
    else:
        scrub_interpretation = "SUBSTANTIAL effect on output logits"
    print(f"    --> Path direction has {scrub_interpretation}")

    # Recency control cosine profile analysis
    rec_peak = rec_layer_mean[1:].max()
    rec_peak_layer = int(np.argmax(rec_layer_mean[1:]) + 1)
    rec_output = float(rec_layer_mean[last_layer])
    orig_peak = orig_cosine_per_layer[1:].max()
    orig_peak_layer = int(np.argmax(orig_cosine_per_layer[1:]) + 1)
    orig_output = float(orig_cosine_per_layer[last_layer])

    rec_peak_output_ratio = rec_peak / rec_output if rec_output > 1e-10 else float("inf")
    orig_peak_output_ratio = orig_peak / orig_output if orig_output > 1e-10 else float("inf")

    print(f"\n  Recency control cosine profile:")
    print(f"    Original:    peak={orig_peak:.6f} (layer {orig_peak_layer}), "
          f"output={orig_output:.6f}, ratio={orig_peak_output_ratio:.2f}x")
    print(f"    Recency-eq:  peak={rec_peak:.6f} (layer {rec_peak_layer}), "
          f"output={rec_output:.6f}, ratio={rec_peak_output_ratio:.2f}x")

    peak_reduction = 1.0 - (rec_peak / orig_peak) if orig_peak > 1e-10 else 0.0
    print(f"    Peak reduction: {peak_reduction * 100:.1f}%")

    if rec_peak < orig_peak * 0.3:
        recency_interpretation = "Divergence LARGELY explained by recency"
    elif rec_peak < orig_peak * 0.7:
        recency_interpretation = "Divergence PARTIALLY explained by recency"
    else:
        recency_interpretation = "Divergence PERSISTS despite recency equalization"
    print(f"    --> {recency_interpretation}")

    # ---- Plot ----

    print("\n--- Generating plot ---")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    layers = np.arange(n_layers)

    # Left panel: Projection ratio vs layer (with probe accuracy overlay)
    ax1 = axes[0]
    ax2 = ax1.twinx()

    ln1 = ax1.plot(layers, proj_ratio_per_layer, linewidth=2, color="#2c7bb6",
                   label="Projection ratio", marker="o", markersize=3)
    ax1.set_xlabel("Layer", fontsize=12)
    ax1.set_ylabel("Projection Ratio ||W_U @ dh|| / (||W_U||_F * ||dh||)",
                   fontsize=10, color="#2c7bb6")
    ax1.tick_params(axis="y", labelcolor="#2c7bb6")

    ln2 = ax2.plot(layers, probe_accuracies, linewidth=2, color="#d7191c",
                   label="Probe accuracy (LOO-CV)", marker="s", markersize=3)
    ax2.axhline(y=0.5, color="gray", linestyle=":", alpha=0.5, label="Chance")
    ax2.set_ylabel("Probe Accuracy", fontsize=12, color="#d7191c")
    ax2.tick_params(axis="y", labelcolor="#d7191c")
    ax2.set_ylim(0.0, 1.05)

    lns = ln1 + ln2
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, fontsize=9, loc="center right")
    ax1.set_title("LM-Head Projection Ratio vs Probe Accuracy", fontsize=13)
    ax1.grid(True, alpha=0.3)

    # Right panel: Recency-equalized cosine distance vs original bank profile
    ax = axes[1]
    ax.plot(layers, orig_cosine_per_layer, linewidth=2, color="#2c7bb6",
            label="Original bank", marker="o", markersize=3)
    ax.plot(layers, rec_layer_mean, linewidth=2, color="#d7191c", linestyle="--",
            label="Recency-equalized", marker="s", markersize=3)
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Cosine Distance", fontsize=12)
    ax.set_title("Recency-Equalized vs Original Divergence Profile", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.suptitle("Experiment 3b: Does the LM Head Use Path Information?",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "experiment3b_projection.png", dpi=150, bbox_inches="tight")
    print(f"  Saved: {PLOT_DIR / 'experiment3b_projection.png'}")
    plt.close()

    # ---- Save CSVs -----------------------------------------------------------

    df_proj.to_csv(OUTPUT_DIR / "experiment3b_lm_projection.csv", index=False)
    df_recency.to_csv(OUTPUT_DIR / "experiment3b_recency_control.csv", index=False)

    # ---- Summary JSON --------------------------------------------------------

    summary = {
        "test1_lm_projection": {
            "projection_ratio_per_layer": {
                str(l): float(proj_ratio_per_layer[l]) for l in range(n_layers)
            },
            "probe_accuracy_per_layer": {
                str(l): float(probe_accuracies[l]) for l in range(n_layers)
            },
            "projection_ratio_peak_layer": int(np.argmax(proj_ratio_per_layer[1:]) + 1),
            "projection_ratio_peak": float(proj_ratio_per_layer[1:].max()),
            "projection_ratio_output": float(proj_ratio_per_layer[last_layer]),
            "scrubbing": {
                "layer": int(last_layer),
                "mean_d_fr_scrubbed": float(mean_d_fr_scrub),
                "interpretation": scrub_interpretation,
                "per_pair": [
                    {
                        "pair_id": int(r["pair_id"]),
                        "d_fr_A": float(r["d_fr_A_scrubbed"]),
                        "d_fr_B": float(r["d_fr_B_scrubbed"]),
                        "d_fr_mean": float(r["d_fr_mean_scrubbed"]),
                    }
                    for r in scrub_results
                ],
            },
        },
        "test2_recency_control": {
            "common_suffix_tokens": int(suffix_token_count),
            "context_length_tokens": int(sample_len),
            "max_position_embeddings": int(max_pos_emb),
            "cosine_profile": {
                "original_peak_layer": int(orig_peak_layer),
                "original_peak": float(orig_peak),
                "original_output": float(orig_output),
                "original_peak_output_ratio": float(orig_peak_output_ratio),
                "recency_peak_layer": int(rec_peak_layer),
                "recency_peak": float(rec_peak),
                "recency_output": float(rec_output),
                "recency_peak_output_ratio": float(rec_peak_output_ratio),
                "peak_reduction_pct": float(peak_reduction * 100),
            },
            "interpretation": recency_interpretation,
            "probe_accuracy": {
                str(l): {"acc": float(v["acc"]), "auc": float(v["auc"])}
                for l, v in rec_probe_results.items()
            },
            "projection_ratio_layer32": {
                "original": float(orig_proj_mean),
                "recency_equalized": float(rec_proj_mean),
            },
        },
        "elapsed_seconds": float(elapsed),
    }

    with open(OUTPUT_DIR / "experiment3b_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Saved: {OUTPUT_DIR / 'experiment3b_lm_projection.csv'}")
    print(f"  Saved: {OUTPUT_DIR / 'experiment3b_recency_control.csv'}")
    print(f"  Saved: {OUTPUT_DIR / 'experiment3b_summary.json'}")

    # ---- Final verdict -------------------------------------------------------

    print(f"\n{'=' * 70}")
    print("RESULTS SUMMARY")
    print(f"{'=' * 70}")

    print(f"\n  Test 1: LM-Head Projection")
    proj_peak = proj_ratio_per_layer[1:].max()
    proj_peak_layer = int(np.argmax(proj_ratio_per_layer[1:]) + 1)
    proj_output = proj_ratio_per_layer[last_layer]
    print(f"    Projection ratio peak: {proj_peak:.6f} at layer {proj_peak_layer}")
    print(f"    Projection ratio output (layer {last_layer}): {proj_output:.6f}")
    print(f"    Probe accuracy output (layer {last_layer}): "
          f"{probe_accuracies[last_layer]:.3f}")
    print(f"    Scrubbing d_FR (layer {last_layer}): {mean_d_fr_scrub:.6f}")
    print(f"    --> {scrub_interpretation}")

    print(f"\n  Test 2: Recency Control")
    print(f"    Original peak/output: {orig_peak_output_ratio:.2f}x "
          f"(layer {orig_peak_layer})")
    print(f"    Recency-eq peak/output: {rec_peak_output_ratio:.2f}x "
          f"(layer {rec_peak_layer})")
    print(f"    Peak reduction: {peak_reduction * 100:.1f}%")
    print(f"    --> {recency_interpretation}")

    print(f"\n  Recency-equalized probe accuracy:")
    for l in rec_probe_layers:
        v = rec_probe_results[l]
        label = ""
        if v["acc"] > 0.9:
            label = " <-- path info fully decodable"
        elif v["acc"] > 0.7:
            label = " <-- path info partially decodable"
        elif v["acc"] <= 0.55:
            label = " <-- near chance"
        print(f"    Layer {l:2d}: acc={v['acc']:.3f}, AUC={v['auc']:.3f}{label}")

    print(f"\n{'=' * 70}")


if __name__ == "__main__":
    main()
