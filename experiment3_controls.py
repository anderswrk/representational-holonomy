"""
Experiment 3 Controls: Is the layer divergence bulge architectural or path-specific?

Experiment 3 found a 2.4x peak/output ratio at layer 16. These controls test
whether that's specific to path-order manipulation or a generic property of
deep networks.

Control A: Unrelated contexts (no shared structure)
Control B: Same-topic, different paragraphs (shared target, no path manipulation)
Control C: Shuffled paragraphs (destroyed semantics, preserved token stats)
Control D: Linear probe (is information discarded or just rotated?)
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

from bank_experiment import (
    FINANCIAL_PARAGRAPHS, NATURE_PARAGRAPHS, WASHOUTS, TARGET, build_context
)

MODEL_NAME = "microsoft/phi-2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = Path("/home/anders/consciousness-geometry/results")
PLOT_DIR = OUTPUT_DIR / "bank_experiment_plots"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)


def _find_anchor_pos(input_ids, anchor_ids):
    """Find the position of the last token of anchor_ids at the end of input_ids."""
    ids = input_ids[0].tolist()
    n = len(anchor_ids)
    if ids[-n:] != anchor_ids:
        raise RuntimeError(
            f"Input does not end with expected anchor tokens. "
            f"Expected suffix {anchor_ids}, got {ids[-n:]}"
        )
    return len(ids) - 1  # last token of the anchor


def extract_layer_distances(model, tokenizer, text_A, text_B, device, anchor_ids, label=""):
    """Forward both texts, extract hidden states at the anchor token position.

    Both texts MUST end with the same anchor token sequence (e.g. TARGET).
    Extraction is at the anchor position, not -1, for robustness.
    """
    inputs_A = tokenizer(text_A, return_tensors="pt").to(device)
    inputs_B = tokenizer(text_B, return_tensors="pt").to(device)

    # Hard assert: both inputs end with the anchor token sequence
    pos_A = _find_anchor_pos(inputs_A.input_ids, anchor_ids)
    pos_B = _find_anchor_pos(inputs_B.input_ids, anchor_ids)

    with torch.no_grad():
        out_A = model(**inputs_A, output_hidden_states=True, use_cache=False)
    hidden_A = out_A.hidden_states

    with torch.no_grad():
        out_B = model(**inputs_B, output_hidden_states=True, use_cache=False)
    hidden_B = out_B.hidden_states

    n_layers = len(hidden_A)
    distances = []
    activations_A = []
    activations_B = []

    for layer_idx in range(n_layers):
        z_A = hidden_A[layer_idx][0, pos_A, :].float()
        z_B = hidden_B[layer_idx][0, pos_B, :].float()

        cos_sim = F.cosine_similarity(z_A.unsqueeze(0), z_B.unsqueeze(0)).clamp(-1.0, 1.0).item()
        cos_dist = 1.0 - cos_sim

        distances.append(cos_dist)
        activations_A.append(z_A.cpu().numpy())
        activations_B.append(z_B.cpu().numpy())

    del out_A, out_B, hidden_A, hidden_B, inputs_A, inputs_B

    return distances, activations_A, activations_B


def main():
    print("=" * 70)
    print("EXPERIMENT 3 CONTROLS")
    print("=" * 70)

    from transformers import AutoModelForCausalLM, AutoTokenizer

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

    washout = WASHOUTS["formatting"]

    # Precompute anchor token IDs for TARGET
    anchor_ids = tokenizer.encode(TARGET, add_special_tokens=False)
    print(f"  Anchor tokens for '{TARGET}': {anchor_ids} = {[tokenizer.decode([t]) for t in anchor_ids]}")

    # ---- Recompute Bank Experiment profiles (need activations for Control D) ----

    print("\n--- Bank pairs (recomputing with activations) ---")
    bank_distances = []  # list of lists: [pair][layer]
    bank_acts_A = []     # [pair][layer] -> numpy array
    bank_acts_B = []
    bank_pair_ids = []

    for fi, fin_para in enumerate(FINANCIAL_PARAGRAPHS):
        for ni, nat_para in enumerate(NATURE_PARAGRAPHS):
            ctx_A = build_context(fin_para, nat_para, washout, TARGET)
            ctx_B = build_context(nat_para, fin_para, washout, TARGET)

            dists, acts_A, acts_B = extract_layer_distances(model, tokenizer, ctx_A, ctx_B, DEVICE, anchor_ids, label=f"Bank_{fi}_{ni}")
            bank_distances.append(dists)
            bank_acts_A.append(acts_A)
            bank_acts_B.append(acts_B)
            bank_pair_ids.append(fi * 4 + ni)

            if DEVICE == "cuda":
                torch.cuda.empty_cache()

            print(f"  Pair {fi*4+ni}: done")

    bank_distances = np.array(bank_distances)  # (16, 33)
    n_layers = bank_distances.shape[1]
    bank_mean = bank_distances.mean(axis=0)

    # ---- Control A: Unrelated contexts ----

    print("\n--- Control A: Different-topic, no path manipulation ---")
    # Single paragraph + washout + TARGET: topics differ but no path-order swap.
    # Both end on "The bank" for token-aligned extraction.
    ctrl_a_distances = []
    for fi in range(len(FINANCIAL_PARAGRAPHS)):
        for ni in range(len(NATURE_PARAGRAPHS)):
            text_A = f"{FINANCIAL_PARAGRAPHS[fi]}\n\n{washout}\n\n{TARGET}"
            text_B = f"{NATURE_PARAGRAPHS[ni]}\n\n{washout}\n\n{TARGET}"
            dists, _, _ = extract_layer_distances(model, tokenizer, text_A, text_B, DEVICE, anchor_ids, label=f"CtrlA_{fi}_{ni}")
            ctrl_a_distances.append(dists)
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
    ctrl_a_distances = np.array(ctrl_a_distances)
    ctrl_a_mean = ctrl_a_distances.mean(axis=0)
    print(f"  {len(ctrl_a_distances)} pairs computed")

    # ---- Control B: Same-topic, different paragraphs ----

    print("\n--- Control B: Same-topic, different paragraphs ---")
    # fin_para_i + washout + "The bank" vs fin_para_j + washout + "The bank"
    ctrl_b_distances = []
    for fi_a in range(len(FINANCIAL_PARAGRAPHS)):
        for fi_b in range(fi_a + 1, len(FINANCIAL_PARAGRAPHS)):
            text_A = f"{FINANCIAL_PARAGRAPHS[fi_a]}\n\n{washout}\n\n{TARGET}"
            text_B = f"{FINANCIAL_PARAGRAPHS[fi_b]}\n\n{washout}\n\n{TARGET}"
            dists, _, _ = extract_layer_distances(model, tokenizer, text_A, text_B, DEVICE, anchor_ids, label=f"CtrlB_fin_{fi_a}_{fi_b}")
            ctrl_b_distances.append(dists)
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
    # Also do nature pairs
    for ni_a in range(len(NATURE_PARAGRAPHS)):
        for ni_b in range(ni_a + 1, len(NATURE_PARAGRAPHS)):
            text_A = f"{NATURE_PARAGRAPHS[ni_a]}\n\n{washout}\n\n{TARGET}"
            text_B = f"{NATURE_PARAGRAPHS[ni_b]}\n\n{washout}\n\n{TARGET}"
            dists, _, _ = extract_layer_distances(model, tokenizer, text_A, text_B, DEVICE, anchor_ids, label=f"CtrlB_nat_{ni_a}_{ni_b}")
            ctrl_b_distances.append(dists)
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
    ctrl_b_distances = np.array(ctrl_b_distances)
    ctrl_b_mean = ctrl_b_distances.mean(axis=0)
    print(f"  {len(ctrl_b_distances)} pairs computed")

    # ---- Control C: Shuffled paragraphs ----

    print("\n--- Control C: Sentence-shuffled paragraphs ---")
    # Shuffle sentence ORDER within each paragraph (preserves individual sentences
    # and their tokenization, but destroys narrative coherence).
    # Compare shuffled_A (shuffled_fin→shuffled_nat) vs shuffled_B (shuffled_nat→shuffled_fin).
    # Path-order swap is preserved. If the bulge survives, the effect doesn't
    # require discourse-level coherence (topic mass and path order suffice).
    import re
    rng = np.random.default_rng(42)
    ctrl_c_distances = []

    def shuffle_sentences(text, rng):
        """Split on sentence boundaries and shuffle."""
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        sentences = [s for s in sentences if s]  # remove empty
        rng.shuffle(sentences)
        return " ".join(sentences)

    for fi, fin_para in enumerate(FINANCIAL_PARAGRAPHS):
        for ni, nat_para in enumerate(NATURE_PARAGRAPHS):
            shuffled_fin = shuffle_sentences(fin_para, rng)
            shuffled_nat = shuffle_sentences(nat_para, rng)

            # Shuffled A: shuffled_fin → shuffled_nat → washout → target
            ctx_A_shuffled = build_context(shuffled_fin, shuffled_nat, washout, TARGET)
            # Shuffled B: shuffled_nat → shuffled_fin → washout → target
            ctx_B_shuffled = build_context(shuffled_nat, shuffled_fin, washout, TARGET)

            dists, _, _ = extract_layer_distances(model, tokenizer, ctx_A_shuffled, ctx_B_shuffled, DEVICE, anchor_ids, label=f"CtrlC_{fi}_{ni}")
            ctrl_c_distances.append(dists)
            if DEVICE == "cuda":
                torch.cuda.empty_cache()

    ctrl_c_distances = np.array(ctrl_c_distances)
    ctrl_c_mean = ctrl_c_distances.mean(axis=0)
    print(f"  {len(ctrl_c_distances)} pairs computed")

    # ---- Control D: Linear probe ----

    print("\n--- Control D: Linear probe (leave-one-pair-out) ---")
    # At each layer, train logistic regression to classify A vs B
    # Leave-one-pair-out CV: train on 15 pairs (30 samples), test on 1 pair (2 samples)
    # Try both C=0.01 and C=0.1, report both

    from sklearn.metrics import roc_auc_score

    probe_results = {c_val: {"acc": [], "auc": []} for c_val in [0.01, 0.1]}
    n_pairs = len(bank_pair_ids)

    for layer_idx in range(n_layers):
        # Build feature matrix: (32, 2560) and labels: (32,) 0=A, 1=B
        X = []
        y = []
        groups = []  # pair index for leave-one-group-out
        for pair_idx in range(n_pairs):
            X.append(bank_acts_A[pair_idx][layer_idx])
            y.append(0)  # A = finance-first
            groups.append(pair_idx)

            X.append(bank_acts_B[pair_idx][layer_idx])
            y.append(1)  # B = nature-first
            groups.append(pair_idx)

        X = np.array(X)
        y = np.array(y)
        groups = np.array(groups)

        logo = LeaveOneGroupOut()

        for c_val in [0.01, 0.1]:
            correct = 0
            total = 0
            all_probs = []
            all_true = []

            for train_idx, test_idx in logo.split(X, y, groups):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                # Standardize per fold
                mean = X_train.mean(axis=0)
                std = X_train.std(axis=0) + 1e-8
                X_train = (X_train - mean) / std
                X_test = (X_test - mean) / std

                clf = LogisticRegression(C=c_val, max_iter=1000, solver="lbfgs")
                clf.fit(X_train, y_train)
                preds = clf.predict(X_test)
                probs = clf.predict_proba(X_test)[:, 1]

                correct += (preds == y_test).sum()
                total += len(y_test)
                all_probs.extend(probs.tolist())
                all_true.extend(y_test.tolist())

            acc = correct / total
            try:
                auc = roc_auc_score(all_true, all_probs)
            except ValueError:
                auc = 0.5  # degenerate case
            probe_results[c_val]["acc"].append(acc)
            probe_results[c_val]["auc"].append(auc)

        if layer_idx % 8 == 0 or layer_idx == n_layers - 1:
            accs = [probe_results[c]["acc"][-1] for c in [0.01, 0.1]]
            aucs = [probe_results[c]["auc"][-1] for c in [0.01, 0.1]]
            print(f"  Layer {layer_idx:2d}: acc(C=0.01)={accs[0]:.3f}, acc(C=0.1)={accs[1]:.3f}, "
                  f"auc(C=0.01)={aucs[0]:.3f}, auc(C=0.1)={aucs[1]:.3f}")

    # Use best C per layer (by AUC) for the primary probe curve
    probe_accuracies = np.array([max(probe_results[0.01]["acc"][l], probe_results[0.1]["acc"][l])
                                  for l in range(n_layers)])
    probe_aucs = np.array([max(probe_results[0.01]["auc"][l], probe_results[0.1]["auc"][l])
                            for l in range(n_layers)])

    # Free model
    del model
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    # ---- Analysis ----

    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    # Peak/output ratios for each condition
    def peak_output_ratio(profile):
        # Exclude embedding layer 0
        peak_val = profile[1:].max()
        peak_layer = np.argmax(profile[1:]) + 1
        output_val = profile[-1]
        ratio = peak_val / output_val if output_val > 1e-10 else float("inf")
        return peak_layer, peak_val, output_val, ratio

    conditions = {
        "Bank (path-order)": bank_mean,
        "Ctrl A (diff topic)": ctrl_a_mean,
        "Ctrl B (same-topic)": ctrl_b_mean,
        "Ctrl C (shuffled)": ctrl_c_mean,
    }

    print("\n  Peak/output ratios:")
    for name, profile in conditions.items():
        pl, pv, ov, r = peak_output_ratio(profile)
        print(f"    {name:25s}: peak layer={pl:2d}, peak={pv:.6f}, output={ov:.6f}, ratio={r:.2f}x")

    # Probe accuracy and AUC
    probe_peak_layer = np.argmax(probe_accuracies[1:]) + 1
    probe_peak_acc = probe_accuracies[1:].max()
    probe_output_acc = probe_accuracies[-1]
    auc_peak_layer = np.argmax(probe_aucs[1:]) + 1
    auc_peak = probe_aucs[1:].max()
    auc_output = probe_aucs[-1]

    print(f"\n  Linear probe (best of C=0.01, C=0.1 per layer):")
    print(f"    Peak accuracy at layer {probe_peak_layer}: {probe_peak_acc:.3f}")
    print(f"    Output layer accuracy: {probe_output_acc:.3f}")
    print(f"    Accuracy drop peak->output: {probe_peak_acc - probe_output_acc:+.3f}")
    print(f"    Peak AUC at layer {auc_peak_layer}: {auc_peak:.3f}")
    print(f"    Output layer AUC: {auc_output:.3f}")
    print(f"    AUC drop peak->output: {auc_peak - auc_output:+.3f}")

    acc_drop = probe_peak_acc - probe_output_acc
    auc_drop = auc_peak - auc_output
    if acc_drop > 0.05 or auc_drop > 0.05:
        print(f"    --> Probe decodability DROPS from intermediate to output")
        print(f"    --> Genuine information loss (compression is real)")
    elif probe_output_acc > 0.7 or auc_output > 0.7:
        print(f"    --> Probe decodability stays HIGH at output")
        print(f"    --> Information rotated, not discarded (cosine bulge is metric artifact)")
    else:
        print(f"    --> Probe decodability low throughout (insufficient signal or n too small)")

    # ---- Plot ----

    print("\n--- Generating plots ---")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left panel: Cosine distance profiles overlaid
    ax = axes[0]
    layers = np.arange(n_layers)
    colors = {"Bank (path-order)": "blue", "Ctrl A (diff topic)": "red",
              "Ctrl B (same-topic)": "orange", "Ctrl C (shuffled)": "green"}

    for name, profile in conditions.items():
        ax.plot(layers, profile, linewidth=2, label=name, color=colors[name])

    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Cosine Distance", fontsize=12)
    ax.set_title("Layer Divergence: Bank vs Controls", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Right panel: Probe accuracy + Bank cosine distance (dual y-axis)
    ax1 = axes[1]
    ax2 = ax1.twinx()

    ln1 = ax1.plot(layers, probe_accuracies, linewidth=2, color="purple",
                    label="Probe accuracy", marker="o", markersize=3)
    ln1b = ax1.plot(layers, probe_aucs, linewidth=2, color="darkviolet", linestyle=":",
                     label="Probe AUC", marker="s", markersize=2)
    ax1.axhline(y=0.5, color="gray", linestyle=":", alpha=0.5, label="Chance")
    ax1.set_xlabel("Layer", fontsize=12)
    ax1.set_ylabel("Probe Accuracy (LOO-CV)", fontsize=12, color="purple")
    ax1.tick_params(axis="y", labelcolor="purple")

    # Normalized cosine distance for comparison
    bank_norm = bank_mean / bank_mean.max() if bank_mean.max() > 0 else bank_mean
    ln2 = ax2.plot(layers, bank_norm, linewidth=2, color="blue", linestyle="--",
                    label="Cosine dist (normalized)", alpha=0.7)
    ax2.set_ylabel("Cosine Distance (normalized)", fontsize=12, color="blue")
    ax2.tick_params(axis="y", labelcolor="blue")

    # Combined legend
    lns = ln1 + ln1b + ln2
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, fontsize=9, loc="lower right")
    ax1.set_title("Control D: Probe Accuracy vs Cosine Distance", fontsize=13)
    ax1.grid(True, alpha=0.3)

    fig.suptitle("Experiment 3 Controls: Is the Bulge Real?",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "experiment3_controls.png", dpi=150, bbox_inches="tight")
    print(f"  Saved: {PLOT_DIR / 'experiment3_controls.png'}")
    plt.close()

    # ---- Save ----

    # Save control profiles
    rows = []
    for name, profile in conditions.items():
        for layer_idx, val in enumerate(profile):
            rows.append({"condition": name, "layer": layer_idx, "value": val, "metric": "cosine_dist"})
    # Add probe metrics
    for layer_idx in range(n_layers):
        rows.append({"condition": "Probe accuracy", "layer": layer_idx, "value": probe_accuracies[layer_idx], "metric": "accuracy"})
        rows.append({"condition": "Probe AUC", "layer": layer_idx, "value": probe_aucs[layer_idx], "metric": "auc"})
    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "experiment3_controls.csv", index=False)

    summary = {
        "conditions": {},
        "probe": {
            "peak_layer_acc": int(probe_peak_layer),
            "peak_accuracy": float(probe_peak_acc),
            "output_accuracy": float(probe_output_acc),
            "accuracy_drop": float(acc_drop),
            "peak_layer_auc": int(auc_peak_layer),
            "peak_auc": float(auc_peak),
            "output_auc": float(auc_output),
            "auc_drop": float(auc_drop),
        },
    }
    for name, profile in conditions.items():
        pl, pv, ov, r = peak_output_ratio(profile)
        summary["conditions"][name] = {
            "peak_layer": int(pl), "peak_cosine": float(pv),
            "output_cosine": float(ov), "ratio": float(r),
        }

    with open(OUTPUT_DIR / "experiment3_controls_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Verdict
    bank_ratio = summary["conditions"]["Bank (path-order)"]["ratio"]
    ctrl_a_ratio = summary["conditions"]["Ctrl A (diff topic)"]["ratio"]
    ctrl_b_ratio = summary["conditions"]["Ctrl B (same-topic)"]["ratio"]

    print(f"\n{'=' * 70}")
    print(f"Bank ratio: {bank_ratio:.2f}x")
    print(f"Ctrl A ratio: {ctrl_a_ratio:.2f}x")
    print(f"Ctrl B ratio: {ctrl_b_ratio:.2f}x")

    if bank_ratio > ctrl_a_ratio * 1.3 and bank_ratio > ctrl_b_ratio * 1.3:
        print("Bank bulge is LARGER than controls → path-specific structure")
    elif bank_ratio < ctrl_a_ratio * 0.9:
        print("Bank bulge is SMALLER than unrelated control → architectural artifact")
    else:
        print("Bank bulge is COMPARABLE to controls → inconclusive")

    acc_drop = probe_peak_acc - probe_output_acc
    if acc_drop > 0.05:
        print(f"Probe accuracy drops {acc_drop:.3f} → genuine information loss")
    else:
        print(f"Probe accuracy drop {acc_drop:.3f} → information may be rotated, not lost")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
