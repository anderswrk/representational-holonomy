"""
Experiment 3: Intermediate Layer Divergence Profile

Tests whether path dependence (finance-first vs nature-first) is stronger
at intermediate layers than at the output layer. A "bulge" at intermediate
layers would indicate the model discards path information -- exactly the
compression that creates nontrivial fiber structure.

Tier 3 diagnostic: chart-dependent, not gauge-invariant.
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

from bank_experiment import (
    FINANCIAL_PARAGRAPHS, NATURE_PARAGRAPHS, WASHOUTS, TARGET, build_context
)

MODEL_NAME = "microsoft/phi-2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = Path("/home/anders/consciousness-geometry/results")
PLOT_DIR = OUTPUT_DIR / "bank_experiment_plots"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("=" * 70)
    print("EXPERIMENT 3: Intermediate Layer Divergence Profile")
    print("=" * 70)

    # Load model
    print(f"\nLoading {MODEL_NAME}...")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if DEVICE == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float32,
            trust_remote_code=True,
        ).to(DEVICE)
    model.eval()

    max_pos_emb = getattr(model.config, "max_position_embeddings", 2048)
    print(f"Model loaded. max_position_embeddings = {max_pos_emb}")

    # Build all 16 pairs
    print("\n--- Building path pairs ---")
    washout = WASHOUTS["formatting"]
    pairs = []
    for fi, fin_para in enumerate(FINANCIAL_PARAGRAPHS):
        for ni, nat_para in enumerate(NATURE_PARAGRAPHS):
            ctx_A = build_context(fin_para, nat_para, washout, TARGET)
            ctx_B = build_context(nat_para, fin_para, washout, TARGET)
            pairs.append({
                "pair_id": fi * 4 + ni,
                "fin_idx": fi,
                "nat_idx": ni,
                "context_A": ctx_A,
                "context_B": ctx_B,
            })
    print(f"  {len(pairs)} pairs (4 fin x 4 nat)")

    # Check context lengths
    sample_len = len(tokenizer.encode(pairs[0]["context_A"]))
    print(f"  Sample context length: {sample_len} tokens (max: {max_pos_emb})")
    if sample_len > max_pos_emb:
        print("  WARNING: Context exceeds max position embeddings!")

    # Forward each pair, extract hidden states at last token
    print("\n--- Forward passes with output_hidden_states=True ---")
    results = []
    t_start = time.time()

    for pp in pairs:
        # Forward A
        inputs_A = tokenizer(pp["context_A"], return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            out_A = model(**inputs_A, output_hidden_states=True, use_cache=False)
        hidden_A = out_A.hidden_states  # tuple of (n_layers+1,) tensors

        # Forward B
        inputs_B = tokenizer(pp["context_B"], return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            out_B = model(**inputs_B, output_hidden_states=True, use_cache=False)
        hidden_B = out_B.hidden_states

        n_layers = len(hidden_A)  # 33 for Phi-2 (embedding + 32 layers)

        for layer_idx in range(n_layers):
            z_A = hidden_A[layer_idx][0, -1, :].float()
            z_B = hidden_B[layer_idx][0, -1, :].float()

            # Cosine distance
            cos_sim = F.cosine_similarity(z_A.unsqueeze(0), z_B.unsqueeze(0)).clamp(-1.0, 1.0).item()
            cos_dist = 1.0 - cos_sim

            # L2 distance
            l2_dist = torch.norm(z_A - z_B).item()

            # Relative L2
            norm_A = torch.norm(z_A).item()
            norm_B = torch.norm(z_B).item()
            mean_norm = 0.5 * (norm_A + norm_B)
            l2_relative = l2_dist / mean_norm if mean_norm > 1e-12 else 0.0

            results.append({
                "pair_id": pp["pair_id"],
                "fin_idx": pp["fin_idx"],
                "nat_idx": pp["nat_idx"],
                "layer": layer_idx,
                "cosine_dist": cos_dist,
                "l2_dist": l2_dist,
                "l2_relative": l2_relative,
                "norm_A": norm_A,
                "norm_B": norm_B,
            })

        # Free memory
        del out_A, out_B, hidden_A, hidden_B, inputs_A, inputs_B
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

        print(f"  Pair {pp['pair_id']:2d}: {n_layers} layers extracted")

    elapsed = time.time() - t_start
    print(f"  Done in {elapsed:.1f}s")

    # Free model
    del model
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    # Build dataframe
    df = pd.DataFrame(results)

    # ---- Analysis -----------------------------------------------------------

    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    # Mean distance per layer
    layer_stats = df.groupby("layer").agg(
        cosine_mean=("cosine_dist", "mean"),
        cosine_std=("cosine_dist", "std"),
        l2_mean=("l2_dist", "mean"),
        l2_std=("l2_dist", "std"),
        l2_rel_mean=("l2_relative", "mean"),
        l2_rel_std=("l2_relative", "std"),
        norm_A_mean=("norm_A", "mean"),
        norm_B_mean=("norm_B", "mean"),
    ).reset_index()

    # Peak layer (exclude embedding layer 0)
    layer_stats_no_emb = layer_stats[layer_stats["layer"] > 0]
    peak_idx = layer_stats_no_emb["cosine_mean"].idxmax()
    peak_layer = int(layer_stats_no_emb.loc[peak_idx, "layer"])
    peak_cosine = float(layer_stats_no_emb.loc[peak_idx, "cosine_mean"])

    # L2 relative peak
    l2_peak_idx = layer_stats_no_emb["l2_rel_mean"].idxmax()
    l2_peak_layer = int(layer_stats_no_emb.loc[l2_peak_idx, "layer"])

    # Output layer
    max_layer = int(layer_stats["layer"].max())
    output_cosine = float(layer_stats[layer_stats["layer"] == max_layer]["cosine_mean"].iloc[0])
    output_l2_rel = float(layer_stats[layer_stats["layer"] == max_layer]["l2_rel_mean"].iloc[0])

    # Ratio and verdict
    ratio = peak_cosine / output_cosine if output_cosine > 1e-10 else float("inf")
    non_monotonic = (peak_layer < max_layer) and (ratio > 1.2)

    print(f"\n  Layers: {max_layer + 1} (embedding + {max_layer} transformer layers)")
    print(f"\n  Cosine distance:")
    print(f"    Peak at layer {peak_layer}: {peak_cosine:.6f}")
    print(f"    Output layer {max_layer}: {output_cosine:.6f}")
    print(f"    Peak / output ratio: {ratio:.2f}x")
    print(f"\n  Relative L2:")
    print(f"    Peak at layer {l2_peak_layer}: {layer_stats_no_emb.loc[l2_peak_idx, 'l2_rel_mean']:.6f}")
    print(f"    Output layer {max_layer}: {output_l2_rel:.6f}")

    print(f"\n  Non-monotonic profile: {non_monotonic}")
    if non_monotonic:
        print(f"  --> INTERMEDIATE LAYERS DIVERGE MORE THAN OUTPUT")
        print(f"  --> Model discards {(1 - 1/ratio)*100:.0f}% of path information")
    else:
        print(f"  --> No evidence of hidden intermediate divergence")

    # Print full layer profile
    print(f"\n  Layer-by-layer cosine distance:")
    for _, row in layer_stats.iterrows():
        layer = int(row["layer"])
        marker = " <-- PEAK" if layer == peak_layer else ""
        marker = " <-- OUTPUT" if layer == max_layer and not marker else marker
        print(f"    Layer {layer:2d}: cos={row['cosine_mean']:.6f} +/- {row['cosine_std']:.6f}  "
              f"L2rel={row['l2_rel_mean']:.6f}{marker}")

    # ---- Plot ---------------------------------------------------------------

    print("\n--- Generating plot ---")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left: cosine distance
    ax = axes[0]
    for pid in df["pair_id"].unique():
        pair_data = df[df["pair_id"] == pid]
        ax.plot(pair_data["layer"], pair_data["cosine_dist"],
                alpha=0.2, linewidth=0.8, color="steelblue")
    ax.plot(layer_stats["layer"], layer_stats["cosine_mean"],
            linewidth=2.5, color="blue", label="Mean across pairs")
    ax.fill_between(layer_stats["layer"],
                    layer_stats["cosine_mean"] - layer_stats["cosine_std"],
                    layer_stats["cosine_mean"] + layer_stats["cosine_std"],
                    alpha=0.15, color="blue")
    ax.axvline(x=peak_layer, color="red", linestyle="--", alpha=0.7,
               label=f"Peak: layer {peak_layer}")
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Cosine Distance (1 - cos sim)", fontsize=12)
    ax.set_title("Cosine Distance: Path A vs Path B", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Right: relative L2
    ax = axes[1]
    for pid in df["pair_id"].unique():
        pair_data = df[df["pair_id"] == pid]
        ax.plot(pair_data["layer"], pair_data["l2_relative"],
                alpha=0.2, linewidth=0.8, color="darkorange")
    ax.plot(layer_stats["layer"], layer_stats["l2_rel_mean"],
            linewidth=2.5, color="orangered", label="Mean across pairs")
    ax.fill_between(layer_stats["layer"],
                    layer_stats["l2_rel_mean"] - layer_stats["l2_rel_std"],
                    layer_stats["l2_rel_mean"] + layer_stats["l2_rel_std"],
                    alpha=0.15, color="orangered")
    ax.axvline(x=l2_peak_layer, color="red", linestyle="--", alpha=0.7,
               label=f"Peak: layer {l2_peak_layer}")
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Relative L2 Distance", fontsize=12)
    ax.set_title("Relative L2: Path A vs Path B", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.suptitle("Experiment 3: Does Divergence Peak at Intermediate Layers?",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "experiment3_layer_profile.png", dpi=150, bbox_inches="tight")
    print(f"  Saved: {PLOT_DIR / 'experiment3_layer_profile.png'}")
    plt.close()

    # ---- Save ---------------------------------------------------------------

    df.to_csv(OUTPUT_DIR / "experiment3_layer_divergence.csv", index=False)
    print(f"  Saved: {OUTPUT_DIR / 'experiment3_layer_divergence.csv'}")

    summary = {
        "n_pairs": len(pairs),
        "n_layers": max_layer + 1,
        "peak_layer_cosine": peak_layer,
        "peak_cosine_dist": peak_cosine,
        "peak_layer_l2_rel": l2_peak_layer,
        "output_layer": max_layer,
        "output_cosine_dist": output_cosine,
        "output_l2_rel": output_l2_rel,
        "peak_output_ratio_cosine": float(ratio),
        "non_monotonic": non_monotonic,
        "layer_profile": layer_stats.to_dict(orient="records"),
    }
    with open(OUTPUT_DIR / "experiment3_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: {OUTPUT_DIR / 'experiment3_summary.json'}")

    # Final verdict
    print(f"\n{'=' * 70}")
    if non_monotonic:
        print(f"RESULT: Non-monotonic profile detected.")
        print(f"  Peak divergence at layer {peak_layer} ({peak_cosine:.6f})")
        print(f"  Output divergence at layer {max_layer} ({output_cosine:.6f})")
        print(f"  Ratio: {ratio:.2f}x -- model discards {(1 - 1/ratio)*100:.0f}% of path info")
    else:
        print(f"RESULT: Monotonic or flat profile -- no hidden intermediate divergence.")
        print(f"  Peak at layer {peak_layer}, output at layer {max_layer}")
        print(f"  Ratio: {ratio:.2f}x (threshold: 1.2x)")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
