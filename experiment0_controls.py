"""
Experiment 0 Controls: Permuted-W and Rotated-W

Tests whether the category structure in d_eff residuals comes from
p-mass aligning with semantic structure in W, or is an artifact of W's spectrum.

Control A (row-permute W): Destroys semantic structure, preserves spectrum.
  If category residual pattern survives → artifact of W's spectrum.
  If it collapses → signal depends on p-W alignment. Framework lives.

Control B (orthogonal rotation): Gauge invariance sanity check.
  d_eff_rot should match d_eff within floating-point noise.
  If not → numerical bug.
"""

import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import time
import json
import random

MODEL_NAME = "microsoft/phi-2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = Path("/home/anders/consciousness-geometry/results")
TOPK = 512
N_PERMUTATIONS = 3


def compute_deff_for_logits(logits: torch.Tensor, W: torch.Tensor,
                            topk: int = TOPK) -> dict:
    """Compute d_eff and H(p) for a single position's logits."""
    V = logits.shape[0]
    d = W.shape[1]
    K = min(topk, V)

    p = torch.softmax(logits.float(), dim=0)

    # Shannon entropy over full distribution
    log_p = torch.log(p.clamp(min=1e-12))
    H_p = -(p * log_p).sum().item()

    # Top-K truncation
    pk, idx = torch.topk(p, k=K, largest=True, sorted=False)
    Wk = W.index_select(0, idx)

    mass = pk.sum()
    if mass < 1e-12:
        return {"d_eff": 0.0, "H_p": H_p}

    pk_norm = pk / mass
    w_bar = pk_norm @ Wk
    W_centered = Wk - w_bar.unsqueeze(0)
    W_scaled = torch.sqrt(pk_norm).unsqueeze(1) * W_centered
    G = W_scaled.T @ W_scaled

    eigenvalues = torch.linalg.eigvalsh(G).clamp(min=0)
    trace = eigenvalues.sum()

    if trace < 1e-12:
        return {"d_eff": 0.0, "H_p": H_p}

    rho = eigenvalues / trace
    mask = rho > 1e-15
    d_eff = -(rho[mask] * torch.log(rho[mask])).sum().item()

    return {"d_eff": d_eff, "H_p": H_p}


def main():
    print("=" * 70)
    print("EXPERIMENT 0 CONTROLS: Permuted-W and Rotated-W")
    print("=" * 70)

    # Load model
    print(f"\nLoading {MODEL_NAME}...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from experiment0 import PROMPTS, make_shuffled_prompts

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="cuda",
        trust_remote_code=True,
    )
    model.eval()

    W_gpu = model.get_output_embeddings().weight.detach().float().to(DEVICE)
    V, d = W_gpu.shape
    print(f"W: {V} × {d}")

    # Add shuffled noise (same as experiment0)
    shuffled = make_shuffled_prompts(tokenizer, PROMPTS, n_shuffled=8)
    PROMPTS["noise_shuffled"] = shuffled

    # ─── Control B: Orthogonal rotation (gauge invariance check) ─────────
    print("\n--- Control B: Orthogonal Rotation (Gauge Invariance) ---")
    A = torch.randn(d, d, device=W_gpu.device)
    Q, _ = torch.linalg.qr(A)
    W_rot = W_gpu @ Q

    # ─── Control A: Row-permuted W (3 independent permutations) ──────────
    print(f"\n--- Control A: {N_PERMUTATIONS} Independent Row Permutations ---")
    W_perms = []
    for i in range(N_PERMUTATIONS):
        perm = torch.randperm(V, device=W_gpu.device)
        W_perms.append(W_gpu[perm])

    # ─── Process all prompts, computing real + controls in one pass ──────
    total_prompts = sum(len(v) for v in PROMPTS.values())
    print(f"\n{total_prompts} prompts to process")

    rows = []
    done = 0
    t_start = time.time()

    for cat, prompt_list in PROMPTS.items():
        for prompt in prompt_list:
            done += 1
            print(f"  [{done}/{total_prompts}] {cat}: {prompt[:50]}...", end="", flush=True)

            inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                outputs = model(**inputs)

            logits_all = outputs.logits[0]  # (seq_len, V)
            seq_len = logits_all.shape[0]

            for pos in range(seq_len):
                logits_pos = logits_all[pos]

                # Real W
                res_real = compute_deff_for_logits(logits_pos, W_gpu)
                # Rotated W
                res_rot = compute_deff_for_logits(logits_pos, W_rot)
                # Permuted W
                perm_deffs = []
                for i in range(N_PERMUTATIONS):
                    res_perm = compute_deff_for_logits(logits_pos, W_perms[i])
                    perm_deffs.append(res_perm["d_eff"])

                row = {
                    "category": cat,
                    "prompt": prompt[:80],
                    "position": pos,
                    "seq_len": seq_len,
                    "H_p": res_real["H_p"],
                    "d_eff": res_real["d_eff"],
                    "d_eff_rot": res_rot["d_eff"],
                }
                for i in range(N_PERMUTATIONS):
                    row[f"d_eff_perm{i}"] = perm_deffs[i]
                row["d_eff_perm_mean"] = np.mean(perm_deffs)
                rows.append(row)

            print(f" {seq_len} pos, done")

    elapsed = time.time() - t_start
    print(f"\nTotal computation: {elapsed:.1f}s")

    # Free model
    del model
    torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    print(f"Built dataframe: {len(df)} rows")

    # ─── Gauge invariance check ──────────────────────────────────────────
    print("\n--- Gauge Invariance Check ---")
    rot_diff = (df["d_eff"] - df["d_eff_rot"]).abs()
    print(f"  |d_eff - d_eff_rot|: mean={rot_diff.mean():.6f}, max={rot_diff.max():.6f}")
    gauge_ok = rot_diff.max() < 0.01
    print(f"  Gauge invariance: {'PASS' if gauge_ok else 'FAIL'}")

    # ─── Residual analysis for permuted W ────────────────────────────────
    # Fit cubic to d_eff_perm_mean vs H_p, compute residuals
    coeffs_perm = np.polyfit(df["H_p"].values, df["d_eff_perm_mean"].values, deg=3)
    df["d_eff_perm_predicted"] = np.polyval(coeffs_perm, df["H_p"].values)
    df["d_eff_perm_residual"] = df["d_eff_perm_mean"] - df["d_eff_perm_predicted"]

    # Use existing residuals from original run
    coeffs_real = np.polyfit(df["H_p"].values, df["d_eff"].values, deg=3)
    df["d_eff_predicted"] = np.polyval(coeffs_real, df["H_p"].values)
    df["d_eff_residual"] = df["d_eff"] - df["d_eff_predicted"]

    # Per-permutation residuals for error bars
    for i in range(N_PERMUTATIONS):
        coeffs_i = np.polyfit(df["H_p"].values, df[f"d_eff_perm{i}"].values, deg=3)
        df[f"d_eff_perm{i}_predicted"] = np.polyval(coeffs_i, df["H_p"].values)
        df[f"d_eff_perm{i}_residual"] = df[f"d_eff_perm{i}"] - df[f"d_eff_perm{i}_predicted"]

    # ─── Category residual means comparison ──────────────────────────────
    print("\n--- Category Residual Means: Real W vs Permuted W ---")

    cats = df["category"].unique()
    cat_stats = []
    for cat in sorted(cats):
        mask = df["category"] == cat
        real_mean = df.loc[mask, "d_eff_residual"].mean()
        perm_means = [df.loc[mask, f"d_eff_perm{i}_residual"].mean() for i in range(N_PERMUTATIONS)]
        perm_mean = np.mean(perm_means)
        perm_std = np.std(perm_means)
        rot_mean = (df.loc[mask, "d_eff_rot"] - np.polyval(
            np.polyfit(df["H_p"].values, df["d_eff_rot"].values, deg=3),
            df.loc[mask, "H_p"].values)).mean()

        cat_stats.append({
            "category": cat,
            "residual_real": real_mean,
            "residual_perm_mean": perm_mean,
            "residual_perm_std": perm_std,
            "residual_rot": rot_mean,
            "signal_survives_perm": abs(perm_mean) > abs(real_mean) * 0.5,
        })
        print(f"  {cat:20s}  real={real_mean:+.4f}  perm={perm_mean:+.4f}±{perm_std:.4f}  rot={rot_mean:+.4f}")

    stats_df = pd.DataFrame(cat_stats)

    # ─── Key test: does the pattern collapse under permutation? ──────────
    print("\n--- Pattern Collapse Test ---")
    # Correlation between real residual pattern and permuted residual pattern
    pattern_corr = np.corrcoef(stats_df["residual_real"], stats_df["residual_perm_mean"])[0, 1]
    print(f"  Correlation of category residual patterns (real vs perm): {pattern_corr:.4f}")

    # Does the code-below / philosophical-above pattern survive?
    real_range = stats_df["residual_real"].max() - stats_df["residual_real"].min()
    perm_range = stats_df["residual_perm_mean"].max() - stats_df["residual_perm_mean"].min()
    range_ratio = perm_range / real_range if real_range > 0 else 0
    print(f"  Category spread: real={real_range:.4f}, perm={perm_range:.4f}, ratio={range_ratio:.2f}")

    if pattern_corr > 0.7 and range_ratio > 0.5:
        verdict = "ARTIFACT — pattern survives permutation, driven by W's spectrum"
    elif pattern_corr < 0.3 or range_ratio < 0.3:
        verdict = "GEOMETRIC — pattern collapses under permutation, depends on p-W alignment"
    else:
        verdict = "AMBIGUOUS — partial survival, needs further investigation"
    print(f"  Verdict: {verdict}")

    # ─── Plot: Category residual means, three conditions ─────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Bar chart comparison
    ax = axes[0]
    cat_order = stats_df.sort_values("residual_real")["category"].tolist()
    x = np.arange(len(cat_order))
    width = 0.25

    real_vals = [stats_df[stats_df["category"] == c]["residual_real"].values[0] for c in cat_order]
    perm_vals = [stats_df[stats_df["category"] == c]["residual_perm_mean"].values[0] for c in cat_order]
    perm_errs = [stats_df[stats_df["category"] == c]["residual_perm_std"].values[0] for c in cat_order]
    rot_vals = [stats_df[stats_df["category"] == c]["residual_rot"].values[0] for c in cat_order]

    bars1 = ax.bar(x - width, real_vals, width, label="Real W", color="#3498db", alpha=0.8)
    bars2 = ax.bar(x, perm_vals, width, label=f"Permuted W (n={N_PERMUTATIONS})",
                   color="#e74c3c", alpha=0.8, yerr=perm_errs, capsize=3)
    bars3 = ax.bar(x + width, rot_vals, width, label="Rotated W (gauge)", color="#2ecc71", alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(cat_order, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Mean d_eff residual (after cubic detrend)", fontsize=11)
    ax.set_title(f"Category Residuals: Real vs Permuted vs Rotated W\n"
                 f"Pattern corr={pattern_corr:.2f}, range ratio={range_ratio:.2f}",
                 fontsize=12)
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    # Scatter: real residual vs perm residual per category
    ax = axes[1]
    for _, row in stats_df.iterrows():
        ax.scatter(row["residual_real"], row["residual_perm_mean"],
                   s=80, zorder=5, alpha=0.8)
        ax.annotate(row["category"], (row["residual_real"], row["residual_perm_mean"]),
                    fontsize=8, ha="center", va="bottom", xytext=(0, 5),
                    textcoords="offset points")
        ax.errorbar(row["residual_real"], row["residual_perm_mean"],
                    yerr=row["residual_perm_std"], fmt="none", color="gray",
                    alpha=0.5, capsize=3)

    lims = [min(stats_df["residual_real"].min(), stats_df["residual_perm_mean"].min()) - 0.05,
            max(stats_df["residual_real"].max(), stats_df["residual_perm_mean"].max()) + 0.05]
    ax.plot(lims, lims, "k--", linewidth=0.5, alpha=0.5, label="identity")
    ax.axhline(y=0, color="gray", linewidth=0.3)
    ax.axvline(x=0, color="gray", linewidth=0.3)
    ax.set_xlabel("Real W residual mean", fontsize=11)
    ax.set_ylabel("Permuted W residual mean", fontsize=11)
    ax.set_title(f"Category Pattern: Real vs Permuted\nr={pattern_corr:.3f}", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "experiment0_controls.png", dpi=150)
    print(f"\nSaved plot: {OUTPUT_DIR / 'experiment0_controls.png'}")

    # Save updated data
    df.to_csv(OUTPUT_DIR / "experiment0_controls.csv", index=False)

    # Save summary
    summary = {
        "gauge_invariance_max_diff": float(rot_diff.max()),
        "gauge_invariance_pass": gauge_ok,
        "pattern_correlation_real_vs_perm": float(pattern_corr),
        "category_spread_real": float(real_range),
        "category_spread_perm": float(perm_range),
        "range_ratio": float(range_ratio),
        "n_permutations": N_PERMUTATIONS,
        "verdict": verdict,
        "category_stats": cat_stats,
    }
    with open(OUTPUT_DIR / "experiment0_controls_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n{'='*70}")
    print(f"VERDICT: {verdict}")
    print(f"  Gauge invariance: {'PASS' if gauge_ok else 'FAIL'}")
    print(f"  Pattern correlation (real vs perm): {pattern_corr:.4f}")
    print(f"  Category spread ratio (perm/real): {range_ratio:.2f}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
