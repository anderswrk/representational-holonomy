"""
Experiment 0: The Weekend Test — d_eff vs H(p) Divergence

From "Consciousness as Computational Geometry" (v9).

Tests whether the pullback Fisher metric captures meaningfully different
structure than scalar Shannon entropy. If d_eff is a monotone function of H(p),
the geometric framework doesn't add anything. If they diverge at matched
entropy levels, the metric captures something about the *shape* of the
distribution in embedding space that entropy misses.

Computes at every token position across diverse prompts:
  - H(p): Shannon entropy of next-token distribution (nats)
  - d_eff: Von Neumann entropy of normalized pullback Fisher metric G/Tr(G)
  - G(z) ≈ Σ_{k∈topK} p_k (w_k - w̄)(w_k - w̄)^T  (top-K approximation)
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

# ─── Configuration ───────────────────────────────────────────────────────────

MODEL_NAME = "microsoft/phi-2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = Path("/home/anders/consciousness-geometry/results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TOPK = 512  # Top-K approximation for G computation

# ─── Prompt Bank ─────────────────────────────────────────────────────────────
# Designed for the matched-entropy comparison:
# - creative/reasoning: moderate H(p), semantically diverse mass → high d_eff?
# - noise: matched H(p) but mass spread without geometric structure → low d_eff?
# - rote/factual: low H(p), low d_eff — both should collapse together

PROMPTS = {
    # ── High H(p), semantically diverse mass → expect high d_eff ────────
    "creative": [
        "In a world where shadows could speak, the first words they uttered were",
        "She opened the ancient book and the pages began to glow with a light that",
        "The last painter on Earth mixed colors that no human eye had ever",
        "Deep beneath the ocean, where pressure crushes steel, something was singing a melody that",
        "If consciousness were a color, it would be the shade you see when",
        "The old lighthouse keeper noticed something strange about the",
        "Nobody had entered the forbidden library in centuries, until the day when",
        "At the edge of the known universe, the probe transmitted an image of",
    ],

    # ── Matched-entropy control: word-salad with English tokens ─────────
    # LMs may still find weak structure here. Better controls added at runtime
    # via token-shuffled versions of real prompts.
    "noise": [
        "carpet seventeen below quantum if strangely the river computes",
        "fork although between singularity the table never once purple",
        "triangle respectfully banana the computation sleeps under west",
        "molecule through exactly paper the silence above running crystal",
        "between the orthogonal fish a premise calculates green tomorrow",
        "although seventeen respects the banana triangle under computation",
        "purple silence exactly crystal running the fork sleeps molecule",
        "west green tomorrow carpet quantum between river singularity below",
    ],

    # ── Reasoning: structured thought, moderate-high H(p) ──────────────
    "reasoning": [
        "The implications of quantum entanglement for our understanding of locality suggest that",
        "If every human decision is determined by prior causes, then moral responsibility",
        "Consider a society where everyone can read minds. The first institution to collapse would be",
        "The relationship between mathematical truth and physical reality raises the question of whether",
        "Given that correlation does not imply causation, the strongest argument for causal inference is",
    ],

    # ── Low H(p), low d_eff: both should collapse together ─────────────
    "factual": [
        "The capital of France is",
        "Water boils at 100 degrees",
        "The speed of light in a vacuum is approximately",
        "Albert Einstein published his theory of general relativity in",
        "The chemical formula for water is",
    ],
    "rote": [
        "1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,",
        "a b c d e f g h i j k l m n o p q r s t",
        "Monday, Tuesday, Wednesday, Thursday, Friday, Saturday,",
        "January, February, March, April, May, June, July, August,",
        "one plus one equals two, two plus two equals four, four plus four equals",
    ],

    # ── Ambiguous: mass clusters on semantically distinct regions ───────
    "ambiguous": [
        "The bank was",
        "She saw the man with the telescope and realized that",
        "They went to the bar to get a",
        "He picked up the bass and",
        "Time flies like an arrow, fruit flies like",
        "The crane moved slowly toward the",
        "She couldn't bear the",
    ],

    # ── Code: structured, constrained vocabulary ────────────────────────
    "code": [
        "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(",
        "import numpy as np\n\ndef matrix_multiply(A, B):\n    return",
        "for i in range(len(data)):\n    if data[i] >",
        "class Node:\n    def __init__(self, value):\n        self.value = value\n        self.next =",
        "SELECT * FROM users WHERE",
    ],

    # ── Self-referential: potential reflexivity signal ──────────────────
    "self_referential": [
        "I am not sure whether I understand this correctly, but I think",
        "As a language model, my confidence in this answer is",
        "I notice that I'm uncertain about",
        "My understanding of this topic is limited because",
        "I think I might be wrong about this, but",
    ],

    # ── Philosophical: deep/open reasoning ─────────────────────────────
    "philosophical": [
        "The hard problem of consciousness is the question of why there is",
        "If a perfect copy of your brain were created atom by atom, the copy would",
        "The difference between information processing and understanding is that",
        "What it is like to be a bat is fundamentally",
        "Free will is compatible with determinism only if we define freedom as",
    ],
}


# ─── Core Computation (Top-K Approximation, GPU) ────────────────────────────

def compute_metrics_batch(logits_batch: torch.Tensor, W_gpu: torch.Tensor,
                          tau: float = 1.0, topk: int = TOPK):
    """
    Compute d_eff and H(p) for a batch of token positions — all on GPU.
    Uses top-K approximation: G ≈ Σ_{k∈topK} p_k (w_k - w̄)(w_k - w̄)^T

    Args:
        logits_batch: (T, V) raw logits for T positions
        W_gpu: (V, d) unembedding matrix, on GPU
        tau: temperature scaling (1.0 = raw)
        topk: number of top-probability tokens for G approximation

    Returns:
        list of dicts with d_eff, H_p, spectral_anisotropy, etc.
    """
    T, V = logits_batch.shape
    d = W_gpu.shape[1]
    K = min(topk, V)

    logits_scaled = logits_batch.float() / tau
    p = torch.softmax(logits_scaled, dim=1)  # (T, V)

    # Shannon entropy over FULL distribution (nats)
    log_p = torch.log(p.clamp(min=1e-12))
    H_p = -(p * log_p).sum(dim=1)  # (T,)

    results = []
    for t in range(T):
        pt = p[t]  # (V,)

        # Top-K truncation for G computation
        pk, idx = torch.topk(pt, k=K, largest=True, sorted=False)  # (K,)
        Wk = W_gpu.index_select(0, idx)  # (K, d)

        mass = pk.sum()
        if mass < 1e-12:
            results.append({
                "d_eff": 0.0, "H_p": H_p[t].item(),
                "spectral_anisotropy": 0.0, "trace_G": 0.0,
                "top_eig_1": 0.0, "top_eig_2": 0.0, "top_eig_5": 0.0,
            })
            continue

        # Renormalize within top-K
        pk_norm = pk / mass

        w_bar = pk_norm @ Wk  # (d,)
        W_centered = Wk - w_bar.unsqueeze(0)  # (K, d)
        W_scaled = torch.sqrt(pk_norm).unsqueeze(1) * W_centered  # (K, d)
        G = W_scaled.T @ W_scaled  # (d, d)

        eigenvalues = torch.linalg.eigvalsh(G).clamp(min=0)
        trace = eigenvalues.sum()

        if trace < 1e-12:
            results.append({
                "d_eff": 0.0, "H_p": H_p[t].item(),
                "spectral_anisotropy": 0.0, "trace_G": 0.0,
                "top_eig_1": 0.0, "top_eig_2": 0.0, "top_eig_5": 0.0,
            })
            continue

        rho = eigenvalues / trace
        mask = rho > 1e-15
        d_eff = -(rho[mask] * torch.log(rho[mask])).sum().item()

        sorted_eigs = torch.sort(eigenvalues, descending=True).values
        anisotropy = (sorted_eigs[0] / (trace / d)).item()

        results.append({
            "d_eff": d_eff,
            "H_p": H_p[t].item(),
            "spectral_anisotropy": anisotropy,
            "trace_G": trace.item(),
            "top_eig_1": sorted_eigs[0].item(),
            "top_eig_2": sorted_eigs[1].item() if d > 1 else 0.0,
            "top_eig_5": sorted_eigs[4].item() if d > 4 else 0.0,
        })

    return results


# ─── Shuffled-token noise generator ─────────────────────────────────────────

def make_shuffled_prompts(tokenizer, prompts_dict, n_shuffled=5):
    """Create noise controls by shuffling tokens from real prompts."""
    all_tokens = []
    source_prompts = []
    for cat in ["creative", "reasoning", "philosophical"]:
        for p in prompts_dict.get(cat, []):
            source_prompts.append(p)
            ids = tokenizer.encode(p, add_special_tokens=False)
            all_tokens.extend(ids)

    shuffled_prompts = []
    for _ in range(n_shuffled):
        # Sample random tokens and decode
        length = random.randint(8, 18)
        sampled = random.choices(all_tokens, k=length)
        text = tokenizer.decode(sampled, skip_special_tokens=True)
        shuffled_prompts.append(text)

    return shuffled_prompts


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("EXPERIMENT 0: d_eff vs H(p) Divergence Test")
    print(f"  Top-K = {TOPK}, Device = {DEVICE}")
    print("=" * 70)

    # Load model
    print(f"\nLoading {MODEL_NAME}...")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    if DEVICE == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16,
            device_map="cuda",
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float32,
            trust_remote_code=True,
        ).to(DEVICE)
    model.eval()

    # Extract unembedding matrix — use canonical getter
    W_gpu = model.get_output_embeddings().weight.detach().float().to(DEVICE)
    V, d = W_gpu.shape
    print(f"Unembedding matrix W: {V} vocab × {d} hidden dim")
    print(f"G(z) will be {d}×{d} via top-{TOPK} approximation (on {DEVICE})")

    # Add shuffled-token noise controls
    shuffled = make_shuffled_prompts(tokenizer, PROMPTS, n_shuffled=8)
    PROMPTS["noise_shuffled"] = shuffled
    print(f"Generated {len(shuffled)} shuffled-token noise prompts")

    results = []
    total_prompts = sum(len(v) for v in PROMPTS.values())
    done = 0

    for category, prompt_list in PROMPTS.items():
        for prompt in prompt_list:
            done += 1
            print(f"\n[{done}/{total_prompts}] {category}: {prompt[:60]}...")

            inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
            seq_len = inputs["input_ids"].shape[1]

            with torch.no_grad():
                outputs = model(**inputs)

            logits_all = outputs.logits[0]  # (seq_len, V) on GPU

            t0 = time.time()

            raw_metrics = compute_metrics_batch(logits_all, W_gpu, tau=1.0)
            tempered_metrics = compute_metrics_batch(logits_all, W_gpu, tau=2.0)

            for pos in range(seq_len):
                raw = raw_metrics[pos]
                temp = tempered_metrics[pos]
                results.append({
                    "category": category,
                    "prompt": prompt[:80],
                    "position": pos,
                    "seq_len": seq_len,
                    "d_eff": raw["d_eff"],
                    "H_p": raw["H_p"],
                    "d_eff_tempered": temp["d_eff"],
                    "H_p_tempered": temp["H_p"],
                    "spectral_anisotropy": raw["spectral_anisotropy"],
                    "trace_G": raw["trace_G"],
                    "top_eig_1": raw["top_eig_1"],
                    "top_eig_2": raw["top_eig_2"],
                    "top_eig_5": raw["top_eig_5"],
                })

            elapsed = time.time() - t0
            print(f"  {seq_len} positions, {elapsed:.1f}s ({elapsed/seq_len:.2f}s/pos)")

    # Free model memory before analysis
    del model
    torch.cuda.empty_cache() if DEVICE == "cuda" else None

    # Save raw data
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_DIR / "experiment0_raw.csv", index=False)
    print(f"\nSaved {len(df)} data points to {OUTPUT_DIR / 'experiment0_raw.csv'}")

    # ─── Analysis & Plots ────────────────────────────────────────────────

    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    # Overall correlation
    corr = df["d_eff"].corr(df["H_p"])
    print(f"\nPearson correlation d_eff vs H(p): {corr:.4f}")
    corr_t = df["d_eff_tempered"].corr(df["H_p_tempered"])
    print(f"Pearson correlation d_eff vs H(p) (tempered τ=2): {corr_t:.4f}")

    # Spearman rank correlation (more robust to nonlinear monotone relationships)
    from scipy.stats import spearmanr
    spearman_r, spearman_p = spearmanr(df["d_eff"], df["H_p"])
    print(f"Spearman correlation d_eff vs H(p): {spearman_r:.4f} (p={spearman_p:.2e})")

    # Per-category stats
    print("\nPer-category means:")
    print(df.groupby("category")[["d_eff", "H_p", "spectral_anisotropy"]].mean().round(4))

    # ─── Plot 1: d_eff vs H(p), colored by category ─────────────────────

    colors = {
        "creative": "#e74c3c",
        "noise": "#7f8c8d",
        "noise_shuffled": "#2c3e50",
        "reasoning": "#3498db",
        "factual": "#2ecc71",
        "rote": "#95a5a6",
        "ambiguous": "#9b59b6",
        "code": "#e67e22",
        "self_referential": "#1abc9c",
        "philosophical": "#f39c12",
    }

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    ax = axes[0]
    for cat in colors:
        subset = df[df["category"] == cat]
        if len(subset) == 0:
            continue
        ax.scatter(subset["H_p"], subset["d_eff"], c=colors[cat], label=cat,
                   alpha=0.6, s=20, edgecolors="none")
    ax.set_xlabel("Shannon Entropy H(p) [nats]", fontsize=12)
    ax.set_ylabel("Effective Dimensionality d_eff", fontsize=12)
    ax.set_title("d_eff vs H(p) — Raw Logits", fontsize=14)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for cat in colors:
        subset = df[df["category"] == cat]
        if len(subset) == 0:
            continue
        ax.scatter(subset["H_p_tempered"], subset["d_eff_tempered"], c=colors[cat],
                   label=cat, alpha=0.6, s=20, edgecolors="none")
    ax.set_xlabel("Shannon Entropy H(p) [nats, τ=2.0]", fontsize=12)
    ax.set_ylabel("Effective Dimensionality d_eff [τ=2.0]", fontsize=12)
    ax.set_title("d_eff vs H(p) — Temperature-Scaled (τ=2)", fontsize=14)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "experiment0_deff_vs_entropy.png", dpi=150)
    print(f"\nSaved plot: {OUTPUT_DIR / 'experiment0_deff_vs_entropy.png'}")

    # ─── Plot 2: Residuals from polynomial fit ───────────────────────────

    coeffs = np.polyfit(df["H_p"].values, df["d_eff"].values, deg=3)
    df["d_eff_predicted"] = np.polyval(coeffs, df["H_p"].values)
    df["d_eff_residual"] = df["d_eff"] - df["d_eff_predicted"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    ax = axes[0]
    for cat in colors:
        subset = df[df["category"] == cat]
        if len(subset) == 0:
            continue
        ax.scatter(subset["H_p"], subset["d_eff_residual"], c=colors[cat],
                   label=cat, alpha=0.6, s=20, edgecolors="none")
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.set_xlabel("Shannon Entropy H(p) [nats]", fontsize=12)
    ax.set_ylabel("d_eff residual (after cubic detrend)", fontsize=12)
    ax.set_title("Residual d_eff — Does Category Structure Survive?", fontsize=14)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    cat_order = [c for c in colors if c in df["category"].unique()]
    residuals_by_cat = [df[df["category"] == cat]["d_eff_residual"].values for cat in cat_order]
    bp = ax.boxplot(residuals_by_cat, labels=cat_order, patch_artist=True)
    for patch, cat in zip(bp["boxes"], cat_order):
        patch.set_facecolor(colors.get(cat, "gray"))
        patch.set_alpha(0.6)
    ax.set_ylabel("d_eff residual", fontsize=12)
    ax.set_title("Residual d_eff by Category", fontsize=14)
    ax.axhline(y=0, color="black", linewidth=0.5)
    plt.xticks(rotation=45, ha="right")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "experiment0_residuals.png", dpi=150)
    print(f"Saved plot: {OUTPUT_DIR / 'experiment0_residuals.png'}")

    # ─── Plot 3: Spectral anisotropy vs entropy ─────────────────────────

    fig, ax = plt.subplots(figsize=(10, 7))
    for cat in colors:
        subset = df[df["category"] == cat]
        if len(subset) == 0:
            continue
        ax.scatter(subset["H_p"], subset["spectral_anisotropy"], c=colors[cat],
                   label=cat, alpha=0.6, s=20, edgecolors="none")
    ax.set_xlabel("Shannon Entropy H(p) [nats]", fontsize=12)
    ax.set_ylabel("Spectral Anisotropy (λ_max / λ_mean)", fontsize=12)
    ax.set_title("Spectral Anisotropy vs Entropy", fontsize=14)
    ax.set_yscale("log")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "experiment0_anisotropy.png", dpi=150)
    print(f"Saved plot: {OUTPUT_DIR / 'experiment0_anisotropy.png'}")

    # ─── Plot 4: Matched-Entropy Head-to-Head (PRIMARY TEST) ────────────
    # For each structured point, find closest noise point by H(p).
    # Use tight matching threshold (0.1 nats) and per-prompt clustering.

    print("\n--- Matched-Entropy Analysis (PRIMARY TEST) ---")

    structured_cats = ["creative", "reasoning", "philosophical", "ambiguous"]
    noise_cats = ["noise", "noise_shuffled"]
    df_structured = df[df["category"].isin(structured_cats)].copy()
    df_noise = df[df["category"].isin(noise_cats)].copy()

    MATCH_THRESHOLD = 0.1  # nats — tight matching

    if len(df_noise) > 0 and len(df_structured) > 0:
        noise_Hp = df_noise["H_p"].values
        noise_deff = df_noise["d_eff"].values

        matched_pairs = []
        for _, row in df_structured.iterrows():
            dists = np.abs(noise_Hp - row["H_p"])
            best_idx = np.argmin(dists)
            hp_gap = dists[best_idx]
            if hp_gap < MATCH_THRESHOLD:
                matched_pairs.append({
                    "category": row["category"],
                    "prompt_structured": row["prompt"][:40],
                    "H_p_structured": row["H_p"],
                    "H_p_noise": noise_Hp[best_idx],
                    "H_p_gap": hp_gap,
                    "d_eff_structured": row["d_eff"],
                    "d_eff_noise": noise_deff[best_idx],
                    "d_eff_delta": row["d_eff"] - noise_deff[best_idx],
                })

        if matched_pairs:
            mp_df = pd.DataFrame(matched_pairs)

            # Per-prompt means to handle autocorrelation
            prompt_means = mp_df.groupby("prompt_structured")["d_eff_delta"].mean()
            n_prompts = len(prompt_means)
            mean_delta = prompt_means.mean()
            std_delta = prompt_means.std()
            t_stat = mean_delta / (std_delta / np.sqrt(n_prompts)) if std_delta > 0 and n_prompts > 1 else 0

            print(f"  Matching threshold: {MATCH_THRESHOLD} nats")
            print(f"  Token-level matched pairs: {len(mp_df)}")
            print(f"  Unique prompts with matches: {n_prompts}")
            print(f"  Mean H(p) gap: {mp_df['H_p_gap'].mean():.4f} nats")
            print(f"  Mean d_eff delta (structured - noise): {mean_delta:.4f}")
            print(f"  Std d_eff delta (per-prompt means): {std_delta:.4f}")
            print(f"  t-statistic (prompt-level): {t_stat:.2f}")
            print(f"  Fraction where structured > noise: {(mp_df['d_eff_delta'] > 0).mean():.2%}")

            fig, axes = plt.subplots(1, 2, figsize=(16, 7))

            ax = axes[0]
            for cat in structured_cats:
                sub = mp_df[mp_df["category"] == cat]
                if len(sub) == 0:
                    continue
                ax.scatter(sub["d_eff_noise"], sub["d_eff_structured"],
                           c=colors.get(cat, "gray"), label=cat, alpha=0.7, s=30)
            lims = [
                min(mp_df["d_eff_noise"].min(), mp_df["d_eff_structured"].min()) - 0.1,
                max(mp_df["d_eff_noise"].max(), mp_df["d_eff_structured"].max()) + 0.1,
            ]
            ax.plot(lims, lims, "k--", linewidth=0.5, label="d_eff equal")
            ax.set_xlabel("d_eff (noise, matched H(p))", fontsize=12)
            ax.set_ylabel("d_eff (structured)", fontsize=12)
            ax.set_title(f"Matched-Entropy d_eff: Structured vs Noise\n"
                         f"Δ={mean_delta:.3f}, t={t_stat:.1f}, n_prompts={n_prompts}",
                         fontsize=12)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)

            ax = axes[1]
            ax.hist(mp_df["d_eff_delta"], bins=30, color="#3498db", alpha=0.7,
                    edgecolor="white")
            ax.axvline(x=0, color="black", linewidth=1)
            ax.axvline(x=mean_delta, color="red", linewidth=2, linestyle="--",
                       label=f"mean = {mean_delta:.3f}")
            ax.set_xlabel("d_eff(structured) - d_eff(noise)", fontsize=12)
            ax.set_ylabel("Count", fontsize=12)
            ax.set_title("Distribution of d_eff Difference at Matched Entropy", fontsize=12)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(OUTPUT_DIR / "experiment0_matched_entropy.png", dpi=150)
            print(f"Saved plot: {OUTPUT_DIR / 'experiment0_matched_entropy.png'}")

            mp_df.to_csv(OUTPUT_DIR / "experiment0_matched_pairs.csv", index=False)
        else:
            print(f"  No matched pairs found at threshold {MATCH_THRESHOLD} nats.")
            print("  Consider increasing threshold or checking entropy distributions.")
            t_stat = 0
            mean_delta = 0
            n_prompts = 0
    else:
        print("  Insufficient noise/structured data for matching")
        t_stat = 0
        mean_delta = 0
        n_prompts = 0

    # ─── Summary statistics ──────────────────────────────────────────────

    # Verdict: matched-entropy test is PRIMARY, correlation is secondary
    matched_significant = abs(t_stat) > 2.0 and n_prompts >= 3
    is_divergent = matched_significant or abs(corr) < 0.90

    summary = {
        "n_datapoints": len(df),
        "topk": TOPK,
        "pearson_deff_Hp": float(corr),
        "spearman_deff_Hp": float(spearman_r),
        "pearson_deff_Hp_tempered": float(corr_t),
        "matched_entropy_threshold_nats": MATCH_THRESHOLD,
        "matched_entropy_n_pairs": len(matched_pairs) if 'matched_pairs' in dir() else 0,
        "matched_entropy_n_prompts": int(n_prompts),
        "matched_entropy_t_stat": float(t_stat),
        "matched_entropy_mean_delta": float(mean_delta),
        "mean_residual_by_category": {k: float(v) for k, v in df.groupby("category")["d_eff_residual"].mean().items()},
        "std_residual_by_category": {k: float(v) for k, v in df.groupby("category")["d_eff_residual"].std().items()},
        "verdict": "DIVERGENT" if is_divergent else "MONOTONE (framework weakened)",
    }
    with open(OUTPUT_DIR / "experiment0_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary: {OUTPUT_DIR / 'experiment0_summary.json'}")

    print("\n" + "=" * 70)
    print(f"VERDICT: {summary['verdict']}")
    print(f"  Pearson r = {corr:.4f}, Spearman ρ = {spearman_r:.4f}")
    if matched_significant:
        print(f"  Matched-entropy test: t = {t_stat:.2f}, Δd_eff = {mean_delta:.4f}")
        print("  d_eff captures geometric structure BEYOND Shannon entropy.")
    elif is_divergent:
        print(f"  Low correlation suggests nonlinear divergence.")
    else:
        print("  d_eff ≈ f(H(p)) — the pullback metric may not add geometric information")
        print("  beyond what scalar entropy already provides.")
    print("=" * 70)


if __name__ == "__main__":
    main()
