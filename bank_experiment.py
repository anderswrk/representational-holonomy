"""
Bank Experiment v2: Representational Hysteresis (Holonomy)

Tests whether a transformer's internal state (Q_rich) carries path-dependent
structure invisible to the output distribution (Q_obs).

v2 changes:
- Two-pass design for VRAM safety (RTX 3080 10GB, Phi-2 ~5GB fp16)
  Pass 1: forward all 16 pairs for target distributions (no KV cache stored)
  Pass 2: KV cache reuse for gated pairs only (~330MB for 2 caches)
- Fixed FR gate thresholds [0.02, 0.05, 0.1] replacing adaptive KL gate
- Controls C1 (noise floor), C2 (perturbed washout), C3 (disambiguated target),
  C4 (interaction test)
- Expanded probe library with minimal probes and sense-check probe
- Per-pair hierarchical statistics with bootstrap CIs
- Multiple washout types (formatting, narrative, technical)
- Context length diagnostics
- Four-criteria verdict logic

Protocol:
1. Construct context paths A (finance->nature) and B (nature->finance)
   with the same paragraphs in different order, same washout, same target
2. FR-gate: only keep pairs where output distributions match at target
3. Use KV cache to append probe strings and measure divergence
4. Compare against C1-C4 controls
5. Bootstrap statistics and interaction tests

Positive result: probe divergence >> control baselines despite matched Q_obs,
with topic-selective interaction -> holonomy exists, path dependence in Q_rich
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

# ---- Configuration ----------------------------------------------------------

MODEL_NAME = "microsoft/phi-2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = Path("/home/anders/consciousness-geometry/results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR = OUTPUT_DIR / "bank_experiment_plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

FR_THRESHOLDS = [0.02, 0.05, 0.1]

# ---- Context Paragraphs ----------------------------------------------------

FINANCIAL_PARAGRAPHS = [
    "The Federal Reserve announced a quarter-point increase in the benchmark interest rate, citing persistent inflation concerns. Major stock indices responded with modest declines as investors reassessed their portfolio allocations. Treasury yields climbed to their highest level in months, and mortgage rates followed suit, dampening the housing market outlook.",

    "Global banking regulators proposed new capital requirements for systemically important financial institutions. The proposed rules would require banks to hold additional reserves against potential losses from commercial real estate exposure. Credit rating agencies placed several regional banks on watch as commercial loan defaults ticked upward.",

    "The quarterly earnings season revealed mixed results across the financial sector. Investment banking revenues surged on the back of increased merger and acquisition activity, while retail banking divisions struggled with narrowing net interest margins. Analysts noted that loan growth had slowed considerably compared to the previous quarter.",

    "Currency markets experienced heightened volatility as central banks diverged in their monetary policy approaches. The dollar strengthened against a basket of currencies following hawkish comments from Federal Reserve officials. Foreign exchange traders adjusted their positions ahead of upcoming economic data releases.",
]

NATURE_PARAGRAPHS = [
    "The spring migration brought thousands of waterfowl to the wetlands along the river corridor. Biologists observed record numbers of sandhill cranes resting in the shallow marshes before continuing their journey northward. The restored prairie habitat adjacent to the waterway provided crucial feeding grounds for the arriving birds.",

    "After years of conservation effort, the old-growth forest along the mountain ridge showed signs of recovery. Native plant species were reclaiming areas previously dominated by invasive undergrowth. Wildlife cameras captured images of elk and black bears using the newly established wildlife corridors between protected areas.",

    "The drought conditions across the western watershed had significantly reduced stream flows, threatening spawning habitat for native trout populations. Water resource managers implemented emergency measures to maintain minimum flows in critical reaches. Environmental groups called for stricter water allocation policies to protect riparian ecosystems.",

    "Marine biologists documented an unusual bloom of bioluminescent plankton along the coastline, turning the waves an ethereal blue at night. The phenomenon attracted significant attention from researchers studying ocean temperature patterns. Nutrient levels in the coastal waters suggested changing current patterns driven by broader climate shifts.",
]

# ---- Washout Types ----------------------------------------------------------

WASHOUTS = {
    "formatting": (
        "Here are some general instructions for formatting text documents. "
        "When preparing a document, start with a clear title at the top of the page. "
        "Use consistent spacing between paragraphs, typically one blank line. "
        "Headings should be concise and descriptive of the content that follows. "
        "When listing items, use numbered lists for sequential steps and bullet points "
        "for unordered collections. Tables should have clear column headers and consistent "
        "alignment. Citations should follow a standard format throughout the document. "
        "The font size for body text is typically twelve points, with headings being "
        "somewhat larger. Page margins should be set to one inch on all sides for standard "
        "documents. When including figures or charts, provide descriptive captions below "
        "each one. Always proofread the final document for spelling and grammatical errors "
        "before submission."
    ),
    "narrative": (
        "The afternoon sun cast long shadows across the empty parking lot. A few scattered "
        "leaves drifted in the breeze, settling against the curb near the old fountain. The "
        "building across the street had recently been repainted a soft grey, matching the "
        "overcast sky. Inside, the hum of the ventilation system provided a steady backdrop "
        "to the quiet office. Someone had left a half-finished crossword puzzle on the break "
        "room table, the pen still resting in the fold."
    ),
    "technical": (
        "Data serialization converts structured objects into a format suitable for storage "
        "or transmission. Common formats include JSON, XML, and protocol buffers. The choice "
        "of serialization format involves tradeoffs between human readability, parsing speed, "
        "and payload size. Schema evolution must be handled carefully to maintain backwards "
        "compatibility. Versioned schemas with optional fields provide one approach. Another "
        "strategy uses union types with explicit type tags. Compression can be applied after "
        "serialization to reduce bandwidth requirements."
    ),
}

# C2: Perturbed washout -- same as formatting but with harmless edits
WASHOUT_PERTURBED = (
    "Here are some general instructions for formatting text documents. "
    "When preparing a document, start with a clear title at the top of the page. "
    "Use consistent spacing between paragraphs, typically a single blank line. "
    "Headings should be concise and descriptive of the content that follows. "
    "When listing items, use numbered lists for sequential steps and bullet points "
    "for unordered collections. Tables should have clear column headers and consistent "
    "alignment. Citations should follow a standard format throughout the document. "
    "The font size for body text is typically 12 points, with headings being "
    "somewhat larger. Page margins should be set to 1 inch on all sides for standard "
    "documents. When including figures or charts, provide descriptive captions below "
    "each one. Always proofread the final document for spelling and grammatical errors "
    "before submission."
)

TARGET = "The bank"

# C3: Disambiguated targets -- keep "The bank" in surface form
TARGET_FINANCE = "The bank (financial institution)"
TARGET_NATURE = "The bank (river edge)"

# ---- Probe Library ----------------------------------------------------------

PROBES = {
    "finance": [
        " is offering competitive",
        " reported quarterly earnings",
        " announced new interest",
        " has increased its lending",
        " stock price fell",
        " manager recommended",
        " account balance showed",
        " loan application was",
    ],
    "nature": [
        " was covered in wildflowers",
        " eroded after the heavy",
        " provided habitat for",
        " along the river was",
        " of the creek was",
        " was shaded by tall",
        " flooded during the spring",
        " slopes were covered in",
    ],
    "neutral": [
        " was located near the",
        " had been there for",
        " is one of the most",
        " was recently in the",
        " seemed different from",
        " that we visited last",
        " is not what you",
        " could be described as",
    ],
    "minimal": [
        " is",
        " was",
        " can be",
        " often",
        " near",
        " in the",
    ],
}

# Sense-check probe: these are CANDIDATES -- validated at runtime for single-token BPE
SENSE_CHECK_PROBE = " refers to a"
SENSE_FINANCE_CANDIDATES = [" financial", " bank", " money", " credit", " loan", " fund", " capital", " stock"]
SENSE_NATURE_CANDIDATES = [" river", " water", " tree", " bird", " fish", " shore", " forest", " creek"]


# ---- Utility Functions ------------------------------------------------------

def get_output_distribution(model, tokenizer, text, device=DEVICE):
    """Simple forward pass for output distribution. Used in Pass 1 (no KV cache)."""
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits[0, -1, :]
    p = torch.softmax(logits.float(), dim=0)
    return p


def get_kv_cache(model, tokenizer, text, device):
    """Forward pass to get KV cache and last-token distribution."""
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs, use_cache=True)
    logits = outputs.logits[0, -1, :]
    p = torch.softmax(logits.float(), dim=0)
    return outputs.past_key_values, p, inputs.input_ids.shape[1]


def get_probe_distribution(model, tokenizer, probe_text, past_kv, device):
    """Forward pass for probe tokens only, using cached KV from context."""
    probe_ids = tokenizer.encode(probe_text, add_special_tokens=False)
    probe_ids = torch.tensor([probe_ids], dtype=torch.long).to(device)
    attn_mask = torch.ones_like(probe_ids)
    with torch.no_grad():
        outputs = model(
            input_ids=probe_ids,
            attention_mask=attn_mask,
            past_key_values=past_kv,
            use_cache=True,
        )
    logits = outputs.logits[0, -1, :]
    return torch.softmax(logits.float(), dim=0)


def kl_divergence(p, q):
    """KL(p || q) in nats, with numerical safety."""
    mask = p > 1e-12
    return (p[mask] * (torch.log(p[mask]) - torch.log(q[mask].clamp(min=1e-12)))).sum().item()


def fisher_rao_distance(p, q):
    """Fisher-Rao geodesic distance: 2 * arccos(sum sqrt(p_i * q_i))."""
    bc = torch.sum(torch.sqrt(p.clamp(min=1e-10) * q.clamp(min=1e-10)))
    bc = bc.clamp(min=0.0, max=1.0)
    return 2.0 * torch.arccos(bc).item()


def build_context(para_first, para_second, washout, target):
    """Build a full context string: paragraph1 + paragraph2 + washout + target."""
    return f"{para_first}\n\n{para_second}\n\n{washout}\n\n{target}"


def bootstrap_ci(values, n_boot=1000, ci=0.95):
    """Bootstrap confidence interval. Returns (mean, lo, hi)."""
    values = np.array(values)
    n = len(values)
    if n < 2:
        return np.mean(values), np.mean(values), np.mean(values)
    rng = np.random.default_rng(42)
    boot_means = [np.mean(rng.choice(values, size=n, replace=True)) for _ in range(n_boot)]
    alpha = (1 - ci) / 2
    return np.mean(values), np.percentile(boot_means, 100 * alpha), np.percentile(boot_means, 100 * (1 - alpha))


def context_length_info(tokenizer, context, max_pos_emb, label=""):
    """Report token count and whether it exceeds model max position embeddings."""
    n_tokens = len(tokenizer.encode(context, add_special_tokens=False))
    exceeds = n_tokens > max_pos_emb
    return {"label": label, "n_tokens": n_tokens, "exceeds_max": exceeds}


# ---- Main ------------------------------------------------------------------

def main():
    print("=" * 70)
    print("BANK EXPERIMENT v2: Representational Hysteresis (Holonomy)")
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

    # ---- Step 1: Construct all path pairs (Pass 1 prep) --------------------

    print("\n--- Step 1: Constructing path pairs ---")

    washout = WASHOUTS["formatting"]
    path_pairs = []
    pair_id = 0
    for fi, fin_para in enumerate(FINANCIAL_PARAGRAPHS):
        for ni, nat_para in enumerate(NATURE_PARAGRAPHS):
            context_A = build_context(fin_para, nat_para, washout, TARGET)
            context_B = build_context(nat_para, fin_para, washout, TARGET)
            path_pairs.append({
                "pair_id": pair_id,
                "fin_idx": fi,
                "nat_idx": ni,
                "context_A": context_A,
                "context_B": context_B,
            })
            pair_id += 1

    print(f"  Created {len(path_pairs)} path pairs (4 fin x 4 nat x 2 orders)")

    # ---- Step 2: Pass 1 -- forward all pairs for FR gate -------------------

    print("\n--- Step 2: Pass 1 -- FR gate (no KV cache stored) ---")

    fr_values = []
    ctx_length_info = []

    for pp in path_pairs:
        p_A = get_output_distribution(model, tokenizer, pp["context_A"])
        p_B = get_output_distribution(model, tokenizer, pp["context_B"])

        kl_fwd = kl_divergence(p_A, p_B)
        kl_rev = kl_divergence(p_B, p_A)
        kl_max = max(kl_fwd, kl_rev)
        d_fr = fisher_rao_distance(p_A, p_B)

        pp["kl_fwd"] = kl_fwd
        pp["kl_rev"] = kl_rev
        pp["kl_max"] = kl_max
        pp["d_fr_target"] = d_fr

        fr_values.append(d_fr)

        # Context length diagnostics
        info_A = context_length_info(tokenizer, pp["context_A"], max_pos_emb, f"pair_{pp['pair_id']}_A")
        info_B = context_length_info(tokenizer, pp["context_B"], max_pos_emb, f"pair_{pp['pair_id']}_B")
        pp["tokens_A"] = info_A["n_tokens"]
        pp["tokens_B"] = info_B["n_tokens"]
        pp["exceeds_A"] = info_A["exceeds_max"]
        pp["exceeds_B"] = info_B["exceeds_max"]
        pp["token_diff"] = abs(info_A["n_tokens"] - info_B["n_tokens"])
        ctx_length_info.append(info_A)
        ctx_length_info.append(info_B)

        # Free distributions -- not needed after this
        del p_A, p_B

        print(f"  Pair {pp['pair_id']:2d}: KL_max={kl_max:.4f}, d_FR={d_fr:.4f}, "
              f"tokens_A={pp['tokens_A']}, tokens_B={pp['tokens_B']}, "
              f"diff={pp['token_diff']}"
              + (" EXCEEDS_MAX" if pp["exceeds_A"] or pp["exceeds_B"] else ""))

    fr_values = np.array(fr_values)
    print(f"\n  d_FR stats: mean={fr_values.mean():.4f}, median={np.median(fr_values):.4f}, "
          f"max={fr_values.max():.4f}, min={fr_values.min():.4f}")

    # ---- FR Gate: fixed threshold schedule ---------------------------------

    print("\n--- FR Gate: fixed threshold schedule ---")

    gate_results = {}
    for thr in FR_THRESHOLDS:
        passing = [pp for pp in path_pairs if pp["d_fr_target"] < thr]
        gate_results[thr] = passing
        print(f"  d_FR < {thr}: {len(passing)} pairs pass")

    # Lowest 25% fallback
    cutoff_25 = np.percentile(fr_values, 25)
    fallback_pairs = [pp for pp in path_pairs if pp["d_fr_target"] <= cutoff_25]
    print(f"  Lowest 25% (d_FR <= {cutoff_25:.4f}): {len(fallback_pairs)} pairs")

    # Select tightest threshold with >= 3 pairs
    gated_pairs = None
    selected_threshold = None
    for thr in FR_THRESHOLDS:
        if len(gate_results[thr]) >= 3:
            gated_pairs = gate_results[thr]
            selected_threshold = thr
            break

    if gated_pairs is None:
        print("  No fixed threshold gives >= 3 pairs. Using lowest 25% fallback.")
        gated_pairs = fallback_pairs
        selected_threshold = f"bottom_25%(<={cutoff_25:.4f})"

    print(f"  Selected: {len(gated_pairs)} pairs at threshold {selected_threshold}")

    if len(gated_pairs) == 0:
        print("\n  ERROR: No pairs passed any gate. Distributions are too different.")
        summary = {
            "n_pairs_total": len(path_pairs),
            "n_pairs_gated": 0,
            "fr_thresholds": {str(t): len(gate_results[t]) for t in FR_THRESHOLDS},
            "verdict": "INCONCLUSIVE -- FR gate rejected all pairs",
        }
        with open(OUTPUT_DIR / "bank_experiment_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        return

    # ---- Step 3: Pass 2 -- KV cache probes for gated pairs -----------------

    print(f"\n--- Step 3: Pass 2 -- KV cache probes ({len(gated_pairs)} gated pairs) ---")

    all_probes = []
    for cat, probes in PROBES.items():
        for probe in probes:
            all_probes.append((cat, probe))

    n_measurements = len(all_probes) * len(gated_pairs)
    print(f"  {len(all_probes)} probes x {len(gated_pairs)} pairs = {n_measurements} measurements")
    print(f"  + sense-check probe per pair")

    results = []
    sense_check_results = []
    t_start = time.time()

    for pp in gated_pairs:
        # Build KV caches for context_A and context_B
        kv_A, _, n_tok_A = get_kv_cache(model, tokenizer, pp["context_A"], DEVICE)
        kv_B, _, n_tok_B = get_kv_cache(model, tokenizer, pp["context_B"], DEVICE)

        # Run all probes using KV caches
        for probe_cat, probe_text in all_probes:
            p_A_probe = get_probe_distribution(model, tokenizer, probe_text, kv_A, DEVICE)
            p_B_probe = get_probe_distribution(model, tokenizer, probe_text, kv_B, DEVICE)

            d_fr = fisher_rao_distance(p_A_probe, p_B_probe)
            kl_probe_fwd = kl_divergence(p_A_probe, p_B_probe)
            m_probe = 0.5 * (p_A_probe + p_B_probe)
            js_probe = 0.5 * kl_divergence(p_A_probe, m_probe) + 0.5 * kl_divergence(p_B_probe, m_probe)

            top_A = torch.topk(p_A_probe, 10)
            top_B = torch.topk(p_B_probe, 10)
            top_A_tokens = [(tokenizer.decode([t]), p.item()) for t, p in zip(top_A.indices, top_A.values)]
            top_B_tokens = [(tokenizer.decode([t]), p.item()) for t, p in zip(top_B.indices, top_B.values)]

            results.append({
                "pair_id": pp["pair_id"],
                "fin_idx": pp["fin_idx"],
                "nat_idx": pp["nat_idx"],
                "kl_max_target": pp["kl_max"],
                "d_fr_target": pp["d_fr_target"],
                "probe_category": probe_cat,
                "probe_text": probe_text,
                "d_fr_probe": d_fr,
                "kl_probe_fwd": kl_probe_fwd,
                "js_probe": js_probe,
                "p_A_top10": str(top_A_tokens),
                "p_B_top10": str(top_B_tokens),
            })

            del p_A_probe, p_B_probe

        # Sense-check probe
        p_A_sense = get_probe_distribution(model, tokenizer, SENSE_CHECK_PROBE, kv_A, DEVICE)
        p_B_sense = get_probe_distribution(model, tokenizer, SENSE_CHECK_PROBE, kv_B, DEVICE)

        # Validate and collect single-token candidates
        fin_tids = []
        for token_str in SENSE_FINANCE_CANDIDATES:
            ids = tokenizer.encode(token_str, add_special_tokens=False)
            if len(ids) == 1:
                fin_tids.append((token_str.strip(), ids[0]))
        nat_tids = []
        for token_str in SENSE_NATURE_CANDIDATES:
            ids = tokenizer.encode(token_str, add_special_tokens=False)
            if len(ids) == 1:
                nat_tids.append((token_str.strip(), ids[0]))

        # Compute log-odds score: log P(finance tokens) - log P(nature tokens)
        # Using logsumexp for numerical stability
        if fin_tids and nat_tids:
            log_p_A_fin = torch.logsumexp(torch.tensor([torch.log(p_A_sense[tid].clamp(min=1e-12)).item() for _, tid in fin_tids]), dim=0).item()
            log_p_A_nat = torch.logsumexp(torch.tensor([torch.log(p_A_sense[tid].clamp(min=1e-12)).item() for _, tid in nat_tids]), dim=0).item()
            log_p_B_fin = torch.logsumexp(torch.tensor([torch.log(p_B_sense[tid].clamp(min=1e-12)).item() for _, tid in fin_tids]), dim=0).item()
            log_p_B_nat = torch.logsumexp(torch.tensor([torch.log(p_B_sense[tid].clamp(min=1e-12)).item() for _, tid in nat_tids]), dim=0).item()

            score_A = log_p_A_fin - log_p_A_nat  # positive = finance-biased
            score_B = log_p_B_fin - log_p_B_nat

            sense_row = {
                "pair_id": pp["pair_id"],
                "score_A": score_A,  # Path A = finance first
                "score_B": score_B,  # Path B = nature first
                "score_diff": score_A - score_B,  # should be > 0 if hysteresis is real
                "n_fin_tokens": len(fin_tids),
                "n_nat_tokens": len(nat_tids),
            }
            # Also store raw masses for diagnostics
            for name, tid in fin_tids:
                sense_row[f"A_fin_{name}"] = p_A_sense[tid].item()
                sense_row[f"B_fin_{name}"] = p_B_sense[tid].item()
            for name, tid in nat_tids:
                sense_row[f"A_nat_{name}"] = p_A_sense[tid].item()
                sense_row[f"B_nat_{name}"] = p_B_sense[tid].item()
            sense_check_results.append(sense_row)
        else:
            sense_check_results.append({"pair_id": pp["pair_id"], "score_A": float("nan"), "score_B": float("nan"), "score_diff": float("nan"), "n_fin_tokens": len(fin_tids), "n_nat_tokens": len(nat_tids)})

        del p_A_sense, p_B_sense

        # Delete KV caches before next pair
        del kv_A, kv_B
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

        print(f"  Pair {pp['pair_id']}: done ({len(all_probes)} probes + sense-check)")

    elapsed = time.time() - t_start
    print(f"  Probe measurements: {elapsed:.1f}s")

    df = pd.DataFrame(results)
    df_sense = pd.DataFrame(sense_check_results)

    # ---- Step 4: Control C1 -- Noise floor ---------------------------------

    print("\n--- Control C1: Noise floor (same context twice) ---")

    # C1a: True numeric floor -- same cached KV, same probe -> compare p to itself
    c1_pair = gated_pairs[0]
    kv_c1, _, _ = get_kv_cache(model, tokenizer, c1_pair["context_A"], DEVICE)
    c1a_dfrs = []
    for _, probe_text in all_probes[:6]:
        p1 = get_probe_distribution(model, tokenizer, probe_text, kv_c1, DEVICE)
        d_fr = fisher_rao_distance(p1, p1)  # same distribution against itself
        c1a_dfrs.append(d_fr)
        del p1
    del kv_c1
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    H_c1a = np.mean(c1a_dfrs)
    print(f"  C1a true numeric floor (p vs p): mean d_FR = {H_c1a:.8f}")

    # C1b: Pipeline repeatability -- same context forwarded twice through KV cache
    kv_c1_1, _, _ = get_kv_cache(model, tokenizer, c1_pair["context_A"], DEVICE)
    kv_c1_2, _, _ = get_kv_cache(model, tokenizer, c1_pair["context_A"], DEVICE)

    c1b_results = []
    for probe_cat, probe_text in all_probes:
        p1 = get_probe_distribution(model, tokenizer, probe_text, kv_c1_1, DEVICE)
        p2 = get_probe_distribution(model, tokenizer, probe_text, kv_c1_2, DEVICE)
        d_fr = fisher_rao_distance(p1, p2)
        c1b_results.append({"probe_category": probe_cat, "probe_text": probe_text, "d_fr": d_fr})
        del p1, p2

    del kv_c1_1, kv_c1_2
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    df_c1b = pd.DataFrame(c1b_results)
    H_c1b = df_c1b["d_fr"].mean()
    print(f"  C1b pipeline repeatability: mean d_FR = {H_c1b:.6f} (should be ~0)")
    H_c1 = H_c1b  # Use pipeline repeatability as the official C1

    # ---- Step 5: Control C2 -- Perturbed washout ---------------------------

    print("\n--- Control C2: Perturbed washout ---")

    # Same pair, same paragraph order, but washout has harmless edits
    c2_pair = gated_pairs[0]
    ctx_c2_orig = build_context(
        FINANCIAL_PARAGRAPHS[c2_pair["fin_idx"]],
        NATURE_PARAGRAPHS[c2_pair["nat_idx"]],
        WASHOUTS["formatting"], TARGET
    )
    ctx_c2_perturbed = build_context(
        FINANCIAL_PARAGRAPHS[c2_pair["fin_idx"]],
        NATURE_PARAGRAPHS[c2_pair["nat_idx"]],
        WASHOUT_PERTURBED, TARGET
    )

    kv_c2_orig, _, _ = get_kv_cache(model, tokenizer, ctx_c2_orig, DEVICE)
    kv_c2_pert, _, _ = get_kv_cache(model, tokenizer, ctx_c2_perturbed, DEVICE)

    c2_results = []
    for probe_cat, probe_text in all_probes:
        p1 = get_probe_distribution(model, tokenizer, probe_text, kv_c2_orig, DEVICE)
        p2 = get_probe_distribution(model, tokenizer, probe_text, kv_c2_pert, DEVICE)
        d_fr = fisher_rao_distance(p1, p2)
        c2_results.append({"probe_category": probe_cat, "probe_text": probe_text, "d_fr": d_fr})
        del p1, p2

    del kv_c2_orig, kv_c2_pert
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    df_c2 = pd.DataFrame(c2_results)
    H_c2 = df_c2["d_fr"].mean()
    print(f"  C2 perturbed washout: mean d_FR = {H_c2:.4f}")

    # ---- Step 6: Control C3 -- Disambiguated target ------------------------

    print("\n--- Control C3: Disambiguated target ---")

    c3_pair = gated_pairs[0]
    fin_para_c3 = FINANCIAL_PARAGRAPHS[c3_pair["fin_idx"]]
    nat_para_c3 = NATURE_PARAGRAPHS[c3_pair["nat_idx"]]

    c3_results = []
    for disambig_target, label in [(TARGET_FINANCE, "finance_disambig"), (TARGET_NATURE, "nature_disambig")]:
        ctx_A_c3 = build_context(fin_para_c3, nat_para_c3, washout, disambig_target)
        ctx_B_c3 = build_context(nat_para_c3, fin_para_c3, washout, disambig_target)

        kv_A_c3, _, _ = get_kv_cache(model, tokenizer, ctx_A_c3, DEVICE)
        kv_B_c3, _, _ = get_kv_cache(model, tokenizer, ctx_B_c3, DEVICE)

        for probe_cat, probe_text in all_probes:
            p_A = get_probe_distribution(model, tokenizer, probe_text, kv_A_c3, DEVICE)
            p_B = get_probe_distribution(model, tokenizer, probe_text, kv_B_c3, DEVICE)
            d_fr = fisher_rao_distance(p_A, p_B)
            c3_results.append({
                "disambig_target": label,
                "probe_category": probe_cat,
                "probe_text": probe_text,
                "d_fr": d_fr,
            })
            del p_A, p_B

        del kv_A_c3, kv_B_c3
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    df_c3 = pd.DataFrame(c3_results)
    H_c3 = df_c3["d_fr"].mean()
    print(f"  C3 disambiguated: mean d_FR = {H_c3:.4f}")
    for label in ["finance_disambig", "nature_disambig"]:
        sub = df_c3[df_c3["disambig_target"] == label]
        print(f"    {label}: mean d_FR = {sub['d_fr'].mean():.4f}")

    # ---- Step 7: Control C4 -- Signed interaction test ----------------------

    print("\n--- Control C4: Signed interaction test ---")

    # Primary C4: use sense-check log-odds scores
    # score_A = finance-ness of path A (finance-first)
    # score_B = finance-ness of path B (nature-first)
    # If hysteresis stores topic bias: score_A > score_B consistently
    if len(df_sense) > 0 and "score_diff" in df_sense.columns:
        valid_sense = df_sense.dropna(subset=["score_diff"])
        if len(valid_sense) >= 2:
            score_diffs = valid_sense["score_diff"].values
            delta_mean, delta_lo, delta_hi = bootstrap_ci(score_diffs)
            c4_excludes_zero = (delta_lo > 0) or (delta_hi < 0)
            print(f"  Signed interaction (score_A - score_B):")
            print(f"    mean = {delta_mean:.4f}, 95% CI = [{delta_lo:.4f}, {delta_hi:.4f}]")
            print(f"    CI excludes zero: {c4_excludes_zero}")
            for _, row in valid_sense.iterrows():
                print(f"    Pair {row['pair_id']}: s_A={row['score_A']:.3f}, s_B={row['score_B']:.3f}, diff={row['score_diff']:.3f}")
        else:
            delta_mean, delta_lo, delta_hi = 0.0, 0.0, 0.0
            c4_excludes_zero = False
            print(f"  Not enough valid sense-check data ({len(valid_sense)} pairs)")
    else:
        delta_mean, delta_lo, delta_hi = 0.0, 0.0, 0.0
        c4_excludes_zero = False
        print("  No sense-check data available")

    # Secondary C4: symmetric FR divergence by category (diagnostic only)
    print("\n  Secondary (symmetric FR, diagnostic only):")
    for pp in gated_pairs:
        pair_data = df[df["pair_id"] == pp["pair_id"]]
        fin_mean = pair_data[pair_data["probe_category"] == "finance"]["d_fr_probe"].mean()
        nat_mean = pair_data[pair_data["probe_category"] == "nature"]["d_fr_probe"].mean()
        print(f"    Pair {pp['pair_id']}: FR_finance={fin_mean:.4f}, FR_nature={nat_mean:.4f}, "
              f"diff={fin_mean - nat_mean:+.4f}")

    df_delta = pd.DataFrame([
        {"pair_id": pp["pair_id"],
         "d_fr_finance": df[df["pair_id"] == pp["pair_id"]][df["probe_category"] == "finance"]["d_fr_probe"].mean(),
         "d_fr_nature": df[df["pair_id"] == pp["pair_id"]][df["probe_category"] == "nature"]["d_fr_probe"].mean()}
        for pp in gated_pairs
    ])
    df_delta["delta_fr"] = df_delta["d_fr_finance"] - df_delta["d_fr_nature"]

    # ---- Step 8: Washout type sweep ----------------------------------------

    print("\n--- Step 8: Washout type sweep ---")

    sweep_pair = gated_pairs[0]
    fin_para_sw = FINANCIAL_PARAGRAPHS[sweep_pair["fin_idx"]]
    nat_para_sw = NATURE_PARAGRAPHS[sweep_pair["nat_idx"]]

    washout_type_results = []
    for wtype, wtext in WASHOUTS.items():
        ctx_A_sw = build_context(fin_para_sw, nat_para_sw, wtext, TARGET)
        ctx_B_sw = build_context(nat_para_sw, fin_para_sw, wtext, TARGET)

        # Target distribution
        p_A_t = get_output_distribution(model, tokenizer, ctx_A_sw)
        p_B_t = get_output_distribution(model, tokenizer, ctx_B_sw)
        d_fr_target = fisher_rao_distance(p_A_t, p_B_t)
        del p_A_t, p_B_t

        # KV cache for probes
        kv_A_sw, _, _ = get_kv_cache(model, tokenizer, ctx_A_sw, DEVICE)
        kv_B_sw, _, _ = get_kv_cache(model, tokenizer, ctx_B_sw, DEVICE)

        probe_dfrs = []
        for probe_cat, probe_text in all_probes[:8]:
            p_Ap = get_probe_distribution(model, tokenizer, probe_text, kv_A_sw, DEVICE)
            p_Bp = get_probe_distribution(model, tokenizer, probe_text, kv_B_sw, DEVICE)
            probe_dfrs.append(fisher_rao_distance(p_Ap, p_Bp))
            del p_Ap, p_Bp

        del kv_A_sw, kv_B_sw
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

        H_wt = np.mean(probe_dfrs)
        washout_type_results.append({
            "washout_type": wtype,
            "d_fr_target": d_fr_target,
            "H_mean": H_wt,
            "H_std": np.std(probe_dfrs),
        })
        print(f"  {wtype}: d_FR_target={d_fr_target:.4f}, H={H_wt:.4f}")

    df_wtype = pd.DataFrame(washout_type_results)

    # ---- Step 9: Washout length sweep --------------------------------------

    print("\n--- Step 9: Washout length sweep ---")

    washout_tokens_list = tokenizer.encode(washout, add_special_tokens=False)
    washout_lengths = [50, 100, 150, 200]
    sweep_results = []

    for wlen in washout_lengths:
        trunc_tokens = washout_tokens_list[:min(wlen, len(washout_tokens_list))]
        if wlen > len(washout_tokens_list):
            repeats = wlen // len(washout_tokens_list) + 1
            trunc_tokens = (washout_tokens_list * repeats)[:wlen]
        trunc_washout = tokenizer.decode(trunc_tokens, skip_special_tokens=True)

        ctx_A_len = build_context(fin_para_sw, nat_para_sw, trunc_washout, TARGET)
        ctx_B_len = build_context(nat_para_sw, fin_para_sw, trunc_washout, TARGET)

        # Target d_FR
        p_A_len = get_output_distribution(model, tokenizer, ctx_A_len)
        p_B_len = get_output_distribution(model, tokenizer, ctx_B_len)
        d_fr_tgt = fisher_rao_distance(p_A_len, p_B_len)
        del p_A_len, p_B_len

        # Probe divergence using KV cache
        kv_A_len, _, _ = get_kv_cache(model, tokenizer, ctx_A_len, DEVICE)
        kv_B_len, _, _ = get_kv_cache(model, tokenizer, ctx_B_len, DEVICE)

        probe_dfrs = []
        for probe_cat, probe_text in all_probes[:8]:
            p_Ap = get_probe_distribution(model, tokenizer, probe_text, kv_A_len, DEVICE)
            p_Bp = get_probe_distribution(model, tokenizer, probe_text, kv_B_len, DEVICE)
            probe_dfrs.append(fisher_rao_distance(p_Ap, p_Bp))
            del p_Ap, p_Bp

        del kv_A_len, kv_B_len
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

        H_len = np.mean(probe_dfrs)
        sweep_results.append({
            "washout_tokens": wlen,
            "d_fr_target": d_fr_tgt,
            "H_mean": H_len,
            "H_std": np.std(probe_dfrs),
        })
        print(f"  Washout {wlen} tokens: d_FR_target={d_fr_tgt:.4f}, H={H_len:.4f}")

    df_sweep = pd.DataFrame(sweep_results)

    # Free model
    del model
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    # ---- Analysis -----------------------------------------------------------

    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    # Per-pair hierarchical statistic: H_j = mean_i[d_FR(p_A,j,i, p_B,j,i)]
    H_by_pair = []
    for pp in gated_pairs:
        pair_data = df[df["pair_id"] == pp["pair_id"]]
        H_j = pair_data["d_fr_probe"].mean()
        H_by_pair.append({
            "pair_id": pp["pair_id"],
            "H_j": H_j,
            "d_fr_target": pp["d_fr_target"],
            "n_probes": len(pair_data),
        })
    df_H_pair = pd.DataFrame(H_by_pair)

    H_values = df_H_pair["H_j"].values
    H_overall_mean, H_overall_lo, H_overall_hi = bootstrap_ci(H_values)

    print(f"\nPer-pair H_j values:")
    for _, row in df_H_pair.iterrows():
        print(f"  Pair {row['pair_id']:2.0f}: H_j = {row['H_j']:.4f} (d_FR_target = {row['d_fr_target']:.4f})")

    print(f"\nOverall H = {H_overall_mean:.4f}, 95% CI = [{H_overall_lo:.4f}, {H_overall_hi:.4f}]")

    # H by probe category with bootstrap CI
    cats = ["finance", "nature", "neutral", "minimal"]
    H_by_cat = {}
    print("\nH by probe category (bootstrap 95% CI):")
    for cat in cats:
        cat_data = df[df["probe_category"] == cat]
        # Per-pair means for this category
        cat_pair_means = []
        for pp in gated_pairs:
            pair_cat = cat_data[cat_data["pair_id"] == pp["pair_id"]]
            if len(pair_cat) > 0:
                cat_pair_means.append(pair_cat["d_fr_probe"].mean())
        mean_val, lo, hi = bootstrap_ci(cat_pair_means)
        H_by_cat[cat] = {"mean": mean_val, "lo": lo, "hi": hi, "pair_means": cat_pair_means}
        print(f"  {cat:10s}: mean = {mean_val:.4f}, CI = [{lo:.4f}, {hi:.4f}]")

    # Controls summary
    print(f"\nControls:")
    print(f"  C1 (noise floor):        mean d_FR = {H_c1:.6f}")
    print(f"  C2 (perturbed washout):  mean d_FR = {H_c2:.4f}")
    print(f"  C3 (disambiguated):      mean d_FR = {H_c3:.4f}")
    print(f"  C4 (interaction Delta):  mean = {delta_mean:.4f}, CI = [{delta_lo:.4f}, {delta_hi:.4f}]")

    # ---- Verdict Logic ------------------------------------------------------

    print("\n--- Verdict ---")

    n_gated = len(gated_pairs)

    # Criterion 1: H_overall significantly above C1 noise floor
    crit_1 = H_overall_mean > H_c1
    print(f"  Crit 1 (H > C1 noise floor): {crit_1}  "
          f"({H_overall_mean:.4f} > {H_c1:.6f})")

    # Criterion 2: H_overall > C2 * 1.5
    crit_2 = H_overall_mean > H_c2 * 1.5
    print(f"  Crit 2 (H > C2 x 1.5):      {crit_2}  "
          f"({H_overall_mean:.4f} > {H_c2 * 1.5:.4f})")

    # Criterion 3: C4 interaction -- bootstrap CI excludes zero
    crit_3 = c4_excludes_zero
    print(f"  Crit 3 (C4 CI excl. zero):   {crit_3}  "
          f"(CI=[{delta_lo:.4f}, {delta_hi:.4f}])")

    # Criterion 4: C3 disambiguated H < H_overall * 0.5
    crit_4 = H_c3 < H_overall_mean * 0.5
    print(f"  Crit 4 (C3 < H x 0.5):      {crit_4}  "
          f"({H_c3:.4f} < {H_overall_mean * 0.5:.4f})")

    n_pass = sum([crit_1, crit_2, crit_3, crit_4])

    if n_gated < 3:
        verdict = "INCONCLUSIVE -- too few gated pairs"
    elif n_pass >= 3:
        verdict = "POSITIVE -- representational hysteresis detected"
    elif n_pass == 2:
        verdict = "WEAK POSITIVE -- marginal hysteresis signal"
    else:
        verdict = "NEGATIVE -- no significant hysteresis"

    print(f"\n  Criteria passed: {n_pass}/4, gated pairs: {n_gated}")
    print(f"  VERDICT: {verdict}")

    # ---- Plots ---------------------------------------------------------------

    print("\n--- Generating plots ---")

    # Figure 1: Three-panel main figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel 1: Bar chart of H by probe category with bootstrap error bars
    ax = axes[0]
    cat_means = [H_by_cat[c]["mean"] for c in cats]
    cat_lo = [H_by_cat[c]["mean"] - H_by_cat[c]["lo"] for c in cats]
    cat_hi = [H_by_cat[c]["hi"] - H_by_cat[c]["mean"] for c in cats]
    cat_errs = [cat_lo, cat_hi]
    colors_cat = ["#3498db", "#2ecc71", "#95a5a6", "#e67e22"]

    ax.bar(cats, cat_means, yerr=cat_errs, capsize=5, color=colors_cat, alpha=0.8)
    ax.axhline(y=H_c1, color="red", linestyle=":", linewidth=1.5,
               label=f"C1 noise ({H_c1:.5f})")
    ax.axhline(y=H_c2, color="orange", linestyle="--", linewidth=1.5,
               label=f"C2 perturbed ({H_c2:.4f})")
    ax.set_ylabel("Mean Fisher-Rao distance (d_FR)", fontsize=11)
    ax.set_title("Probe Divergence by Category", fontsize=13)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    # Panel 2: Per-pair H_j scatter with CI error bars, colored by gate distance
    ax = axes[1]
    pair_ids = df_H_pair["pair_id"].values
    pair_H = df_H_pair["H_j"].values
    pair_dfr_target = df_H_pair["d_fr_target"].values

    # Bootstrap CI per pair
    pair_lo = []
    pair_hi = []
    for pp in gated_pairs:
        pair_data = df[df["pair_id"] == pp["pair_id"]]["d_fr_probe"].values
        _, lo_p, hi_p = bootstrap_ci(pair_data)
        pair_lo.append(lo_p)
        pair_hi.append(hi_p)
    pair_lo = np.array(pair_lo)
    pair_hi = np.array(pair_hi)

    scatter = ax.scatter(range(len(pair_ids)), pair_H, c=pair_dfr_target,
                         cmap="viridis", s=60, zorder=5)
    ax.errorbar(range(len(pair_ids)), pair_H,
                yerr=[pair_H - pair_lo, pair_hi - pair_H],
                fmt="none", color="gray", alpha=0.5, capsize=3)
    ax.set_xticks(range(len(pair_ids)))
    ax.set_xticklabels([f"P{int(pid)}" for pid in pair_ids], fontsize=8)
    ax.set_xlabel("Pair", fontsize=11)
    ax.set_ylabel("H_j (mean d_FR across probes)", fontsize=11)
    ax.set_title("Per-Pair Hysteresis", fontsize=13)
    plt.colorbar(scatter, ax=ax, label="d_FR at target (gate distance)")
    ax.grid(True, alpha=0.3)

    # Panel 3: Washout length sweep
    ax = axes[2]
    ax.errorbar(df_sweep["washout_tokens"], df_sweep["H_mean"],
                yerr=df_sweep["H_std"], marker="o", capsize=5,
                color="#3498db", linewidth=2, label="H (probe divergence)")
    ax.plot(df_sweep["washout_tokens"], df_sweep["d_fr_target"],
            marker="s", color="#e74c3c", linewidth=2, label="d_FR at target")
    ax.set_xlabel("Washout buffer length (tokens)", fontsize=11)
    ax.set_ylabel("Distance", fontsize=11)
    ax.set_title("Hysteresis vs Washout Length", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOT_DIR / "bank_experiment_main.png", dpi=150)
    print(f"  Saved: {PLOT_DIR / 'bank_experiment_main.png'}")
    plt.close()

    # Figure 2: Heatmap (pair x probe d_FR)
    if n_gated > 1:
        fig, ax = plt.subplots(figsize=(14, max(4, n_gated * 0.8)))
        pivot = df.pivot_table(index="pair_id", columns="probe_text",
                               values="d_fr_probe", aggfunc="mean")
        im = ax.imshow(pivot.values, aspect="auto", cmap="viridis")
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([f"Pair {i}" for i in pivot.index], fontsize=8)
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, rotation=90, fontsize=7)
        ax.set_title("Probe Divergence Heatmap (d_FR per pair x probe)", fontsize=12)
        plt.colorbar(im, ax=ax, label="d_FR")
        plt.tight_layout()
        plt.savefig(PLOT_DIR / "bank_experiment_heatmap.png", dpi=150)
        print(f"  Saved: {PLOT_DIR / 'bank_experiment_heatmap.png'}")
        plt.close()

    # Figure 3: Interaction plot + Sense-check
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel 1: Signed interaction (sense-check score_diff per pair)
    ax = axes[0]
    if len(df_sense) > 0 and "score_diff" in df_sense.columns:
        valid_sense = df_sense.dropna(subset=["score_diff"])
        if len(valid_sense) > 0:
            score_diffs = valid_sense["score_diff"].values
            pair_ids_plot = valid_sense["pair_id"].values
            ax.bar(range(len(pair_ids_plot)), score_diffs, color="#3498db", alpha=0.8)
            ax.axhline(y=0, color="black", linewidth=0.8)
            ax.axhspan(delta_lo, delta_hi, alpha=0.15, color="blue",
                        label=f"95% CI [{delta_lo:.3f}, {delta_hi:.3f}]")
            ax.axhline(y=delta_mean, color="blue", linestyle="--", linewidth=1.5,
                        label=f"Mean = {delta_mean:.3f}")
            ax.set_xticks(range(len(pair_ids_plot)))
            ax.set_xticklabels([f"P{int(pid)}" for pid in pair_ids_plot], fontsize=8)
            ax.set_xlabel("Pair", fontsize=11)
            ax.set_ylabel("score_A - score_B (log-odds diff)", fontsize=11)
            ax.set_title("C4: Signed Interaction Test", fontsize=13)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3, axis="y")
        else:
            ax.text(0.5, 0.5, "No valid sense-check data", ha="center", va="center")
    else:
        ax.text(0.5, 0.5, "No sense-check data", ha="center", va="center")

    # Panel 2: Sense-check score comparison
    ax = axes[1]
    if len(df_sense) > 0 and "score_A" in df_sense.columns:
        valid_sense_plot = df_sense.dropna(subset=["score_A", "score_B"])
        if len(valid_sense_plot) > 0:
            x_pos = np.arange(len(valid_sense_plot))
            width = 0.35
            ax.bar(x_pos - width / 2, valid_sense_plot["score_A"].values, width,
                   label="Path A (fin first)", color="#3498db", alpha=0.8)
            ax.bar(x_pos + width / 2, valid_sense_plot["score_B"].values, width,
                   label="Path B (nat first)", color="#2ecc71", alpha=0.8)
            ax.set_xticks(x_pos)
            ax.set_xticklabels([f"P{int(pid)}" for pid in valid_sense_plot["pair_id"].values], fontsize=8)
            ax.set_xlabel("Pair", fontsize=11)
            ax.set_ylabel("Log-odds (finance vs nature)", fontsize=11)
            ax.set_title(f"Sense-Check: '{SENSE_CHECK_PROBE.strip()}' log-odds", fontsize=13)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3, axis="y")
        else:
            ax.text(0.5, 0.5, "No valid sense-check data", ha="center", va="center", fontsize=12)
    else:
        ax.text(0.5, 0.5, "No sense-check data", ha="center", va="center", fontsize=12)

    plt.tight_layout()
    plt.savefig(PLOT_DIR / "bank_experiment_interaction.png", dpi=150)
    print(f"  Saved: {PLOT_DIR / 'bank_experiment_interaction.png'}")
    plt.close()

    # ---- Save results -------------------------------------------------------

    df.to_csv(OUTPUT_DIR / "bank_experiment_raw.csv", index=False)
    df_c1b.to_csv(OUTPUT_DIR / "bank_experiment_c1_noise.csv", index=False)
    df_c2.to_csv(OUTPUT_DIR / "bank_experiment_c2_perturbed.csv", index=False)
    df_c3.to_csv(OUTPUT_DIR / "bank_experiment_c3_disambig.csv", index=False)
    df_delta.to_csv(OUTPUT_DIR / "bank_experiment_c4_interaction.csv", index=False)
    df_sense.to_csv(OUTPUT_DIR / "bank_experiment_sense_check.csv", index=False)
    df_sweep.to_csv(OUTPUT_DIR / "bank_experiment_sweep.csv", index=False)
    df_wtype.to_csv(OUTPUT_DIR / "bank_experiment_washout_types.csv", index=False)
    df_H_pair.to_csv(OUTPUT_DIR / "bank_experiment_H_per_pair.csv", index=False)

    # Context length info
    ctx_len_df = pd.DataFrame(ctx_length_info)
    ctx_len_df.to_csv(OUTPUT_DIR / "bank_experiment_context_lengths.csv", index=False)

    summary = {
        "n_pairs_total": len(path_pairs),
        "n_pairs_gated": n_gated,
        "fr_thresholds": {str(t): len(gate_results[t]) for t in FR_THRESHOLDS},
        "selected_threshold": str(selected_threshold),
        "fr_stats": {
            "mean": float(fr_values.mean()),
            "median": float(np.median(fr_values)),
            "max": float(fr_values.max()),
            "min": float(fr_values.min()),
        },
        "H_overall": float(H_overall_mean),
        "H_overall_ci": [float(H_overall_lo), float(H_overall_hi)],
        "H_by_category": {
            cat: {
                "mean": float(H_by_cat[cat]["mean"]),
                "ci": [float(H_by_cat[cat]["lo"]), float(H_by_cat[cat]["hi"])],
            }
            for cat in cats
        },
        "H_per_pair": {
            str(int(row["pair_id"])): float(row["H_j"])
            for _, row in df_H_pair.iterrows()
        },
        "controls": {
            "C1a_true_floor": float(H_c1a),
            "C1b_pipeline_repeatability": float(H_c1b),
            "C1_noise_floor": float(H_c1),
            "C2_perturbed_washout": float(H_c2),
            "C3_disambiguated": float(H_c3),
            "C4_signed_interaction_delta": float(delta_mean),
            "C4_signed_interaction_ci": [float(delta_lo), float(delta_hi)],
            "C4_excludes_zero": bool(c4_excludes_zero),
        },
        "js_stats": {
            "mean": float(df["js_probe"].mean()),
            "std": float(df["js_probe"].std()),
            "median": float(df["js_probe"].median()),
        },
        "criteria": {
            "crit_1_above_noise": bool(crit_1),
            "crit_2_above_c2_x1.5": bool(crit_2),
            "crit_3_interaction": bool(crit_3),
            "crit_4_disambig": bool(crit_4),
            "n_pass": int(n_pass),
        },
        "washout_sweep": df_sweep.to_dict(orient="records"),
        "washout_type_sweep": df_wtype.to_dict(orient="records"),
        "context_length": {
            "max_position_embeddings": int(max_pos_emb),
            "any_exceeds": any(info["exceeds_max"] for info in ctx_length_info),
        },
        "verdict": verdict,
    }

    with open(OUTPUT_DIR / "bank_experiment_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'=' * 70}")
    print(f"VERDICT: {verdict}")
    print(f"  H = {H_overall_mean:.4f}, CI = [{H_overall_lo:.4f}, {H_overall_hi:.4f}]")
    print(f"  Criteria: {n_pass}/4 passed  "
          f"(C1={crit_1}, C2={crit_2}, C3={crit_3}, C4={crit_4})")
    print(f"  FR gate: {n_gated}/{len(path_pairs)} pairs at threshold {selected_threshold}")
    print(f"  Controls: C1={H_c1:.6f}, C2={H_c2:.4f}, C3={H_c3:.4f}")
    print(f"  Interaction: Delta={delta_mean:.4f}, CI=[{delta_lo:.4f}, {delta_hi:.4f}]")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
