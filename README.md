# Gauge-Invariant Diagnostics for Hidden Representational Structure in Transformer Language Models

Code and results for testing whether transformer language models contain internal state variation not captured by output distributions, formalized via fiber bundle geometry. All experiments use Microsoft Phi-2 (2.7B parameters, fp16). All return negative results for the hypothesis that gauge-invariant holonomy exists in these representations.

## Key Findings

**Experiment 0 (d_eff vs Entropy):** The effective dimension d_eff of the pullback Fisher metric is a monotone function of Shannon entropy H(p) once spectral artifacts (anisotropy of the embedding matrix) are controlled for. The geometric framework does not capture structure beyond what entropy already provides.

**Experiment 1 (Bank Experiment -- Representational Hysteresis):** Genuine path dependence exists in hidden states: finance-first and nature-first context orderings produce distinguishable internal representations even after a washout paragraph. However, output distributions do not match well enough for the Fisher-Rao gate to pass at strict thresholds, and where they do pass, probe divergence does not exceed control baselines. No evidence of holonomy.

**Experiment 3 (Intermediate Layer Divergence):** Path-dependent divergence is real at intermediate layers but reflects subspace rotation rather than information compression. A linear probe decodes path identity at 100% accuracy at every layer. The LM head projection analysis (Experiment 3b) confirms the model functionally uses this path information -- scrubbing it changes output logits. The divergence is not discarded; it is carried through and used. This is the opposite of what fiber structure requires.

## Repository Structure

```
bank_experiment.py          Experiment 1: representational hysteresis (holonomy test)
                            Also provides shared imports for Experiment 3 scripts
experiment0.py              Experiment 0: d_eff vs Shannon entropy
experiment0_controls.py     Controls for Experiment 0 (row-permutation control)
experiment3_layers.py       Experiment 3: intermediate layer divergence profile
experiment3_controls.py     Controls A-D for Experiment 3
experiment3b_projection.py  Experiment 3b: LM-head projection and recency control
results/                    CSV data, JSON summaries, and PNG plots
```

## Running the Experiments

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For PyTorch with CUDA, you may need to install from the PyTorch index:

```bash
pip install torch==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

Each experiment is a standalone script:

```bash
python experiment0.py              # Experiment 0: d_eff vs entropy
python experiment0_controls.py     # Controls for Experiment 0

python bank_experiment.py          # Experiment 1: bank/holonomy test (run this first)
python experiment3_layers.py       # Experiment 3: layer divergence profile
python experiment3_controls.py     # Controls for Experiment 3
python experiment3b_projection.py  # Experiment 3b: LM-head projection + recency
```

`bank_experiment.py` must run before the Experiment 3 scripts because they import shared data definitions (paragraph lists, washout text, context builder) from it.

Results are written to `results/`.

## Hardware

Developed and tested on an RTX 3080 (10GB VRAM). Phi-2 loads in fp16 (~5GB), leaving headroom for KV caches and intermediate computations. Individual experiments take 2-10 minutes.

## Paper

Manuscript in preparation.

## Citation

```
@misc{consciousness-geometry-2026,
  author = {TODO},
  title  = {Gauge-Invariant Diagnostics for Hidden Representational Structure in Transformer Language Models},
  year   = {2026},
  note   = {Manuscript in preparation}
}
```

## License

TODO
