# Algorithms and Mathematical Formulations

This page documents the mathematically important workflow logic that users need to interpret reports and make scientifically correct decisions.

## Masked Normalized Cross-Correlation

For an experimental pattern $E$ and simulated or reference pattern $S$, masked normalized cross-correlation is used as an interpretable similarity baseline.

For mask $M \in \{0,1\}^{H \times W}$:

$$
\mu_E = \frac{\sum M \odot E}{\sum M}, \qquad
\mu_S = \frac{\sum M \odot S}{\sum M}
$$

$$
\operatorname{NCC}(E,S) =
\frac{\sum M \odot (E-\mu_E)(S-\mu_S)}
{\sqrt{\sum M \odot (E-\mu_E)^2}\sqrt{\sum M \odot (S-\mu_S)^2}}
$$

The point of this formulation is twofold:

- it keeps the scoring intensity-scale-aware and normalized,
- it preserves interpretability as a phase-conditioned evidence measure.

The practical implementation is aligned with the fast normalized cross-correlation family described by Lewis (1995).

## Quality Filtering Semantics

Dataset preparation and sampled `.oh5` inference both evaluate scalar quality criteria before accepting a pixel into downstream logic. In expression form, a typical rule looks like:

$$
\text{accept}(p) =
(\mathrm{CI}(p) > 0.2)\land(\mathrm{Fit}(p) < 1.5)\land \mathrm{Valid}(p)
$$

This is conceptually simple, but scientifically important:

- poor patterns degrade both ML supervision and interpretive confidence,
- filtering early keeps accepted counts and orientation diagnostics meaningful.

## Balanced Split Logic

If accepted phase counts are:

$$
n_{\mathrm{Al}},\ n_{\mathrm{Ni}},\ n_{\mathrm{Cu}},
$$

and balancing is enabled, the selected count per phase becomes:

$$
n_{\text{target}} = \min\left(n_{\mathrm{Al}}, n_{\mathrm{Ni}}, n_{\mathrm{Cu}}\right).
$$

Each phase is then downsampled to $n_{\text{target}}$ before split assignment.

This policy is intentionally conservative. It reduces total data volume, but it avoids training a model that is numerically impressive only because it mainly learns the majority phase.

## Deterministic Split Policy

For split fractions $(r_\text{train}, r_\text{val}, r_\text{test})$ with a fixed seed, the repository uses deterministic assignment so that:

- the same config yields the same split membership,
- benchmark comparisons stay scientifically comparable,
- downstream report interpretation is stable.

When exact per-phase caps are supplied in debug or inspection runs, the split policy becomes:

$$
n_\text{train}^{(k)} = c_\text{train},\quad
n_\text{val}^{(k)} = c_\text{val},\quad
n_\text{test}^{(k)} = c_\text{test}
$$

for each phase $k$, provided enough accepted samples exist.

## ML Classification Semantics

A trained classifier outputs logits $z_k$ for each class $k$. The reported probability vector is:

$$
p_k = \frac{e^{z_k}}{\sum_j e^{z_j}}
$$

and the predicted class is:

$$
\hat{k} = \arg\max_k p_k.
$$

The GUI and sampled-inference tools report confidence as:

$$
\max_k p_k.
$$

This is not a calibrated uncertainty estimate. It is a usable ranking/confidence proxy that helps:

- compare pixels within a scan,
- dull low-confidence pixels in full-scan map mode,
- prioritize manual review regions.

The general deep CNN classification framing follows the modern supervised image-classification pattern popularized by Krizhevsky, Sutskever, and Hinton (2012), though the repository uses EBSD-specific data preparation and evaluation logic rather than generic natural-image assumptions.
