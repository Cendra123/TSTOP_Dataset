# Hybrid LSTM-DDPG for Two-Stage Transesterification Optimisation

Reference implementation of a three-stage pipeline for optimising the operating
parameters of a two-stage biodiesel transesterification process: a supervised
**LSTM** guidance model, a **Bayesian optimisation** static baseline, and a
**hybrid LSTM-DDPG** reinforcement-learning agent that controls six continuous
process variables.

Code and data for the paper *"A Hybrid LSTM-DDPG Framework for
Operating-Parameter Optimisation of a Two-Stage Transesterification Process."*

---

## What this does

Conversion rate is one of the most economically consequential control variables
in biodiesel production, where feedstock accounts for roughly 70% of production
cost. Classical approaches — response surface methodology, Bayesian optimisation
— find a one-shot optimum but cannot express sequential decision-making,
multi-step constraints, or adaptation to shifting feedstock.

This repository frames the problem as a Markov decision process instead. A
pretrained LSTM supplies a guidance action that is blended with the DDPG policy
through a time-varying weight λ, decayed across three phases so that the agent
transitions from imitation to autonomy.

The six manipulated variables are the methanol/oil molar ratio, temperature, and
reaction time in each of the two stages.

## Headline results

All three methods are evaluated on an **identical held-out test set** through a
shared nearest-neighbour surrogate environment.

| Metric | Bayesian Opt. | Pure DDPG | **Hybrid LSTM-DDPG** |
|---|---|---|---|
| Best conversion (%) | 99.51 | 97.79 | 99.25 |
| Mean conversion (%) | 97.45 | 96.47 | **97.90** |
| Std. dev. (%) | 2.67 | 1.78 | **0.87** |
| Variance | 7.151 | 3.167 | **0.756** |
| Min conversion (%) | 85.49 | 93.52 | **97.01** |
| Evaluations | 80 | 500 × 50 steps | 2,000 × 50 steps |
| Runtime (s) | 41.9 | — | 1,359.9 |

**The honest headline is variance, not peak.** The hybrid agent's contribution
is low-variance operational consistency at near-maximum conversion — it commits
to a near-optimal operating point rather than oscillating between
configurations. BO still reaches a marginally higher single best conversion, and
with far fewer evaluations.

### Where the hybrid agent loses

| Scenario | BO best (%) | Hybrid RL best (%) |
|---|---|---|
| Normal | 99.15 | **99.25** |
| Constraint change (stage-1 temp bound −10%) | **99.31** | 93.01 |
| Feedstock drift (stage-1 MeOH bound −10%) | **99.57** | 99.36 |

Under bound shifts, BO re-optimised inside the modified envelope beats the
**frozen** hybrid policy. The Phase-2 online LSTM updates happen during training
only, so the deployed policy is no more adaptive than a static one. Genuine
online adaptability would require continual learning after deployment; that is
future work, not a claim this repository supports.

## Pipeline

**Stage 1 — LSTM guidance model.** Two stacked LSTM layers (128 hidden units
each) with layer normalisation, then FC(128) → FC(64) → FC(1), dropout 0.2.
Adam, lr 1e-3, weight decay 1e-4, `ReduceLROnPlateau`, MSE loss, batch 32, up to
150 epochs, early stopping at patience 20, gradient clipping at norm 1.0.
Sequence length 10.

Test-set performance: MAE 2.541, RMSE 3.275, MAPE 2.71%, **R² = −0.0003**.

That R² is not a typo. The LSTM is statistically indistinguishable from a
constant-mean predictor at the point-wise level and should not be read as a
high-fidelity surrogate. It is retained purely as a directional warm-start
signal, where relative ordering rather than point-wise accuracy is what matters.
This is a real limitation and is documented as such.

**Stage 2 — Bayesian optimisation baseline.** Nearest-neighbour surrogate
restricted entirely to `X_train`; 10 random initialisations + 70 BO iterations,
`random_state=42`. All 80 proposals are then scored on `X_test`.

**Stage 3 — Hybrid LSTM-DDPG.** Actor and critic each have three 256-unit hidden
layers with batch norm after layers 1 and 2, dropout 0.1, Xavier uniform init.
Actor output is `tanh` scaled by 2.0 (actions bounded to [−2, 2]); the critic
concatenates action after the first batch-norm layer. Actor lr 1e-3, critic lr
2e-3, τ = 0.001, γ = 0.99, replay buffer 100,000, batch 64. Ornstein-Uhlenbeck
exploration noise, σ from 0.2 decaying to 0.01.

Action `a_t = (1 − λ_t)·a_t^RL + λ_t·a_t^LSTM`, with the three-phase schedule:

| Phase | Episodes | λ | Behaviour |
|---|---|---|---|
| 1 | 1–100 | ≈ 0.95 | LSTM dominates exploration |
| 2 | 100–500 | 0.95 → 0.05 | LSTM updated online every 5 episodes on high-reward replay transitions (lr 1e-4) |
| 3 | 500–2,000 | 0.05 | LSTM frozen; DDPG drives the policy |

**Reward.** Linear surplus `α(y_t − y_b)` above the training-set mean, penalty
`−β` below it, bonuses of +10 and +5 for reaching the dataset's top 5% and top
10%, and an action-norm penalty of `−0.1‖a‖` to discourage large swings in the
operating variables.

**Environment.** Nearest-neighbour surrogate over `X_train`. Actions apply a
bounded perturbation `δ = 0.1 × a` to the current state, clipped to [−3, 3] in
scaled space to keep exploration inside the reliable data region.

## Data

21,001 records, six manipulated variables plus conversion rate (%). Outliers
removed with a 3 × IQR filter, then split 70/15/15 with seed 42. `StandardScaler`
is fitted on the training partition only and applied to validation and test, so
there is no leakage. The same split is used by all three methods.

| Variable | Min | Max | Levels | Unit |
|---|---|---|---|---|
| MeOH/oil ratio, stage 1 | 7.0 | 9.0 | 4 | mol/mol |
| Temperature, stage 1 | 80 | 100 | 5 | °C |
| Reaction time, stage 1 | 3.5 | 8.0 | 7 | h |
| MeOH/oil ratio, stage 2 | 3.04 | 4.00 | 6 | mol/mol |
| Temperature, stage 2 | 40 | 80 | 7 | °C |
| Reaction time, stage 2 | 0.5 | 1.5 | 6 | h |
| Conversion (target) | 81.95 | 99.58 | ~21k | % |

Dataset card and download: https://huggingface.co/datasets/CendraCendra/TSTOP_Dataset 

## Installation

```bash
git clone https://github.com/Cendra123/TSTOP_Dataset.git
cd TSTOP_Dataset
pip install -r requirements.txt
```

Requires Python 3.9+, PyTorch, scikit-learn, `bayesian-optimization`, pandas,
numpy, matplotlib, openpyxl. GPU is optional and used automatically when
available.

## Reproducing the results

```bash
python train_lstm.py            # Stage 1 — supervised guidance model
python run_bo.py                # Stage 2 — Bayesian optimisation baseline
python train_hybrid.py          # Stage 3 — hybrid LSTM-DDPG, 2,000 episodes
python train_ddpg.py --lambda 0 # Ablation — pure DDPG, 500 episodes
python evaluate.py              # Comparison tables + dynamic drift scenarios
```

NumPy and PyTorch RNGs are seeded to 42 throughout. Policy evaluation uses 100
deterministic rollouts with exploration noise disabled.

## Known limitations

- **Surrogate fidelity.** The nearest-neighbour environment is only trustworthy
  inside the convex hull of `X_train`. It should be replaced by a
  first-principles kinetic model validated against experimental data.
- **No conversion beyond the dataset maximum.** The converged value *is* the
  dataset ceiling. No discovery beyond the training envelope is claimed.
- **Critic overestimation.** DDPG critic loss grows monotonically to a
  3,000–4,000 moving-average band — the classical Q-value overestimation
  signature. Soft target updates at τ = 0.001 attenuate the spikes but not the
  trend. TD3 or SAC would be the appropriate fix.
- **Weak teacher.** See the Stage-1 R² note above.
- **No experimental validation.** Recommended operating parameters have not been
  tested against a physical reactor.

## Roadmap

- [ ] Replace the nearest-neighbour surrogate with a validated kinetic model
- [ ] TD3 and SAC baselines alongside BO
- [ ] Continual learning at deployment to test genuine online adaptation
- [ ] Constraint-aware reward terms for catalyst lifetime, energy cost, throughput
- [ ] Physical reactor validation

## Citation

```bibtex
@inproceedings{butarbutar2026hybrid,
  title     = {A Hybrid {LSTM-DDPG} Framework for Operating-Parameter
               Optimisation of a Two-Stage Transesterification Process},
  author    = {Butarbutar, Chardinal Martin and
               Widada, Bartolomeus Priya Perkasa Utama and
               Putra, Cendra Devayana and
               Putri, Berliana Devianti and
               Taipabu, Ikhsan},
  booktitle = {TBA},
  year      = {2026}
}
```

The hybrid warm-start architecture is adapted from Fu, Yu and Li, *Multi-Feature
Supervised Reinforcement Learning for Stock Trading*, IEEE Access 11 (2023),
77840–77855.

## Authors

Chardinal Martin Butarbutar (Surabaya State University) ·
Bartolomeus Priya Perkasa Utama Widada (National Cheng Kung University) ·
Cendra Devayana Putra (Surabaya State University, corresponding) ·
Berliana Devianti Putri (Universitas Airlangga) ·
Ikhsan Taipabu (Universitas Pattimura)

## License
MIT is the usual choice for code; CC BY 4.0 for the dataset.
