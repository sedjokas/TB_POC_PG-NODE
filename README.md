# TB PG-NODE: Physics-Guided Neural ODE for Tuberculosis Transmission Modeling

**Proof-of-concept code** for the paper:

> Kasereka S.K., Al Machot F., Kabengele E.M., Kyamakya K.
> *Compartmental Mathematical Models of TB Transmission: A Critical Survey of Differential Equation Frameworks, Current Gaps, and Perspectives on Physics-Guided Neural ODE.*
> Results in Engineering (under review).

---

## Overview

This repository contains the Python scripts and processed data used to
produce the PG-NODE proof of concept in the paper (Democratic Republic of
Congo, DRC) and an exploratory external check on a second country
(Mozambique). The model augments a four-compartment SLIT
(Susceptible-Latent-Infectious-Treated) ODE with a neural residual to learn
time-varying transmission dynamics from TB surveillance data.

**Headline finding of the external check:** on Mozambique, with no
country-specific retuning, PG-NODE does *not* outperform the classical SLIT
baseline, and part of the fitted transmission trend is plausibly
attributable to rising case-detection effort rather than transmission
itself. This tempers claims of general cross-country applicability; see
`pgnode_moz_tb.py` and the paper's "Exploratory External Check: Mozambique"
section.

---

## Repository Structure

```
TB_POC_PG-NODE/
│
├── pgnode_drc_tb.py              # Main script: PG-NODE training, calibration,
│                                 #   Reff(t) trajectory, bootstrap CI, ARIMA baseline
│
├── pgnode_moz_tb.py              # Exploratory external check: identical architecture
│                                 #   refit to annual-only Mozambique data (no PNLT-
│                                 #   equivalent quarterly data available)
│
├── download_who_data.py          # Utility: extracts DRC rows from WHO Global TB
│                                 #   Programme CSV downloads (ISO3=COD)
│
├── generate_architecture_fig.py  # Generates the PG-NODE architecture figure
│                                 #   using matplotlib only (no LaTeX required)
│
└── data/
    ├── DRC_TB_burden.csv         # WHO estimated incidence & mortality, DRC 2015-2022
    ├── DRC_TB_notifications.csv  # WHO annual case notifications, DRC 2015-2022
    ├── DRC_TB_outcomes.csv       # WHO treatment outcome data, DRC 2015-2022
    ├── DRC_MDR_burden.csv        # WHO MDR/RR-TB estimates, DRC 2015-2022
    ├── DRC_TB_merged_dataset.csv # Merged training set (12 quarterly + 5 annual obs)
    │                             #   Quarterly 2015-2017: scaled from PNLT reports
    │                             #   Annual 2018-2022: from WHO CSV
    ├── DRC_TB_quarterly_pnlt_raw.csv  # Raw PNLT quarterly counts (before scaling)
    ├── DRC_TB_agesex_quarterly.csv    # Age-sex stratified quarterly data (2015-2017)
    ├── MOZ_TB_burden.csv         # WHO estimated incidence & mortality, Mozambique 2015-2022
    ├── MOZ_TB_notifications.csv  # WHO annual case notifications, Mozambique 2015-2022
    └── MOZ_TB_outcomes.csv       # WHO treatment outcome data, Mozambique 2013-2020
```

---

## Requirements

Python 3.8 or higher.

```
pip install torch numpy pandas matplotlib scipy statsmodels
```

No GPU is required. Each script trains two models (classical SLIT + PG-NODE)
on CPU; the DRC script (bootstrap included) takes longer than the
Mozambique script (no bootstrap).

---

## Usage

### Step 1 — Download WHO data (optional)

The processed DRC and Mozambique data are already included in `data/`.
To regenerate the DRC data from the raw WHO CSV files:

1. Download the four CSV files from the [WHO Global TB Programme data portal](https://www.who.int/teams/global-tuberculosis-programme/data) into `data/`.
2. Run:

```bash
python download_who_data.py
```

Mozambique's `data/MOZ_TB_*.csv` files were extracted the same way, filtering
the WHO bulk CSVs (`generateCSV.asp?ds=notifications`, `ds=estimates`,
`ds=outcomes`) for `iso3=="MOZ"`.

### Step 2 — Run the DRC PG-NODE proof of concept

```bash
python pgnode_drc_tb.py
```

This script trains two models on the DRC merged dataset:
- **Classical SLIT** (fixed parameters, 800 epochs)
- **PG-NODE** (neural residual augmentation, 1500 epochs)

It produces:
- Time-varying $\mathcal{R}_{\mathrm{eff}}(t)$ trajectory (2015-2022)
- Notification fit and signed residuals vs. ARIMA(1,1,1) baseline
- Residual bootstrap 95% confidence intervals on mechanistic parameters
- SLIT compartment trajectories (S, L, I, T)
- Training loss convergence curves

Output figures are saved in `figures/`.

### Step 3 — Run the Mozambique exploratory external check

```bash
python pgnode_moz_tb.py
```

Same architecture and hyperparameters as Step 2, refit to annual-only WHO
data for Mozambique ($n=6$ train / $n=2$ validation, no bootstrap). Output
figures are saved in `figures_moz/`.

### Step 4 — Generate architecture figure

```bash
python generate_architecture_fig.py
```

---

## Data Sources

| Source | Description | Period |
|--------|-------------|--------|
| WHO Global TB Programme CSV | Annual incidence, notifications, outcomes (DRC, Mozambique) | 2015-2022 |
| DRC PNLT National Synthesis Reports | Quarterly case notifications by province (DRC only) | 2015-2017 |

PNLT quarterly counts are scaled to WHO annual totals via proportional temporal disaggregation (Denton 1971) to produce a harmonized merged dataset for DRC. No PNLT-equivalent quarterly reports are available for Mozambique, so that check uses annual WHO data only.

---

## Model Summary

| Component | Description |
|-----------|-------------|
| Mechanistic backbone | Modified Blower-Small SLIT (4 compartments: S, L, I, T; recruitment $\mu N$, treatment exits via relapse to I at rate $\delta$ or recovery to S at rate $\gamma$) |
| Neural residual | MLP, 1 hidden layer, 32 units, Softplus output |
| Inputs to NN | [S/N, L/N, I/N, cos(2pt), sin(2pt), cos(2pt/7), sin(2pt/7)] |
| ODE solver | Custom RK4 in PyTorch (BPTT, no torchdiffeq required) |
| Optimizer | Adam, cosine annealing, gradient clipping |
| DRC training set | n=15 (12 quarterly 2015-2017 + 3 annual 2018-2020) |
| DRC validation set | n=2 (2021-2022 annual WHO data) |
| Mozambique training/validation | n=6 (2015-2020) / n=2 (2021-2022), annual only |

---

## Key Results

### DRC

- PG-NODE recovers $\mathcal{R}_{\mathrm{eff}}(t)$ declining from **2.08** (Q1 2015) to **1.05** (2018 trough), including susceptible-fraction depletion.
- A rebound to $\mathcal{R}_{\mathrm{eff}} = $ **1.40** in 2020 is temporally coincident with, but not established as causally attributable to, COVID-19-related service disruption; no explicit COVID-19 covariate is used.
- Validation RMSE: **6.21K** (PG-NODE) vs. **6.80K** (classical SLIT) vs. **13.81K** (ARIMA).
- PG-NODE achieves an **8.7% RMSE reduction** over classical SLIT and a **55% RMSE reduction** over the ARIMA baseline (the two baselines are not data-equivalent; see paper).

### Mozambique (exploratory external check)

- Classical SLIT: fixed $\mathcal{R}_0 = $ **3.79**. PG-NODE: $\mathcal{R}_{\mathrm{eff}}(t) \in $ **[2.43, 4.62]**.
- Validation RMSE: **4.77K** (classical SLIT) vs. **5.15K** (PG-NODE) — PG-NODE is **7.9% worse**, unlike DRC.
- Mozambique's notification-to-incidence ratio rises from 61% (2015) to 93% (2022), consistent with improving case detection rather than rising transmission; neither model separates the two.

---

## License

This code is released for research reproducibility under the MIT License.

---

## Contact

**Selain K. Kasereka**
University of Kinshasa / University of Klagenfurt
selain.kasereka@unikin.ac.cd
