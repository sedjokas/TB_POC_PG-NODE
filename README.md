# TB PG-NODE: Physics-Guided Neural ODE for Tuberculosis Transmission Modeling

**Proof-of-concept code** for the paper:

> Kasereka S.K., Al Machot F., Kabengele E.M., Kyamakya K.  
> *Compartmental Mathematical Models of TB Transmission: A Critical Survey of Differential Equation Frameworks, Current Gaps, and Perspectives on Physics-Guided Neural ODE.*  
> IEEE Access, 2025 (under review).

---

## Overview

This repository contains the Python scripts and processed DRC data used to produce the PG-NODE proof of concept in the paper.
The model augments a four-compartment SLIT (Susceptible-Latent-Infectious-Treated) ODE with a neural residual to learn time-varying transmission dynamics from TB surveillance data in the Democratic Republic of Congo (2015-2022).

---

## Repository Structure

```
TB_POC_PG-NODE/
│
├── pgnode_drc_tb.py              # Main script: PG-NODE training, calibration,
│                                 #   R0(t) trajectory, bootstrap CI, ARIMA baseline
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
    └── DRC_TB_agesex_quarterly.csv    # Age-sex stratified quarterly data (2015-2017)
```

---

## Requirements

Python 3.8 or higher.

```
pip install torch numpy pandas matplotlib scipy statsmodels
```

No GPU is required. Training runs on CPU in approximately 10 minutes.

---

## Usage

### Step 1 — Download WHO data (optional)

The processed DRC data is already included in `data/`.
To regenerate from the raw WHO CSV files:

1. Download the four CSV files from the [WHO Global TB Programme data portal](https://www.who.int/teams/global-tuberculosis-programme/data) into `data/`.
2. Run:

```bash
python download_who_data.py
```

### Step 2 — Run the PG-NODE proof of concept

```bash
python pgnode_drc_tb.py
```

This script trains two models on the DRC merged dataset:
- **Classical SLIT** (fixed parameters, 800 epochs)
- **PG-NODE** (neural residual augmentation, 1500 epochs)

It produces:
- Time-varying R0(t) trajectory (2015-2022)
- Notification fit and signed residuals vs. ARIMA(1,1,1) baseline
- Residual bootstrap 95% confidence intervals on mechanistic parameters
- SLIT compartment trajectories (S, L, I, T)
- Training loss convergence curves

Output figures are saved in `figures/`.

### Step 3 — Generate architecture figure

```bash
python generate_architecture_fig.py
```

---

## Data Sources

| Source | Description | Period |
|--------|-------------|--------|
| WHO Global TB Programme CSV | Annual incidence, notifications, outcomes | 2015-2022 |
| DRC PNLT National Synthesis Reports | Quarterly case notifications by province | 2015-2017 |

PNLT quarterly counts are scaled to WHO annual totals via proportional temporal disaggregation (Denton 1971) to produce a harmonized merged dataset.

---

## Model Summary

| Component | Description |
|-----------|-------------|
| Mechanistic backbone | Blower-Small SLIT (4 compartments: S, L, I, T) |
| Neural residual | MLP, 1 hidden layer, 32 units, Softplus output |
| Inputs to NN | [S/N, L/N, I/N, cos(2pt), sin(2pt), cos(2pt/7), sin(2pt/7)] |
| ODE solver | Custom RK4 in PyTorch (BPTT, no external adjoint library) |
| Optimizer | Adam, cosine annealing, gradient clipping |
| Training set | n=15 (12 quarterly 2015-2017 + 3 annual 2018-2020) |
| Validation set | n=2 (2021-2022 annual WHO data) |

---

## Key Results

- PG-NODE recovers R0(t) declining from **2.80** (Q1 2015) to **1.44** (2018 trough).
- A COVID-19-associated rebound to **R0 = 1.92** is detected in 2020 without any explicit COVID-19 covariate.
- Validation RMSE: **6.21K** (PG-NODE) vs. **6.80K** (classical SLIT) vs. **13.81K** (ARIMA).
- PG-NODE achieves a **55% RMSE reduction** over the ARIMA baseline.

---

## License

This code is released for research reproducibility under the MIT License.

---

## Contact

**Selain K. Kasereka**  
University of Kinshasa / University of Klagenfurt  
selain.kasereka@unikin.ac.cd
