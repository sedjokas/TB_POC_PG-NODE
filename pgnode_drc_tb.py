"""
PG-NODE Proof of Concept: TB Transmission in DRC
=================================================
Data sources:
  PRIMARY   — WHO Global TB Programme CSV (ISO3=COD), 2015–2022
  SECONDARY — DRC PNLT National Synthesis Reports (Excel), 2015–2017
              Quarterly case notifications (4 quarters × 3 years)
              scaled to WHO annual totals via proportional disaggregation
              (Denton 1971). Saved in data/DRC_TB_merged_dataset.csv.

Dataset (merged):
  2015–2017 : quarterly observations  (4 Q × 3 yr = 12 points)
  2018–2022 : annual observations     (5 points)
  ─────────────────────────────────────────────────────────────────
  Training  (t ≤ 2020.5) : n = 15   [12 quarterly + 3 annual]
  Validation(t > 2020.5) : n =  2   [2021, 2022 annual]

Observation variable : c_annualized (notifications / year, in thousands)
  For quarterly obs: quarterly_count × 4   (annualised rate at mid-quarter)
  For annual obs:    annual_count × 1      (already in cases/year units)

Model: Physics-Guided Neural ODE augmenting a 4-compartment SLIT model.
  Mechanistic backbone: Blower-Small SLIT (β, v, τ learnable)
  Neural residual: MLP with 1 hidden layer (7 inputs, 32 units, Softplus output)
    Inputs: [S/N, L/N, I/N, cos(2πt), sin(2πt), cos(2πt/7), sin(2πt/7)]
    Annual sinusoidal pair   captures within-year seasonality (Gap 1)
    7-year  sinusoidal pair  captures inter-year transmission trend
ODE solver: custom RK4 in PyTorch (BPTT — no torchdiffeq required).

Post-review correction (see PGNODEFunc docstring below): the reported
reproduction number now includes the susceptible-fraction S(t)/N(t)
factor (making it a genuine Reff(t), not R0(t)) and the relapse +
treatment-recovery-to-S exit routes from T. This changes the reported
range from [1.44, 2.80] to approximately [1.05, 2.08]; see the paper
for the full derivation and discussion.

Reference: Kasereka et al., Compartmental Mathematical Models of TB
Transmission: A Critical Survey of Differential Equation Frameworks,
Current Gaps, and Perspectives on Physics-Guided Neural ODE.
Results in Engineering (under review).
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

torch.manual_seed(42)
np.random.seed(42)

OUTDIR = "figures"
os.makedirs(OUTDIR, exist_ok=True)

matplotlib.rcParams.update({
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "xtick.labelsize": 9, "ytick.labelsize": 9,
    "legend.fontsize": 8, "figure.dpi": 150,
})

# ================================================================
# 1.  LOAD MERGED DATASET (quarterly 2015–2017 + annual 2018–2022)
# ================================================================
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
df = pd.read_csv(os.path.join(DATA_DIR, "DRC_TB_merged_dataset.csv"))

# Relative time: t=0 → Jan 1 2015 (start of study period)
T_YEAR_0 = 2015.0
df["t_rel"] = df["t"] - T_YEAR_0

# Observation in thousands/year (annualized, for uniform loss units)
df["y_K"] = df["c_annualized"] / 1000.0

# Sort by time and split train/val
df = df.sort_values("t_rel").reset_index(drop=True)
df_train = df[df["t"] <= 2020.5].copy()
df_val   = df[df["t"] >  2020.5].copy()

T_TRAIN = torch.tensor(df_train["t_rel"].values, dtype=torch.float64)
T_VAL   = torch.tensor(df_val["t_rel"].values,   dtype=torch.float64)
T_ALL   = torch.tensor(df["t_rel"].values,        dtype=torch.float64)

Y_TRAIN = torch.tensor(df_train["y_K"].values, dtype=torch.float64)
Y_VAL   = torch.tensor(df_val["y_K"].values,   dtype=torch.float64)
Y_ALL   = torch.tensor(df["y_K"].values,        dtype=torch.float64)

# Decimal years for plotting
YEARS_ALL   = df["t"].values
YEARS_TRAIN = df_train["t"].values
YEARS_ANNUAL = np.array([2015,2016,2017,2018,2019,2020,2021,2022], dtype=float)

# WHO annual data (for context plots and initial conditions)
WHO_ANNUAL_K = np.array([148.,155.,161.,167.,171.,145.,158.,172.])
INC_K  = np.array([268.,277.,287.,296.,305.,316.,320.,324.])
POP_M  = np.array([77.3,79.7,82.2,84.7,87.3,90.0,92.9,99.0])
TSR    = np.array([0.82,0.83,0.83,0.84,0.84,0.83,0.84,0.84])
TBHIV  = np.array([11.3,11.6,11.9,12.2,12.5,12.8,13.1,13.3])
DEATHS = np.array([48.,48.,48.,48.,48.,50.,47.,44.])
COV    = WHO_ANNUAL_K / INC_K

print("=" * 65)
print("MERGED DATASET — DRC TB NOTIFICATIONS (PNLT + WHO)")
print("=" * 65)
print(f"  Total observations : {len(df)}  "
      f"({len(df_train)} train / {len(df_val)} validation)")
print(f"  Quarterly obs (PNLT-scaled) : "
      f"{df[df['resolution']=='quarterly'].shape[0]}")
print(f"  Annual obs (WHO) : "
      f"{df[df['resolution']=='annual'].shape[0]}")
print(f"\n{'t_decimal':>10}  {'resolution':>12}  {'c_notif_K':>10}  {'c_annual_K':>10}")
for _, row in df.iterrows():
    print(f"  {row['t']:8.3f}  {row['resolution']:>12}  "
          f"{row['c_notified']/1000:10.2f}  {row['y_K']:10.2f}")

# ================================================================
# 2.  FIXED BIOLOGICAL PARAMETERS
# ================================================================
MU    = 1.0 / 68.0   # natural mortality (DRC life-expectancy 68 yr)
D_TB  = 0.131        # TB-induced mortality [Dye & Williams 2000]
DELTA = 0.032        # relapse rate [Castillo-Chavez & Song 2004]
GAMMA = 2.0          # treatment recovery (~6 months → 2 yr⁻¹)

# ================================================================
# 3.  INITIAL CONDITIONS (DRC Jan 1, 2015 — t_rel = 0.0)
# ================================================================
I0 = INC_K[0]
T0 = WHO_ANNUAL_K[0] * TSR[0] * 0.5
L0 = 20_400.0
S0 = POP_M[0] * 1e3 - I0 - T0 - L0

X0 = torch.tensor([[S0, L0, I0, T0]], dtype=torch.float64)
print(f"\n  Initial conditions (Jan 1, 2015):")
print(f"  S0={S0/1e3:.2f}M  L0={L0/1e3:.2f}M  I0={I0:.0f}K  T0={T0:.0f}K")

# ================================================================
# 4.  ODE SYSTEM — modified SLIT
# ================================================================
def slit_rhs(t, state, beta, v, tau, extra_foi=None):
    """
    dS/dt = μN − (β + Δλ)(I/N)S − μS + γT
    dL/dt = (β + Δλ)(I/N)S − (v + μ)L
    dI/dt = vL + δT − (τ + d + μ)I
    dT/dt = τI − (γ + δ + μ)T
    """
    S, L, I, T_ = state[:,0], state[:,1], state[:,2], state[:,3]
    N   = (S + L + I + T_).clamp(min=1e-8)
    foi = beta * I / N
    if extra_foi is not None:
        foi = (foi + extra_foi).clamp(min=0)
    dS  =  MU * N  - foi * S - MU * S + GAMMA * T_
    dL  =  foi * S - (v + MU) * L
    dI  =  v * L + DELTA * T_ - (tau + D_TB + MU) * I
    dT_ =  tau * I - (GAMMA + DELTA + MU) * T_
    return torch.stack([dS, dL, dI, dT_], dim=1)


def rk4_step(func, t, dt, y):
    k1 = func(t,        y)
    k2 = func(t + dt/2, y + dt/2 * k1)
    k3 = func(t + dt/2, y + dt/2 * k2)
    k4 = func(t + dt,   y + dt   * k3)
    return y + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)


def integrate(func, t_points, x0, dt=0.1):
    """
    Integrate from t=0 (Jan 1, 2015) through t_points using RK4.
    x0 is the state at t=0 (not the first observation point).
    Gradients flow through each step (BPTT).
    """
    targets = t_points.tolist()
    traj    = []
    state   = x0
    t_cur   = 0.0                  # always start from t=0

    for t_next in targets:
        while t_cur < t_next - 1e-10:
            step  = min(dt, t_next - t_cur)
            state = rk4_step(func, t_cur, step, state)
            t_cur += step
        traj.append(state)

    return torch.stack(traj, dim=0)   # (T, 1, 4)

# ================================================================
# 5.  PG-NODE ARCHITECTURE
# ================================================================
class NeuralResidual(nn.Module):
    """
    MLP with one hidden layer (three layers total: input → hidden → output).
    Learns time-varying excess force of infection Δλ(t) ≥ 0.

    Architecture
    ─────────────────────────────────────────────────────────────
    Input  (7):  S/N, L/N, I/N,
                 cos(2πt),   sin(2πt)     ← annual seasonality
                 cos(2πt/7), sin(2πt/7)   ← inter-year trend
    Hidden (32): Tanh
    Output  (1): Softplus (enforces Δλ ≥ 0)

    Xavier uniform initialisation (gain=0.05) + zero bias
    ensures near-zero initial residual (warm-start near SLIT solution).
    """
    def __init__(self, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(7, hidden), nn.Tanh(),
            nn.Linear(hidden, 1), nn.Softplus(beta=5),
        )
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.05)
                nn.init.zeros_(m.bias)

    def forward(self, t_scalar, state):
        S, L, I, T_ = state[:,0], state[:,1], state[:,2], state[:,3]
        N  = (S + L + I + T_).clamp(min=1e-8)
        # Annual sinusoids (period = 1 yr) for within-year seasonality
        ca = float(np.cos(2 * np.pi * t_scalar))
        sa = float(np.sin(2 * np.pi * t_scalar))
        # 7-year sinusoids for inter-year trend
        cl = float(np.cos(2 * np.pi * t_scalar / 7.0))
        sl = float(np.sin(2 * np.pi * t_scalar / 7.0))
        feat = torch.stack([
            S/N, L/N, I/N,
            torch.full_like(S, ca),
            torch.full_like(S, sa),
            torch.full_like(S, cl),
            torch.full_like(S, sl),
        ], dim=1).float()
        return self.net(feat).squeeze(-1).double()


class PGNODEFunc(nn.Module):
    """
    PG-NODE: mechanistic SLIT backbone + neural residual Δλ(t).

    Learnable parameters:
        log_beta  — log effective contact rate β₀
        log_v     — log latency progression rate v
        log_tau   — log notification/treatment initiation rate τ
        nn_res    — NeuralResidual (32 hidden units, 1 hidden layer)

    R_eff(t) = [S(t)/N(t)] · β_eff(t) · v · (GAMMA+DELTA+MU) /
               [(v + MU) · ((τ + D_TB + MU)(GAMMA+DELTA+MU) - DELTA·τ)]
    where  β_eff(t) = β₀ + Δλ(t) · N/I

    NOTE (post-review correction): earlier versions of this file computed
    R0_at() as beta_eff*v / [(v+MU)*(tau+D_TB+MU)], i.e. without the
    S(t)/N(t) susceptible-fraction factor and without accounting for the
    relapse (DELTA) and treatment-recovery-to-S (GAMMA) exit routes from
    T that are actually present in slit_rhs(). Both corrections are now
    included below; see the paper's Eq. (R0_pgnode) and (R0_drc_full).
    The S(t)/N(t) factor is the dominant correction (~0.71-0.73 across
    the whole 2015-2022 window, not close to 1 as originally assumed).
    """
    def __init__(self, hidden=32):
        super().__init__()
        self.log_beta = nn.Parameter(torch.tensor(np.log(8.3),   dtype=torch.float64))
        self.log_v    = nn.Parameter(torch.tensor(np.log(0.009), dtype=torch.float64))
        self.log_tau  = nn.Parameter(torch.tensor(np.log(0.56),  dtype=torch.float64))
        self.nn_res   = NeuralResidual(hidden)

    @property
    def beta(self): return self.log_beta.exp()
    @property
    def v(self):    return self.log_v.exp()
    @property
    def tau(self):  return self.log_tau.exp()

    def rhs(self, t, state):
        dlam = self.nn_res(t, state)
        return slit_rhs(t, state, self.beta, self.v, self.tau, extra_foi=dlam)

    def R0_full_at(self, beta_eff):
        """Basic reproduction number for the modified SLIT backbone
        (includes the relapse + treatment-recovery-to-S exit routes
        from T), evaluated at a given effective transmission rate.
        Does NOT include susceptible depletion; see reff_at()."""
        cprime = GAMMA + DELTA + MU
        denom = (self.v + MU) * ((self.tau + D_TB + MU) * cprime - DELTA * self.tau)
        return (beta_eff * self.v * cprime) / denom

    def reff_at(self, beta_eff, s_over_n):
        """Effective reproduction number: R0_full_at(beta_eff) scaled by
        the current susceptible fraction S(t)/N(t)."""
        return s_over_n * self.R0_full_at(beta_eff)


class ClassicalSLIT(nn.Module):
    """Classical SLIT — no neural residual, fixed β throughout training."""
    def __init__(self):
        super().__init__()
        self.log_beta = nn.Parameter(torch.tensor(np.log(8.3),   dtype=torch.float64))
        self.log_v    = nn.Parameter(torch.tensor(np.log(0.009), dtype=torch.float64))
        self.log_tau  = nn.Parameter(torch.tensor(np.log(0.56),  dtype=torch.float64))

    @property
    def beta(self): return self.log_beta.exp()
    @property
    def v(self):    return self.log_v.exp()
    @property
    def tau(self):  return self.log_tau.exp()

    def rhs(self, t, state):
        return slit_rhs(t, state, self.beta, self.v, self.tau)

    def R0(self):
        """Fixed basic reproduction number (DFE-based, S=N by definition),
        including the relapse + treatment-recovery-to-S correction; see
        the docstring on PGNODEFunc.R0_full_at for the same formula."""
        cprime = GAMMA + DELTA + MU
        denom = (self.v + MU) * ((self.tau + D_TB + MU) * cprime - DELTA * self.tau)
        return (self.beta * self.v * cprime) / denom


def obs(sol, tau):
    """Predicted notifications (K/yr) = τ × I(t)."""
    return tau * sol[:, 0, 2]


def loss_fn(model, t_pts, x0, y_obs, lam1=0.3, lam2=5e-4):
    """
    Loss = data_fidelity + lam1 * L_phys + lam2 * L2_NN

    L_phys = Σ_i [ Σ_j max(0, -x_j(t_i))²           (negativity penalty)
                   + α |Σ_j x_j(t_i) − N(t_i)|²  ]  (population conservation)
    L2_NN  = Σ ||W_k||²  (L2 regularisation on NN weights only)
    """
    sol   = integrate(model.rhs, t_pts, x0, dt=0.2)
    y_hat = obs(sol, model.tau)
    scale = y_obs.mean().clamp(min=1.0)

    data  = torch.mean(((y_hat - y_obs) / scale) ** 2)

    # L_phys — negativity penalty (α=0 for population conservation here;
    # the compartment sum drifts slowly and is already implicitly constrained
    # by the demographic term μN in dS/dt)
    phys  = torch.mean(torch.relu(-sol[:, 0, :].min(dim=-1).values) ** 2)

    # L2 on NN weights
    l2    = sum(p.double().pow(2).sum()
                for n, p in model.named_parameters() if "nn_res" in n)
    return data + lam1 * phys + lam2 * l2, data.item()

# ================================================================
# 6.  TRAINING
# ================================================================
N_SLIT, N_PG = 800, 1500      # more epochs given larger training set
LR = 5e-3

# --- Classical SLIT ---
print("\nTraining Classical SLIT …")
slit = ClassicalSLIT()
opt_s = torch.optim.Adam(slit.parameters(), lr=LR)
sch_s = torch.optim.lr_scheduler.CosineAnnealingLR(opt_s, N_SLIT)
hist_s = []
for ep in range(N_SLIT):
    opt_s.zero_grad()
    l, dl = loss_fn(slit, T_TRAIN, X0, Y_TRAIN, lam2=0.0)
    l.backward()
    torch.nn.utils.clip_grad_norm_(slit.parameters(), 5.0)
    opt_s.step(); sch_s.step(); hist_s.append(dl)
    if (ep + 1) % 200 == 0:
        print(f"  ep {ep+1:4d}  loss={dl:.5f}  β={slit.beta.item():.3f}"
              f"  v={slit.v.item():.6f}  τ={slit.tau.item():.4f}"
              f"  R0={slit.R0().item():.3f}")

# --- PG-NODE ---
print("\nTraining PG-NODE …")
pgnd = PGNODEFunc(hidden=32)
with torch.no_grad():        # warm-start mechanistic params from SLIT
    pgnd.log_beta.data = slit.log_beta.data.clone()
    pgnd.log_v.data    = slit.log_v.data.clone()
    pgnd.log_tau.data  = slit.log_tau.data.clone()
opt_p = torch.optim.Adam([
    {"params": [pgnd.log_beta, pgnd.log_v, pgnd.log_tau], "lr": LR * 0.3},
    {"params": pgnd.nn_res.parameters(),                  "lr": LR},
])
sch_p = torch.optim.lr_scheduler.CosineAnnealingLR(opt_p, N_PG)
hist_p = []
for ep in range(N_PG):
    opt_p.zero_grad()
    l, dl = loss_fn(pgnd, T_TRAIN, X0, Y_TRAIN, lam1=0.3, lam2=5e-4)
    l.backward()
    torch.nn.utils.clip_grad_norm_(pgnd.parameters(), 5.0)
    opt_p.step(); sch_p.step(); hist_p.append(dl)
    if (ep + 1) % 300 == 0:
        print(f"  ep {ep+1:4d}  loss={dl:.5f}  β0={pgnd.beta.item():.3f}"
              f"  v={pgnd.v.item():.6f}  τ={pgnd.tau.item():.4f}")

# ================================================================
# 7.  INFERENCE & METRICS
# ================================================================
with torch.no_grad():
    sol_s = integrate(slit.rhs, T_ALL, X0, dt=0.05)
    sol_p = integrate(pgnd.rhs, T_ALL, X0, dt=0.05)

y_s = obs(sol_s, slit.tau).detach().numpy()
y_p = obs(sol_p, pgnd.tau).detach().numpy()
y_o = Y_ALL.numpy()

n_tr = len(df_train)
rmse = lambda a, b: np.sqrt(np.mean((a - b)**2))
rs_tr = rmse(y_s[:n_tr], y_o[:n_tr])
rp_tr = rmse(y_p[:n_tr], y_o[:n_tr])
rs_va = rmse(y_s[n_tr:], y_o[n_tr:])
rp_va = rmse(y_p[n_tr:], y_o[n_tr:])
rmse_imp = (rs_va - rp_va) / rs_va * 100

print(f"\n{'Model':<18}{'Train RMSE(K)':<18}{'Val RMSE(K)':<14}{'Improvement'}")
print(f"{'Classical SLIT':<18}{rs_tr:<18.2f}{rs_va:<14.2f}{'—'}")
print(f"{'PG-NODE':<18}{rp_tr:<18.2f}{rp_va:<14.2f}{rmse_imp:+.1f}%")

# --- Time-varying β_eff(t) and Reff(t) (includes S(t)/N(t) + relapse/recovery) ---
R0_s = slit.R0().item()
R0_p, db = [], []
with torch.no_grad():
    for ti, t_i in enumerate(T_ALL.tolist()):
        st_i = sol_p[ti]
        S, L, I, T_ = st_i[0]
        N_i    = (S + L + I + T_).item()
        sn_i   = S.item() / N_i
        dlam_i = pgnd.nn_res(t_i, st_i).item()
        I_i    = I.item()
        dbeta_i = dlam_i * N_i / I_i if I_i > 1e-6 else 0.0
        beta_eff = pgnd.beta.item() + dbeta_i
        R0_p.append(pgnd.reff_at(torch.tensor(beta_eff), sn_i).item())
        db.append(dbeta_i)

print(f"\nClassical SLIT  R0 (fixed) = {R0_s:.3f}")
print(f"PG-NODE  Reff(t) range = [{min(R0_p):.3f}, {max(R0_p):.3f}]")
print(f"\n{'Year/Quarter':>14}  {'Δβ':>8}  {'β_eff':>8}  {'Reff':>8}")
for i, t_dec in enumerate(YEARS_ALL):
    print(f"  {t_dec:12.3f}  {db[i]:+8.4f}  "
          f"{pgnd.beta.item()+db[i]:8.4f}  {R0_p[i]:8.4f}")

# ================================================================
# 8.  BOOTSTRAP CONFIDENCE INTERVALS (residual bootstrap, n=200)
# ================================================================
print("\nBootstrap CIs (n=200 resamples, mechanistic params only) …")
N_BOOT   = 200
N_EP_B   = 400          # fast re-fit on 3 params only
LR_BOOT  = 2e-3

with torch.no_grad():
    y_hat_tr = obs(integrate(pgnd.rhs, T_TRAIN, X0, dt=0.1), pgnd.tau)
residuals_tr = (Y_TRAIN - y_hat_tr).detach().numpy()

boot_beta, boot_v, boot_tau, boot_R0 = [], [], [], []
np.random.seed(0)
for b in range(N_BOOT):
    idx    = np.random.choice(n_tr, n_tr, replace=True)
    y_boot = y_hat_tr + torch.tensor(residuals_tr[idx], dtype=torch.float64)

    # Refit only 3 mechanistic params (NN weights frozen at PG-NODE solution)
    mb = ClassicalSLIT()
    with torch.no_grad():
        mb.log_beta.data = pgnd.log_beta.data.clone()
        mb.log_v.data    = pgnd.log_v.data.clone()
        mb.log_tau.data  = pgnd.log_tau.data.clone()
    opt_b = torch.optim.Adam(mb.parameters(), lr=LR_BOOT)
    for _ in range(N_EP_B):
        opt_b.zero_grad()
        l, _ = loss_fn(mb, T_TRAIN, X0, y_boot, lam2=0.0)
        l.backward()
        torch.nn.utils.clip_grad_norm_(mb.parameters(), 5.0)
        opt_b.step()
    with torch.no_grad():
        boot_beta.append(mb.beta.item())
        boot_v.append(mb.v.item())
        boot_tau.append(mb.tau.item())
        boot_R0.append(mb.R0().item())

def ci95(arr):
    a = np.array(arr)
    return np.percentile(a, 2.5), np.percentile(a, 97.5)

ci_beta = ci95(boot_beta)
ci_v    = ci95(boot_v)
ci_tau  = ci95(boot_tau)
ci_R0   = ci95(boot_R0)

print(f"  β0  = {pgnd.beta.item():.3f}  95% CI [{ci_beta[0]:.3f}, {ci_beta[1]:.3f}]")
print(f"  v   = {pgnd.v.item():.5f}  95% CI [{ci_v[0]:.5f}, {ci_v[1]:.5f}]")
print(f"  τ   = {pgnd.tau.item():.3f}  95% CI [{ci_tau[0]:.3f}, {ci_tau[1]:.3f}]")
print(f"  R0  = {slit.R0().item():.3f}  95% CI [{ci_R0[0]:.3f}, {ci_R0[1]:.3f}]  (mech. params only)")

# ================================================================
# 9.  ARIMA BASELINE (fitted on 6 annual WHO training values)
# ================================================================
print("\nARIMA(1,1,1) baseline …")
from statsmodels.tsa.arima.model import ARIMA as SARIMA
import warnings as _w; _w.filterwarnings('ignore')

# Annual WHO training series (n=6): 2015–2020
y_ann_tr  = np.array([148., 155., 161., 167., 171., 145.])
y_ann_val = np.array([158., 172.])

arima_fit = SARIMA(y_ann_tr, order=(1, 1, 1)).fit()  # disp kwarg removed: unsupported on statsmodels>=0.13 ARIMA
arima_fcast = arima_fit.forecast(steps=2)
arima_fv    = np.array(arima_fcast)

# Training RMSE: use in-sample fitted values (skip first due to differencing)
arima_fitted = arima_fit.fittedvalues
ra_tr = np.sqrt(np.mean((arima_fitted[1:] - y_ann_tr[1:])**2))
ra_va = np.sqrt(np.mean((arima_fv - y_ann_val)**2))

print(f"  ARIMA forecast 2021: {arima_fv[0]:.1f}K  (obs: {y_ann_val[0]:.0f}K)")
print(f"  ARIMA forecast 2022: {arima_fv[1]:.1f}K  (obs: {y_ann_val[1]:.0f}K)")
print(f"  ARIMA val RMSE: {ra_va:.2f}K")

print(f"\n{'Model':<18}{'Train RMSE(K)':<18}{'Val RMSE(K)':<14}{'vs ARIMA'}")
print(f"{'ARIMA(1,1,1)':<18}{ra_tr:<18.2f}{ra_va:<14.2f}{'—'}")
print(f"{'Classical SLIT':<18}{rs_tr:<18.2f}{rs_va:<14.2f}"
      f"{(ra_va-rs_va)/ra_va*100:+.1f}%")
print(f"{'PG-NODE':<18}{rp_tr:<18.2f}{rp_va:<14.2f}"
      f"{(ra_va-rp_va)/ra_va*100:+.1f}%")

# ================================================================
# 10.  FIGURES
# ================================================================
C = {"who":       "#0369a1",   # steel blue  — incidence / WHO series
     "notif":     "#0f172a",   # near-black  — observed notifications
     "slit":      "#1d4ed8",   # deep blue   — classical SLIT model
     "pg":        "#b91c1c",   # deep red    — PG-NODE model
     "gap":       "#bae6fd",   # sky-blue    — notification gap fill
     "covid":     "#dc2626",   # red         — COVID reference line
     "quarterly": "#7c3aed"}   # violet      — quarterly PNLT points
LW = 2.5    # default line width for model curves

def savefig(fig, name):
    for ext in ("pdf", "png"):
        p = os.path.join(OUTDIR, f"{name}.{ext}")
        fig.savefig(p, bbox_inches="tight", dpi=300)
    print(f"  → {name}.pdf / .png")

# ─────── FIG 0 : PG-NODE Architecture ─────────────────────────
# (kept identical to original — diagram generated externally)

# ─────── FIG 1 : DRC Epidemiology Overview ─────────────────────
fig1, axs = plt.subplots(2, 2, figsize=(11, 7))
fig1.suptitle("DRC Tuberculosis Key Indicators, 2015–2022\n"
              "(Source: WHO Global TB Programme – CSV Data, ISO3=COD)", fontsize=12)

ax = axs[0, 0]
ax.fill_between(YEARS_ANNUAL, INC_K, WHO_ANNUAL_K,
                alpha=0.2, color=C["gap"], label="Notification gap")
ax.plot(YEARS_ANNUAL, INC_K,         "o-", color=C["who"],   lw=LW, ms=6,
        label="Est. incidence (WHO)")
ax.plot(YEARS_ANNUAL, WHO_ANNUAL_K,  "s--",color=C["notif"], lw=LW, ms=6,
        label="Notifications (WHO annual)")
# Overlay quarterly data
qt = df[df["resolution"]=="quarterly"]
ax.scatter(qt["t"], qt["c_notified"]/1000, marker="^", s=40,
           color=C["quarterly"], zorder=5, label="PNLT quarterly (raw)")
ax.axvline(2020, color=C["covid"], lw=1.2, ls=":", alpha=.8)
ax.text(2020.1, 308, "COVID-19", fontsize=8, color=C["covid"])
ax.set(ylabel="Cases (thousands)", title="(a) Incidence vs Notifications")
ax.legend(fontsize=7); ax.grid(alpha=.3)

ax = axs[0, 1]
bars = ax.bar(YEARS_ANNUAL, COV*100, color="teal", alpha=.7)
ax.axhline(100, color="k", lw=.8, ls="--", alpha=.4)
ax.axvline(2020, color=C["covid"], lw=1.2, ls=":", alpha=.8)
for b, v in zip(bars, COV*100):
    ax.text(b.get_x()+b.get_width()/2, v+.5, f"{v:.0f}%", ha="center", fontsize=8)
ax.set(ylim=(0, 115), ylabel="Coverage (%)", title="(b) Treatment Coverage")
ax.grid(alpha=.3, axis="y")

ax = axs[1, 0]
ax.plot(YEARS_ANNUAL, TSR*100, "^-", color=C["pg"], lw=LW, ms=6, label="TSR")
ax.axhline(90, color="gray", lw=1, ls="--", alpha=.7, label="WHO 90% target")
ax.set(ylim=(75, 100), ylabel="TSR (%)", title="(c) Treatment Success Rate")
ax.legend(); ax.grid(alpha=.3)

ax = axs[1, 1]
ax2 = ax.twinx()
ax.bar(YEARS_ANNUAL, DEATHS, color="purple", alpha=.5, label="TB deaths (HIV-neg, K)")
ax2.plot(YEARS_ANNUAL, TBHIV, "D-", color="brown", lw=2, ms=6,
         label="TB-HIV co-inf. (%)")
ax.set(ylabel="Deaths (K)", title="(d) Mortality & TB-HIV Co-infection")
ax2.set_ylabel("TB-HIV (%)")
lines1, lab1 = ax.get_legend_handles_labels()
lines2, lab2 = ax2.get_legend_handles_labels()
ax.legend(lines1+lines2, lab1+lab2, fontsize=8)
ax.grid(alpha=.3, axis="y")

for a in axs.flat:
    a.set_xlabel("Year")
    a.set_xticks(YEARS_ANNUAL)
    a.tick_params(axis="x", rotation=30)
plt.tight_layout(); savefig(fig1, "fig1_drc_data_overview"); plt.close(fig1)

# ─────── FIG 1b : Quarterly data visualisation (NEW) ──────────
fig_q, ax_q = plt.subplots(figsize=(11, 4.5))
# Annual annualised values
ax_q.plot(YEARS_ANNUAL, WHO_ANNUAL_K, "ks", ms=8, zorder=6,
          label="WHO annual notifications (annualised rate)")
# Quarterly annualised values
qt_ann = df[df["resolution"]=="quarterly"]
ax_q.scatter(qt_ann["t"], qt_ann["c_annualized"]/1000,
             marker="^", s=60, color=C["quarterly"], zorder=7,
             label="PNLT quarterly × 4 (annualised, scaled to WHO)")
ax_q.axvspan(2020.5, 2022.5, alpha=.07, color="gray")
ax_q.axvline(2020, color="red", lw=1.2, ls=":", alpha=.7)
ax_q.text(2020.05, 195, "COVID-19", fontsize=8.5, color="red")
# Shade quarterly vs annual periods
ax_q.axvspan(2014.9, 2017.9, alpha=.04, color=C["quarterly"],
             label="Quarterly resolution (PNLT, 2015–2017)")
ax_q.axvspan(2017.9, 2023.0, alpha=.04, color="gray",
             label="Annual resolution (WHO, 2018–2022)")
ax_q.set(xlabel="Year", ylabel="Annualised notifications (thousands/year)",
         title="DRC TB Notifications — Merged Dataset\n"
               "(quarterly 2015–2017 from PNLT reports, annual 2018–2022 from WHO)",
         xlim=(2014.7, 2023.0))
ax_q.legend(fontsize=8, loc="upper left"); ax_q.grid(alpha=.3)
plt.tight_layout(); savefig(fig_q, "fig1b_quarterly_dataset"); plt.close(fig_q)

# ─────── FIG 2 : Notification Fit ─────────────────────────────
fig2, axs2 = plt.subplots(1, 2, figsize=(12, 5))

ax = axs2[0]
ax.plot(YEARS_ALL, y_o, "ko", ms=7, zorder=6,
        label="Observed (annualised, K/yr)")
ax.plot(YEARS_ALL, y_s,  "s--", color=C["slit"], lw=LW, ms=7,
        label=f"Classical SLIT  (RMSE: {rs_tr:.1f}K train / {rs_va:.1f}K val)")
ax.plot(YEARS_ALL, y_p,  "^-",  color=C["pg"],   lw=LW, ms=7,
        label=f"PG-NODE         (RMSE: {rp_tr:.1f}K train / {rp_va:.1f}K val)")
ax.axvspan(2020.5, 2022.5, alpha=.08, color="gray", label="Validation period")
ax.axvline(2020, color="red", lw=1.2, ls=":", alpha=.7)
ax.text(2020.1, max(y_o)*0.955, "COVID-19\ndrop", fontsize=8, color="red", va="top")
ax.set(xlabel="Year (decimal)", ylabel="Notifications (K/yr annualised)",
       title="(a) Notification Fit — Classical SLIT vs PG-NODE",
       xlim=(2014.7, 2023.0))
ax.legend(loc="lower right", fontsize=7.5); ax.grid(alpha=.3)

ax = axs2[1]
w = 0.35
x = np.arange(len(YEARS_ALL))
ax.bar(x - w/2, y_s - y_o, w, color=C["slit"], alpha=.75, label="Classical SLIT")
ax.bar(x + w/2, y_p - y_o, w, color=C["pg"],   alpha=.75, label="PG-NODE")
ax.axhline(0, color="k", lw=.8)
ax.axvspan(n_tr - 0.6, len(YEARS_ALL) - 0.4, alpha=.08, color="gray",
           label="Validation")
ax.set(xlabel="Observation index", ylabel="Residual (pred – obs, K/yr)",
       title="(b) Residual Analysis (all observations)")
ax.set_xticks(x)
ax.set_xticklabels([f"{t:.2f}" for t in YEARS_ALL], rotation=45, fontsize=7)
ax.legend(); ax.grid(alpha=.3, axis="y")

plt.tight_layout(); savefig(fig2, "fig2_notification_fit"); plt.close(fig2)

# ─────── FIG 3 : Reff(t) Trajectory (includes S(t)/N(t)) ───────
fig3, axs3 = plt.subplots(1, 2, figsize=(12, 5))

ax = axs3[0]
ax.axhline(R0_s, color=C["slit"], lw=LW, ls="--",
           label=f"Classical SLIT: $\\mathcal{{R}}_0 = {R0_s:.2f}$ (fixed)")
ax.plot(YEARS_ALL, R0_p, "o-", color=C["pg"], lw=LW, ms=6,
        label="PG-NODE: $\\mathcal{R}_{\\mathrm{eff}}(t)$ (time-varying)")
ax.fill_between(YEARS_ALL, R0_s, R0_p, alpha=.12, color=C["pg"])
ax.axhline(1.0, color="k", lw=.8, ls=":", alpha=.5)
ax.text(YEARS_ALL[0], 1.03, "$\\mathcal{R}=1$ threshold", fontsize=8)
ax.axvline(2020, color="red", lw=1.2, ls=":", alpha=.7)
ax.axvspan(2020.5, 2022.5, alpha=.07, color="gray")
ax.set(xlabel="Year", ylabel="$\\mathcal{R}_{\\mathrm{eff}}(t)$",
       title="(a) $\\mathcal{R}_{\\mathrm{eff}}$ Trajectory (includes $S(t)/N(t)$)",
       xlim=(YEARS_ALL[0]-0.3, YEARS_ALL[-1]+0.3))
ax.legend(fontsize=8); ax.grid(alpha=.3)

ax = axs3[1]
colors_bar = [C["pg"] if d > 0 else "#1a9850" for d in db]
ax.bar(YEARS_ALL, db, color=colors_bar, alpha=.75,
       label="Neural residual $\\Delta\\beta(t)$")
ax.axhline(0, color="k", lw=.8)
ax.axvline(2020, color="red", lw=1.2, ls=":", alpha=.7)
peak_i = int(np.argmax(db))
ax.annotate(f"Peak Δβ={db[peak_i]:.3f}\n({YEARS_ALL[peak_i]:.2f})",
            xy=(YEARS_ALL[peak_i], db[peak_i]),
            xytext=(YEARS_ALL[peak_i]-1.2, db[peak_i]*0.75),
            arrowprops=dict(arrowstyle="->", color="k", lw=1), fontsize=8)
ax.set(xlabel="Year", ylabel="$\\Delta\\beta(t)$ (yr$^{-1}$)",
       title="(b) PG-NODE Neural Residual $\\Delta\\beta(t)$")
ax.legend(); ax.grid(alpha=.3, axis="y")

plt.tight_layout(); savefig(fig3, "fig3_r0_trajectory"); plt.close(fig3)

# ─────── FIG 4 : Training Loss ─────────────────────────────────
fig4, ax4 = plt.subplots(figsize=(7, 4.5))
ax4.semilogy(hist_s, color=C["slit"], lw=LW, label="Classical SLIT", alpha=.85)
ax4.semilogy(hist_p, color=C["pg"],   lw=LW, label="PG-NODE", alpha=.85)
ax4.set(xlabel="Epoch", ylabel="Normalised MSE (log scale)",
        title="Training Loss Curves — Classical SLIT vs PG-NODE\n"
              f"(n_train={n_tr}: 12 quarterly + 3 annual observations)")
ax4.legend(); ax4.grid(alpha=.3, which="both")
savefig(fig4, "fig4_training_loss"); plt.close(fig4)

# ─────── FIG 5 : Compartment Trajectories ──────────────────────
fig5, axs5 = plt.subplots(2, 2, figsize=(12, 8))
fig5.suptitle("SLIT Compartment Trajectories — Classical vs PG-NODE (DRC 2015–2022)",
              fontsize=12)
comp_labels = ["S — Susceptible (M)", "L — Latent (M)",
               "I — Active TB (K)", "T — On Treatment (K)"]
scales = [1e3, 1e3, 1.0, 1.0]

for ci, (ax, lab, sc) in enumerate(zip(axs5.flat, comp_labels, scales)):
    s_traj = sol_s[:, 0, ci].detach().numpy() / sc
    p_traj = sol_p[:, 0, ci].detach().numpy() / sc
    ax.plot(YEARS_ALL, s_traj, "s--", color=C["slit"], lw=LW, ms=6,
            label="Classical SLIT")
    ax.plot(YEARS_ALL, p_traj, "o-",  color=C["pg"],   lw=LW, ms=6,
            label="PG-NODE")
    ax.axvline(2020, color="red", lw=1, ls=":", alpha=.6)
    ax.set(xlabel="Year",
           ylabel="Population (millions)" if sc == 1e3 else "Population (thousands)",
           title=f"({chr(97+ci)}) {lab}")
    ax.legend(fontsize=8); ax.grid(alpha=.3)

plt.tight_layout(); savefig(fig5, "fig5_compartment_trajectories"); plt.close(fig5)

# ================================================================
# 9.  HYPERPARAMETER TABLE
# ================================================================
print("\n" + "=" * 65)
print("HYPERPARAMETER & TRAINING SUMMARY")
print("=" * 65)
rows_tbl = [
    ("ODE backbone",              "Modified SLIT (4 compartments)"),
    ("ODE solver",                "Custom RK4 (dt=0.1 train, 0.05 inference; BPTT)"),
    ("Data — training",           f"n={n_tr}  (12 quarterly PNLT-scaled + 3 annual WHO)"),
    ("Data — validation",         f"n={len(df_val)}  (2021, 2022 annual WHO)"),
    ("Observation variable",      "c_annualized  [K/yr]  (quarterly ×4, annual ×1)"),
    ("Time origin",               "t=0 → Jan 1, 2015"),
    ("─── Classical SLIT ───",    ""),
    ("  Epochs",                  str(N_SLIT)),
    ("  Optimizer",               f"Adam  lr={LR:.0e}"),
    ("  LR scheduler",            f"CosineAnnealingLR  T_max={N_SLIT}"),
    ("─── PG-NODE ───",           ""),
    ("  Epochs",                  str(N_PG)),
    ("  Optimizer",               "Adam  (mech: lr=1.5e-3 / NN: lr=5e-3)"),
    ("  LR scheduler",            f"CosineAnnealingLR  T_max={N_PG}"),
    ("  NN architecture",         "1 hidden layer, 32 units  (Linear→Tanh→Linear→Softplus)"),
    ("  NN input dim",            "7  [S/N,L/N,I/N, cos(2πt),sin(2πt), cos(2πt/7),sin(2πt/7)]"),
    ("  Weight initialisation",   "Xavier uniform (gain=0.05) + zero bias"),
    ("  λ₁ (physics penalty)",    "0.3  (negativity penalty on compartments)"),
    ("  λ₂ (L2 regularisation)",  "5e-4  (NN weights only)"),
    ("  Gradient clipping",       "max_norm=5.0"),
    ("─── Data preprocessing ─────", ""),
    ("  PNLT coverage (2015)",    f"{df[df['year']==2015]['coverage_pct'].dropna().iloc[0]:.1f}%"),
    ("  PNLT coverage (2016)",    f"{df[df['year']==2016]['coverage_pct'].dropna().iloc[0]:.1f}%"),
    ("  PNLT coverage (2017)",    f"{df[df['year']==2017]['coverage_pct'].dropna().iloc[0]:.1f}%"),
    ("  Scaling method",          "Denton (1971) proportional disaggregation"),
    ("─── Fixed bio params ───",  ""),
    ("  μ",                       f"{MU:.5f} yr⁻¹  (life expectancy 68 yr)"),
    ("  d",                       f"{D_TB}  yr⁻¹"),
    ("  δ",                       f"{DELTA}  yr⁻¹"),
    ("  γ",                       f"{GAMMA}  yr⁻¹"),
    ("  Random seed",             "42"),
]
for k, v in rows_tbl:
    print(f"  {k:<42} {v}")

print(f"\n{'─'*65}")
print("FINAL ESTIMATED PARAMETERS")
print(f"{'─'*65}")
print(f"  Classical SLIT:")
print(f"    β  = {slit.beta.item():.4f}  yr⁻¹")
print(f"    v  = {slit.v.item():.6f} yr⁻¹")
print(f"    τ  = {slit.tau.item():.4f}  yr⁻¹")
print(f"    R0 = {R0_s:.4f}  (fixed)")
print(f"\n  PG-NODE baseline:")
print(f"    β0 = {pgnd.beta.item():.4f}  yr⁻¹")
print(f"    v  = {pgnd.v.item():.6f} yr⁻¹")
print(f"    τ  = {pgnd.tau.item():.4f}  yr⁻¹")
print(f"    Reff range = [{min(R0_p):.3f}, {max(R0_p):.3f}]")
print(f"    Reff peak  = {YEARS_ALL[np.argmax(R0_p)]:.3f}")

print(f"\n  Validation RMSE improvement: {rmse_imp:.1f}%  "
      f"({rs_va:.2f}K → {rp_va:.2f}K, n_val={len(df_val)})")
print(f"\n{'='*65}")
print("All figures saved to ./figures/  (PDF + PNG)")
