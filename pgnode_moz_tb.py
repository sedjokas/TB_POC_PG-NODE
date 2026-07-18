"""
PG-NODE Exploratory External Check: TB Transmission in Mozambique
==================================================================
Companion script to pgnode_drc_tb.py. Refits the identical mechanistic
backbone and PG-NODE architecture (no country-specific retuning) to
annual-only WHO Global TB Programme data for Mozambique (ISO3=MOZ),
2015-2022, as an exploratory external check on the DRC proof of concept.

Data sources:
  WHO Global TB Programme CSV (ISO3=MOZ), 2015-2022
  data/MOZ_TB_burden.csv, data/MOZ_TB_notifications.csv,
  data/MOZ_TB_outcomes.csv

No PNLT-equivalent quarterly national-programme reports are available
for Mozambique, so training uses only n=6 annual points (2015-2020)
with n=2 validation points (2021-2022), matching the DRC study window.
The residual bootstrap used for DRC is NOT repeated here (time budget);
this is a lighter-weight check, not a controlled multi-country benchmark.

Key finding (see paper Section "Exploratory External Check: Mozambique"):
PG-NODE does NOT outperform the classical SLIT baseline on held-out
validation RMSE here (unlike DRC), and Mozambique's own notification-
to-incidence ratio rises substantially over the study period (61% in
2015 to 93% in 2022) -- a pattern more consistent with improving case
detection than with rising transmission, which neither model variant
can separate from transmission in its observation equation
(y = tau * I(t)). This tempers claims of general cross-country
applicability and motivates an explicit observation/reporting model
as a priority extension (see paper Limitation 8).

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
import warnings
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

torch.manual_seed(42)
np.random.seed(42)

OUTDIR = "figures_moz"
os.makedirs(OUTDIR, exist_ok=True)
matplotlib.rcParams.update({
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "xtick.labelsize": 9, "ytick.labelsize": 9,
    "legend.fontsize": 8, "figure.dpi": 150,
})

# ================================================================
# 1.  LOAD MOZAMBIQUE DATA (annual only, 2015-2022)
# ================================================================
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
burden = pd.read_csv(os.path.join(DATA_DIR, "MOZ_TB_burden.csv"))
notif  = pd.read_csv(os.path.join(DATA_DIR, "MOZ_TB_notifications.csv"))
outcomes = pd.read_csv(os.path.join(DATA_DIR, "MOZ_TB_outcomes.csv"))

df = burden.merge(notif, on=["country", "iso3", "year"])
df = df.sort_values("year").reset_index(drop=True)

YEARS = df["year"].tolist()
POP   = df["e_pop_num"].tolist()
INC   = df["e_inc_num"].tolist()
NOTIF = df["c_notified"].tolist()
TSR0  = outcomes[outcomes.year == YEARS[0]]["c_new_tsr"].iloc[0] / 100.0

print("=" * 65)
print("MOZAMBIQUE ANNUAL DATA (WHO Global TB Programme)")
print("=" * 65)
print(df[["year", "e_pop_num", "e_inc_num", "c_notified"]].to_string(index=False))
notif_inc_ratio = [n / i for n, i in zip(NOTIF, INC)]
print("\nNotification-to-incidence ratio by year:")
for y, r in zip(YEARS, notif_inc_ratio):
    print(f"  {y}: {r:.1%}")

# ================================================================
# 2.  FIXED BIOLOGICAL PARAMETERS (same structural constants as DRC;
#     MU uses Mozambique life expectancy)
# ================================================================
MU_LIFE_YEARS = 61.0   # approx. WHO life expectancy for Mozambique
MU    = 1.0 / MU_LIFE_YEARS
D_TB  = 0.131
DELTA = 0.032
GAMMA = 2.0

# ================================================================
# 3.  INITIAL CONDITIONS (Jan 1, 2015 -- t_rel = 0.0)
#     Same L0/population ratio used for DRC (0.264), applied here for
#     consistency; this is a modeling assumption, not a fitted value.
# ================================================================
I0 = INC[0] / 1000.0
T0 = (NOTIF[0] / 1000.0) * TSR0 * 0.5
L0 = 0.264 * (POP[0] / 1000.0)
S0 = POP[0] / 1000.0 - I0 - T0 - L0
X0 = torch.tensor([[S0, L0, I0, T0]], dtype=torch.float64)
print(f"\nInitial conditions (Jan 1, {YEARS[0]}):")
print(f"S0={S0/1e3:.2f}M  L0={L0/1e3:.2f}M  I0={I0:.0f}K  T0={T0:.0f}K")

# ================================================================
# 4.  MERGED TRAINING DATASET (annual only: n=6 train / n=2 val)
# ================================================================
y_all_K = np.array(NOTIF) / 1000.0
t_rel = np.array([y - YEARS[0] + 0.5 for y in YEARS])  # mid-year annual points
N_TRAIN = 6  # 2015-2020
T_TRAIN = torch.tensor(t_rel[:N_TRAIN], dtype=torch.float64)
T_ALL   = torch.tensor(t_rel, dtype=torch.float64)
Y_TRAIN = torch.tensor(y_all_K[:N_TRAIN], dtype=torch.float64)
Y_ALL   = torch.tensor(y_all_K, dtype=torch.float64)

# ================================================================
# 5.  ODE SYSTEM, PG-NODE ARCHITECTURE (identical to pgnode_drc_tb.py)
# ================================================================
def slit_rhs(t, state, beta, v, tau, extra_foi=None):
    S, L, I, T_ = state[:, 0], state[:, 1], state[:, 2], state[:, 3]
    N = (S + L + I + T_).clamp(min=1e-8)
    foi = beta * I / N
    if extra_foi is not None:
        foi = (foi + extra_foi).clamp(min=0)
    dS  = MU * N - foi * S - MU * S + GAMMA * T_
    dL  = foi * S - (v + MU) * L
    dI  = v * L + DELTA * T_ - (tau + D_TB + MU) * I
    dT_ = tau * I - (GAMMA + DELTA + MU) * T_
    return torch.stack([dS, dL, dI, dT_], dim=1)


def rk4_step(func, t, dt, y):
    k1 = func(t, y)
    k2 = func(t + dt / 2, y + dt / 2 * k1)
    k3 = func(t + dt / 2, y + dt / 2 * k2)
    k4 = func(t + dt, y + dt * k3)
    return y + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)


def integrate(func, t_points, x0, dt=0.1):
    targets = t_points.tolist()
    traj, state, t_cur = [], x0, 0.0
    for t_next in targets:
        while t_cur < t_next - 1e-10:
            step = min(dt, t_next - t_cur)
            state = rk4_step(func, t_cur, step, state)
            t_cur += step
        traj.append(state)
    return torch.stack(traj, dim=0)


class NeuralResidual(nn.Module):
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
        S, L, I, T_ = state[:, 0], state[:, 1], state[:, 2], state[:, 3]
        N = (S + L + I + T_).clamp(min=1e-8)
        ca, sa = float(np.cos(2 * np.pi * t_scalar)), float(np.sin(2 * np.pi * t_scalar))
        cl, sl = float(np.cos(2 * np.pi * t_scalar / 7.0)), float(np.sin(2 * np.pi * t_scalar / 7.0))
        feat = torch.stack([S/N, L/N, I/N,
                             torch.full_like(S, ca), torch.full_like(S, sa),
                             torch.full_like(S, cl), torch.full_like(S, sl)], dim=1).float()
        return self.net(feat).squeeze(-1).double()


class PGNODEFunc(nn.Module):
    """See pgnode_drc_tb.py:PGNODEFunc for the full derivation of
    R0_full_at()/reff_at(), which include the relapse + treatment-
    recovery-to-S correction and the susceptible-fraction S(t)/N(t)
    factor respectively."""
    def __init__(self, beta_init, v_init, tau_init, hidden=32):
        super().__init__()
        self.log_beta = nn.Parameter(torch.tensor(np.log(beta_init), dtype=torch.float64))
        self.log_v    = nn.Parameter(torch.tensor(np.log(v_init),    dtype=torch.float64))
        self.log_tau  = nn.Parameter(torch.tensor(np.log(tau_init),  dtype=torch.float64))
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
        cprime = GAMMA + DELTA + MU
        denom = (self.v + MU) * ((self.tau + D_TB + MU) * cprime - DELTA * self.tau)
        return (beta_eff * self.v * cprime) / denom

    def reff_at(self, beta_eff, s_over_n):
        return s_over_n * self.R0_full_at(beta_eff)


class ClassicalSLIT(nn.Module):
    def __init__(self, beta_init, v_init, tau_init):
        super().__init__()
        self.log_beta = nn.Parameter(torch.tensor(np.log(beta_init), dtype=torch.float64))
        self.log_v    = nn.Parameter(torch.tensor(np.log(v_init),    dtype=torch.float64))
        self.log_tau  = nn.Parameter(torch.tensor(np.log(tau_init),  dtype=torch.float64))

    @property
    def beta(self): return self.log_beta.exp()
    @property
    def v(self):    return self.log_v.exp()
    @property
    def tau(self):  return self.log_tau.exp()

    def rhs(self, t, state):
        return slit_rhs(t, state, self.beta, self.v, self.tau)

    def R0(self):
        cprime = GAMMA + DELTA + MU
        denom = (self.v + MU) * ((self.tau + D_TB + MU) * cprime - DELTA * self.tau)
        return (self.beta * self.v * cprime) / denom


def obs(sol, tau):
    return tau * sol[:, 0, 2]


def loss_fn(model, t_pts, x0, y_obs, lam1=0.3, lam2=5e-4):
    sol = integrate(model.rhs, t_pts, x0, dt=0.2)
    y_hat = obs(sol, model.tau)
    scale = y_obs.mean().clamp(min=1.0)
    data = torch.mean(((y_hat - y_obs) / scale) ** 2)
    phys = torch.mean(torch.relu(-sol[:, 0, :].min(dim=-1).values) ** 2)
    l2 = sum(p.double().pow(2).sum() for n, p in model.named_parameters() if "nn_res" in n)
    return data + lam1 * phys + lam2 * l2, data.item()

# ================================================================
# 6.  TRAINING (identical hyperparameters/epochs to pgnode_drc_tb.py;
#     no country-specific retuning)
# ================================================================
N_SLIT, N_PG = 800, 1500
LR = 5e-3
BETA_INIT, V_INIT, TAU_INIT = 3.0, 0.01, 0.5

print("\nTraining Classical SLIT …")
slit = ClassicalSLIT(BETA_INIT, V_INIT, TAU_INIT)
opt_s = torch.optim.Adam(slit.parameters(), lr=LR)
sch_s = torch.optim.lr_scheduler.CosineAnnealingLR(opt_s, N_SLIT)
hist_s = []
for ep in range(N_SLIT):
    opt_s.zero_grad()
    l, dl = loss_fn(slit, T_TRAIN, X0, Y_TRAIN, lam2=0.0)
    l.backward()
    torch.nn.utils.clip_grad_norm_(slit.parameters(), 5.0)
    opt_s.step(); sch_s.step(); hist_s.append(dl)

print("Training PG-NODE …")
pgnd = PGNODEFunc(BETA_INIT, V_INIT, TAU_INIT)
with torch.no_grad():
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

# ================================================================
# 7.  INFERENCE & METRICS
# ================================================================
with torch.no_grad():
    sol_s = integrate(slit.rhs, T_ALL, X0, dt=0.05)
    sol_p = integrate(pgnd.rhs, T_ALL, X0, dt=0.05)

y_s = obs(sol_s, slit.tau).detach().numpy()
y_p = obs(sol_p, pgnd.tau).detach().numpy()
y_o = Y_ALL.numpy()

rmse = lambda a, b: np.sqrt(np.mean((a - b) ** 2))
rs_tr, rp_tr = rmse(y_s[:N_TRAIN], y_o[:N_TRAIN]), rmse(y_p[:N_TRAIN], y_o[:N_TRAIN])
rs_va, rp_va = rmse(y_s[N_TRAIN:], y_o[N_TRAIN:]), rmse(y_p[N_TRAIN:], y_o[N_TRAIN:])
rmse_imp = (rs_va - rp_va) / rs_va * 100

print(f"\n{'Model':<18}{'Train RMSE(K)':<18}{'Val RMSE(K)':<14}{'Improvement'}")
print(f"{'Classical SLIT':<18}{rs_tr:<18.2f}{rs_va:<14.2f}{'—'}")
print(f"{'PG-NODE':<18}{rp_tr:<18.2f}{rp_va:<14.2f}{rmse_imp:+.1f}%")

R0_s = slit.R0().item()
Reff_p, db = [], []
with torch.no_grad():
    for ti, t_i in enumerate(T_ALL.tolist()):
        st_i = sol_p[ti]
        S, L, I, T_ = st_i[0]
        N_i = (S + L + I + T_).item()
        sn_i = S.item() / N_i
        dlam_i = pgnd.nn_res(t_i, st_i).item()
        I_i = I.item()
        dbeta_i = dlam_i * N_i / I_i if I_i > 1e-6 else 0.0
        beta_eff = pgnd.beta.item() + dbeta_i
        Reff_p.append(pgnd.reff_at(torch.tensor(beta_eff), sn_i).item())
        db.append(dbeta_i)

print(f"\nClassical SLIT  R0 (fixed) = {R0_s:.3f}")
print(f"PG-NODE  Reff(t) range = [{min(Reff_p):.3f}, {max(Reff_p):.3f}]")
print(f"\n{'Year':>6}  {'Δβ':>8}  {'β_eff':>8}  {'Reff':>8}")
for i, y in enumerate(YEARS):
    print(f"  {y:6d}  {db[i]:+8.4f}  {pgnd.beta.item()+db[i]:8.4f}  {Reff_p[i]:8.4f}")

print(f"\n  Validation RMSE change vs classical SLIT: {rmse_imp:+.1f}%  "
      f"({rs_va:.2f}K -> {rp_va:.2f}K, n_val={len(YEARS)-N_TRAIN})")

# ================================================================
# 8.  ARIMA BASELINE (fitted on the 6 annual training values)
# ================================================================
print("\nARIMA(1,1,1) baseline …")
from statsmodels.tsa.arima.model import ARIMA as SARIMA

y_ann_tr = y_all_K[:N_TRAIN]
y_ann_val = y_all_K[N_TRAIN:]
arima_fit = SARIMA(y_ann_tr, order=(1, 1, 1)).fit()
arima_fv = np.array(arima_fit.forecast(steps=len(y_ann_val)))
ra_va = rmse(arima_fv, y_ann_val)
print(f"  ARIMA val RMSE: {ra_va:.2f}K")

# ================================================================
# 9.  FIGURES (mirror pgnode_drc_tb.py fig2 / fig3)
# ================================================================
C = {"slit": "#1d4ed8", "pg": "#b91c1c"}
LW = 2.5


def savefig(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUTDIR, f"{name}.{ext}"), bbox_inches="tight", dpi=300)
    print(f"  -> {name}.pdf / .png")


fig2, axs2 = plt.subplots(1, 2, figsize=(12, 5))
ax = axs2[0]
ax.plot(YEARS, y_o, "ko", ms=7, zorder=6, label="Observed (WHO annual, K/yr)")
ax.plot(YEARS, y_s, "s--", color=C["slit"], lw=LW, ms=7,
        label=f"Classical SLIT (RMSE: {rs_tr:.2f}K train / {rs_va:.2f}K val)")
ax.plot(YEARS, y_p, "^-", color=C["pg"], lw=LW, ms=7,
        label=f"PG-NODE (RMSE: {rp_tr:.2f}K train / {rp_va:.2f}K val)")
ax.axvspan(2020.5, 2022.5, alpha=.08, color="gray", label="Validation period")
ax.set(xlabel="Year", ylabel="Notifications (K/yr)",
       title="(a) Mozambique: Notification Fit -- Classical SLIT vs PG-NODE")
ax.legend(loc="upper left", fontsize=7.5); ax.grid(alpha=.3)

ax = axs2[1]
w = 0.35
x = np.arange(len(YEARS))
ax.bar(x - w/2, y_s - y_o, w, color=C["slit"], alpha=.75, label="Classical SLIT")
ax.bar(x + w/2, y_p - y_o, w, color=C["pg"], alpha=.75, label="PG-NODE")
ax.axhline(0, color="k", lw=.8)
ax.axvspan(N_TRAIN - 0.6, len(YEARS) - 0.4, alpha=.08, color="gray", label="Validation")
ax.set(xlabel="Year", ylabel="Residual (pred - obs, K/yr)", title="(b) Residual Analysis")
ax.set_xticks(x); ax.set_xticklabels([str(y) for y in YEARS])
ax.legend(); ax.grid(alpha=.3, axis="y")
plt.tight_layout(); savefig(fig2, "figMOZ_notification_fit"); plt.close(fig2)

fig3, axs3 = plt.subplots(1, 2, figsize=(12, 5))
ax = axs3[0]
ax.axhline(R0_s, color=C["slit"], lw=LW, ls="--",
           label=f"Classical SLIT: $\\mathcal{{R}}_0 = {R0_s:.2f}$ (fixed)")
ax.plot(YEARS, Reff_p, "o-", color=C["pg"], lw=LW, ms=6,
        label="PG-NODE: $\\mathcal{R}_{\\mathrm{eff}}(t)$ (time-varying)")
ax.fill_between(YEARS, R0_s, Reff_p, alpha=.12, color=C["pg"])
ax.axhline(1.0, color="k", lw=.8, ls=":", alpha=.5)
ax.set(xlabel="Year", ylabel="$\\mathcal{R}_{\\mathrm{eff}}(t)$",
       title="(a) Mozambique: $\\mathcal{R}_{\\mathrm{eff}}$ Trajectory (includes $S(t)/N(t)$)")
ax.legend(fontsize=8); ax.grid(alpha=.3)

ax = axs3[1]
colors_bar = [C["pg"] if d > 0 else "#1a9850" for d in db]
ax.bar(YEARS, db, color=colors_bar, alpha=.75, label="Neural residual $\\Delta\\beta(t)$")
ax.axhline(0, color="k", lw=.8)
ax.set(xlabel="Year", ylabel="$\\Delta\\beta(t)$ (yr$^{-1}$)",
       title="(b) Mozambique: PG-NODE Neural Residual $\\Delta\\beta(t)$")
ax.legend(); ax.grid(alpha=.3, axis="y")
plt.tight_layout(); savefig(fig3, "figMOZ_r0_trajectory"); plt.close(fig3)

print(f"\n{'='*65}")
print("SUMMARY")
print(f"{'='*65}")
print(f"Classical SLIT: beta={slit.beta.item():.3f} v={slit.v.item():.5f} "
      f"tau={slit.tau.item():.3f} R0={R0_s:.3f}")
print(f"PG-NODE:        beta0={pgnd.beta.item():.3f} v={pgnd.v.item():.5f} "
      f"tau={pgnd.tau.item():.3f} Reff range=[{min(Reff_p):.3f},{max(Reff_p):.3f}]")
print(f"Val RMSE: SLIT={rs_va:.2f}K PG-NODE={rp_va:.2f}K ARIMA={ra_va:.2f}K "
      f"(PG-NODE vs SLIT: {rmse_imp:+.1f}%)")
print("\nAll figures saved to ./figures_moz/  (PDF + PNG)")
