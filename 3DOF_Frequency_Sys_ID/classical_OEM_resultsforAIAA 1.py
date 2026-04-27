from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import casadi as ca
from types import SimpleNamespace

from model import build_aircraft_model


# Choose the maneuver type for the synthetic truth experiment.
MANEUVER_TYPE = "near_trim"   # "near_trim" or "aggressive"
GUST_ON = False
ACTUATOR_MODEL_ON = False

# Set simulation time and sample period.
TF = 20.0
TS = 0.01

# Fix the random seed so results are repeatable.
SEED = 7
#This is only used for MAP objective function. The prior is not used in the truth simulation or the MLE OEM case, so it does not affect the other results.
THETA_PRIOR_MEAN = np.array([
    4.0,    # C_La   (positive lift slope)
    0.05,   # C_L0   (small offset)
    0.08,   # k_CLCD (positive)
    0.04,   # C_D0   (small positive)
    0.00,   # C_M0   (near trim)
    0.08,   # C_Me   (moderate control effectiveness)
   -0.15,   # C_MQ   (damping, stabilizing)
   -0.20,   # C_Ma   (static stability)
], dtype=float)

THETA_PRIOR_SCALE = np.array([
    1.5,    # C_La   (loose)
    0.08,   # C_L0
    0.06,   # k_CLCD
    0.02,   # C_D0
    0.08,   # C_M0   (very loose)
    0.10,   # C_Me
    0.12,   # C_MQ
    0.15,   # C_Ma
], dtype=float)

LAMBDA_MAP = 1.0

# Toggle which figures to save.
SHOW_INPUT_FIGURE = True
SHOW_PARAMETER_FIGURE = True
SHOW_RESIDUAL_FIGURE = True
SHOW_SIGMA_FIGURE = True


# Channel-wise weighting for OEM residuals.
# This defines the weighting matrix W = diag(1 / sigma_y^2) used in the cost function of classical OEM.
# In Classical OEM, these are fixed user-defined weights (not a true noise model).
# In MLE OEM, similar quantities are treated as measurement noise variances and are estimated from data.
SIGMA_Y = np.array([0.10, np.deg2rad(0.40), np.deg2rad(0.40), np.deg2rad(1.20)], dtype=float)

# Select the aerodynamic parameters estimated by the simplified OEM model.
AERO_NAMES = ["C_La", "C_L0", "k_CLCD", "C_D0", "C_M0", "C_Me", "C_MQ", "C_Ma"]

# Start the optimizer near with a reasonable guess. 
INITIAL_GUESS_SCALE = np.array([1.15, 0.85, 1.20, 1.10, 0.80, 1.15, 0.85, 1.10], dtype=float)

# Impose reasonable parameter bounds for the synthetic study.
LB_AERO = np.array([0.2, -0.5, 0.0, 0.001, -1.0, -1.0, -3.0, -3.0], dtype=float)
UB_AERO = np.array([8.0,  1.0, 2.5, 0.5,    1.0,  1.0,  1.0,  1.0], dtype=float)

# Define the gust forcing used in the truth model.
GUST_DRIVE_AMPLITUDE = 0.8
GUST_DRIVE_F1 = 0.08
GUST_DRIVE_F2 = 0.17

# Map state names to indices for readability.
IDX = {
    "V": 0, "alpha": 1, "gamma": 2, "Q": 3,
    "x_ig": 4, "x_iv": 5, "z1": 6, "z2": 7,
    "vwp": 8, "gwp": 9, "vref": 10, "gref": 11,
    "tau": 12, "q": 13,
    "T_act": 14, "elv_act": 15,
    "gust": 16,
}


# Store the parameter-vector indices once so the rest of the code stays readable.
@dataclass(frozen=True)
class PIdx:
    m: int
    Jy: int
    g: int
    rho: int
    S: int
    C_La: int
    C_L0: int
    k_CLCD: int
    C_D0: int
    C_M0: int
    C_Me: int
    C_MQ: int
    C_Ma: int
    Kpg: int
    Kig: int
    Kdg: int
    Kpv: int
    Kiv: int
    zeta: int
    omega: int
    v0: int
    g0: int
    Av: int
    Ag: int
    mu_q: int
    Tv_ref: int
    Tg_ref: int
    Tmin: int
    Tmax: int
    dTmax: int
    emin: int
    emax: int
    demax: int
    tau_T: int
    tau_elv: int
    tau_gust: int


# Convert symbolic parameter names into fixed integer indices.
def make_param_index(p_syms) -> PIdx:
    name_to_i = {str(s): i for i, s in enumerate(p_syms)}
    return PIdx(
        m=name_to_i["m"],
        Jy=name_to_i["J_y"],
        g=name_to_i["g"],
        rho=name_to_i["rho"],
        S=name_to_i["S"],
        C_La=name_to_i["C_La"],
        C_L0=name_to_i["C_L0"],
        k_CLCD=name_to_i["k_CLCD"],
        C_D0=name_to_i["C_D0"],
        C_M0=name_to_i["C_M0"],
        C_Me=name_to_i["C_Me"],
        C_MQ=name_to_i["C_MQ"],
        C_Ma=name_to_i["C_Ma"],
        Kpg=name_to_i["Kpg"],
        Kig=name_to_i["Kig"],
        Kdg=name_to_i["Kdg"],
        Kpv=name_to_i["Kpv"],
        Kiv=name_to_i["Kiv"],
        zeta=name_to_i["zeta"],
        omega=name_to_i["omega"],
        v0=name_to_i["v0"],
        g0=name_to_i["g0"],
        Av=name_to_i["Av"],
        Ag=name_to_i["Ag"],
        mu_q=name_to_i["mu_q"],
        Tv_ref=name_to_i["Tv_ref"],
        Tg_ref=name_to_i["Tg_ref"],
        Tmin=name_to_i["Tmin"],
        Tmax=name_to_i["Tmax"],
        dTmax=name_to_i["dTmax"],
        emin=name_to_i["emin"],
        emax=name_to_i["emax"],
        demax=name_to_i["demax"],
        tau_T=name_to_i["tau_T"],
        tau_elv=name_to_i["tau_elv"],
        tau_gust=name_to_i["tau_gust"],
    )


# Enforce actuator magnitude and rate limits inside the truth simulation.
def sat_rate_limit(x: np.ndarray, dx: np.ndarray, p0: np.ndarray, pidx: PIdx) -> np.ndarray:
    Tmin = float(p0[pidx.Tmin])
    Tmax = float(p0[pidx.Tmax])
    dTmax = float(p0[pidx.dTmax])
    emin = float(p0[pidx.emin])
    emax = float(p0[pidx.emax])
    demax = float(p0[pidx.demax])

    T = float(np.clip(x[IDX["T_act"]], Tmin, Tmax))
    elv = float(np.clip(x[IDX["elv_act"]], emin, emax))
    dT = float(np.clip(dx[IDX["T_act"]], -dTmax, dTmax))
    delv = float(np.clip(dx[IDX["elv_act"]], -demax, demax))

    if T >= Tmax and dT > 0.0:
        dT = 0.0
    if T <= Tmin and dT < 0.0:
        dT = 0.0
    if elv >= emax and delv > 0.0:
        delv = 0.0
    if elv <= emin and delv < 0.0:
        delv = 0.0

    dx = dx.copy()
    dx[IDX["T_act"]] = dT
    dx[IDX["elv_act"]] = delv
    return dx


# Generate a smooth truth-side gust forcing signal.
def gust_drive(t: float, amp: float) -> float:
    return amp * (
        0.6 * np.sin(2.0 * np.pi * GUST_DRIVE_F1 * t)
        + 0.4 * np.sin(2.0 * np.pi * GUST_DRIVE_F2 * t + 0.7)
    )


# Reconstruct commanded inputs from the truth state history.
def compute_command_history(x: np.ndarray, p0: np.ndarray, pidx: PIdx) -> np.ndarray:
    Kpg = float(p0[pidx.Kpg])
    Kig = float(p0[pidx.Kig])
    Kdg = float(p0[pidx.Kdg])
    Kpv = float(p0[pidx.Kpv])
    Kiv = float(p0[pidx.Kiv])

    e_g = x[:, IDX["z1"]] - x[:, IDX["gamma"]]
    e_v = x[:, IDX["vref"]] - x[:, IDX["V"]]
    T_cmd = Kpv * e_v + Kiv * x[:, IDX["x_iv"]]
    elv_cmd = Kpg * e_g + Kig * x[:, IDX["x_ig"]] + Kdg * x[:, IDX["Q"]]
    return np.column_stack([T_cmd, elv_cmd])


# Simulate one realistic truth trajectory with controller, actuator, and optional gust.
def simulate_truth(
    f_eval,
    p0: np.ndarray,
    pidx: PIdx,
    *,
    Tsim: float,
    dt: float,
    seed: int,
    maneuvertype: str,
    gust_on: bool,
    actuator_model_on: bool = True,
    rtol: float = 1e-8,
    atol: float = 1e-10,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    p0 = np.asarray(p0, dtype=float).copy()

    v0 = float(p0[pidx.v0])
    Av = float(p0[pidx.Av])
    g0 = float(p0[pidx.g0])
    Ag = float(p0[pidx.Ag])
    mu_q = float(p0[pidx.mu_q])

    if maneuvertype == "aggressive":
        dwell_min, dwell_max = 0.6, 3.0
        levels_v = np.linspace(-1.0, 1.0, 5)
        levels_g = np.linspace(-1.0, 1.0, 5)
    elif maneuvertype == "near_trim":
        dwell_min, dwell_max = 3.0, 10.0
        Av = 0.15
        Ag = 0.01
        levels_v = np.linspace(-1.0, 1.0, 9)
        levels_g = np.linspace(-1.0, 1.0, 9)
    else:
        raise ValueError("MANEUVER_TYPE must be 'near_trim' or 'aggressive'.")

    def next_waypoint(x: np.ndarray) -> Tuple[float, float, float, float]:
        q = float(np.clip(x[IDX["q"]], 1e-6, 1.0 - 1e-6))
        q1 = mu_q * q * (1.0 - q)
        q2 = mu_q * q1 * (1.0 - q1)
        q3 = mu_q * q2 * (1.0 - q2)

        iv = int(np.floor(q1 * len(levels_v))) % len(levels_v)
        ig = int(np.floor(q2 * len(levels_g))) % len(levels_g)

        vwp = v0 + Av * float(levels_v[iv])
        gwp = g0 + Ag * float(levels_g[ig])
        dwell = float(np.clip(dwell_min + (dwell_max - dwell_min) * q3, 0.2, dwell_max))
        return float(q1), float(vwp), float(gwp), dwell

    x0 = np.array([
        6.5, 0.05, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
        6.5, 0.0, 6.5, 0.0,
        0.0, 0.23,
        0.0, 0.0,
        0.0,
    ], dtype=float)

    if maneuvertype == "near_trim":
        sV, sa, sg, sQ = 0.08, 0.005, 0.002, 0.01
    else:
        sV, sa, sg, sQ = 0.3, 0.02, 0.01, 0.05

    x0[IDX["V"]] += sV * rng.standard_normal()
    x0[IDX["alpha"]] += sa * rng.standard_normal()
    x0[IDX["gamma"]] += sg * rng.standard_normal()
    x0[IDX["Q"]] += sQ * rng.standard_normal()
    x0[IDX["q"]] = rng.uniform(0.05, 0.95)
    x0[IDX["gust"]] = 0.0

    q1, vwp, gwp, dwell = next_waypoint(x0)
    x0[IDX["q"]] = q1
    x0[IDX["vwp"]] = vwp
    x0[IDX["gwp"]] = gwp
    x0[IDX["vref"]] = vwp
    x0[IDX["gref"]] = gwp
    x0[IDX["tau"]] = 0.0

    Tmid = 0.5 * (float(p0[pidx.Tmin]) + float(p0[pidx.Tmax]))
    x0[IDX["T_act"]] = float(Tmid + 0.02 * rng.standard_normal())
    x0[IDX["elv_act"]] = float(0.0 + 0.01 * rng.standard_normal())

    dwell_box = [dwell]

    def event_waypoint(t: float, x: np.ndarray) -> float:
        return float(x[IDX["tau"]] - dwell_box[0])

    event_waypoint.terminal = True
    event_waypoint.direction = 1.0

    def rhs(t: float, x: np.ndarray) -> np.ndarray:
        xx = x.copy()

        if actuator_model_on:
            xx[IDX["T_act"]]   = np.clip(xx[IDX["T_act"]],   p0[pidx.Tmin], p0[pidx.Tmax])
            xx[IDX["elv_act"]] = np.clip(xx[IDX["elv_act"]], p0[pidx.emin], p0[pidx.emax])
            dx = np.array(f_eval(list(xx), list(p0)), dtype=float).reshape(-1)
            dx = sat_rate_limit(xx, dx, p0, pidx)
        else:
            # Inject commanded inputs directly so f_eval sees the correct aerodynamic forcing.
            u_now = compute_command_history(xx[np.newaxis, :], p0, pidx)
            xx[IDX["T_act"]]   = float(np.clip(u_now[0, 0], p0[pidx.Tmin], p0[pidx.Tmax]))
            xx[IDX["elv_act"]] = float(np.clip(u_now[0, 1], p0[pidx.emin], p0[pidx.emax]))
            dx = np.array(f_eval(list(xx), list(p0)), dtype=float).reshape(-1)
            # Freeze actuator ODE states — u_act will be set to u_cmd on return.
            dx[IDX["T_act"]]   = 0.0
            dx[IDX["elv_act"]] = 0.0

        if gust_on:
            tau_g = float(p0[pidx.tau_gust])
            dx[IDX["gust"]] = (-xx[IDX["gust"]] / tau_g
                            + gust_drive(t, GUST_DRIVE_AMPLITUDE))
        else:
            # Only zero the derivative — do NOT zero xx here, f_eval already ran.
            dx[IDX["gust"]] = 0.0

        return dx

    t_all = []
    x_all = []
    t0 = 0.0
    x = x0.copy()

    while t0 < Tsim - 1e-12:
        tf = Tsim
        n = int(np.floor((tf - t0) / dt))
        t_eval = t0 + dt * np.arange(n + 1)
        if t_eval[-1] < tf:
            t_eval = np.append(t_eval, tf)

        sol = solve_ivp(
            rhs,
            (t0, tf),
            x,
            t_eval=t_eval,
            method="RK45",
            events=event_waypoint,
            rtol=rtol,
            atol=atol,
        )

        if len(t_all) == 0:
            t_all.append(sol.t)
            x_all.append(sol.y.T)
        else:
            t_all.append(sol.t[1:])
            x_all.append(sol.y.T[1:, :])

        if sol.t_events[0].size == 0:
            break

        x = sol.y[:, -1].copy()
        q1, vwp, gwp, dwell_next = next_waypoint(x)
        x[IDX["q"]] = np.clip(q1, 1e-6, 1.0 - 1e-6)
        x[IDX["vwp"]] = vwp
        x[IDX["gwp"]] = gwp
        x[IDX["tau"]] = 0.0
        dwell_box[0] = max(float(dwell_next), 0.2)
        t0 = float(sol.t_events[0][-1])

    t = np.concatenate(t_all)
    x = np.vstack(x_all)

    u_cmd = compute_command_history(x, p0, pidx)

    if actuator_model_on:
        # Actuator states in x correctly reflect the lagged applied inputs.
        u_act = np.column_stack([x[:, IDX["T_act"]], x[:, IDX["elv_act"]]])
    else:
        # No actuator lag — applied inputs equal commanded inputs exactly.
        u_act = u_cmd.copy()

    return t, x, u_act, u_cmd


def main() -> None:
    f_eval, p_syms, p0_full, _, _ = build_aircraft_model()
    pidx = make_param_index(p_syms)

    # Slow the actuators slightly so the commanded/applied mismatch is visible.
    p0_full = np.array(p0_full, dtype=float)
    p0_full[pidx.tau_elv] = 0.35
    p0_full[pidx.tau_T] = 0.60

    theta_true = np.array([float(p0_full[getattr(pidx, n)]) for n in AERO_NAMES], dtype=float)
    theta0 = np.clip(theta_true * INITIAL_GUESS_SCALE, LB_AERO, UB_AERO)
    consts = {
        "m": float(p0_full[pidx.m]),
        "Jy": float(p0_full[pidx.Jy]),
        "g": float(p0_full[pidx.g]),
        "rho": float(p0_full[pidx.rho]),
        "S": float(p0_full[pidx.S]),
    }

    # One realistic ground-truth trajectory shared by all six estimation cases.
    t, x_truth, u_act, u_cmd = simulate_truth(
        f_eval, p0_full, pidx,
        Tsim=TF, dt=TS, seed=SEED,
        maneuvertype=MANEUVER_TYPE,
        gust_on=GUST_ON, actuator_model_on=ACTUATOR_MODEL_ON,
    )
    y_truth = x_truth[:, :4].copy()
    # x0_oem = y_truth[0, :].copy()
    gust_truth = x_truth[:, IDX["gust"]]

    # Extract states from y_truth
    V_out     = y_truth[:, 0]
    alpha_out = y_truth[:, 1]
    gamma_out = y_truth[:, 2]
    Q_out     = y_truth[:, 3]

    # Stack all data into one array
    data = np.column_stack((
        t,
        V_out,
        alpha_out,
        gamma_out,
        Q_out,
        u_cmd[:, 0],   # commanded thrust
        u_cmd[:, 1],   # commanded elevator
        u_act[:, 0],   # actual thrust
        u_act[:, 1],   # actual elevator
        gust_truth
    ))

    # Create output file path (same folder as script)
    out_dir = "/Users/abhi/Desktop/Projects/Sys_ID/PINN-tutorial/3DOF_Frequency_Sys_ID"
    
    # Build a descriptive suffix from the global flags.
    gust_tag     = "gust"     if GUST_ON             else "nogust"
    actuator_tag = "actuator" if ACTUATOR_MODEL_ON   else "noactuator"
    sim_tag = f"{MANEUVER_TYPE}_{gust_tag}_{actuator_tag}"

    data_file  = os.path.join(out_dir, f"simulation_data_{sim_tag}.csv")
    param_file = os.path.join(out_dir, f"truth_model_params_{sim_tag}.csv")


    # Save to CSV with header
    header = (
        "t, V, alpha, gamma, Q, "
        "T_cmd, elv_cmd, T_act, elv_act, gust"
    )

    np.savetxt(data_file, data, delimiter=",", header=header, comments="")

    print(f"Saved simulation data to: {data_file}")


    # -------------------------------
    # Save truth model parameters CSV
    # -------------------------------

    param_names = [
        "g", "rho", "S", "cbar",
        "m", "Jy",
        "C_L0", "C_La",
        "C_D0", "k_CLCD",
        "C_M0", "C_Ma", "C_MQ", "C_Me",
        "tau_T", "tau_elv", "tau_gust"
    ]

    param_values = [
        float(p0_full[pidx.g]),
        float(p0_full[pidx.rho]),
        float(p0_full[pidx.S]),
        1.0,  # cbar (not in model, constant)

        float(p0_full[pidx.m]),
        float(p0_full[pidx.Jy]),

        float(p0_full[pidx.C_L0]),
        float(p0_full[pidx.C_La]),

        float(p0_full[pidx.C_D0]),
        float(p0_full[pidx.k_CLCD]),

        float(p0_full[pidx.C_M0]),
        float(p0_full[pidx.C_Ma]),
        float(p0_full[pidx.C_MQ]),
        float(p0_full[pidx.C_Me]),

        float(p0_full[pidx.tau_T]),
        float(p0_full[pidx.tau_elv]),
        float(p0_full[pidx.tau_gust]),
    ]

    # Combine into 2-column array: name, value
    param_array = np.column_stack((param_names, param_values))

    # Save CSV
    np.savetxt(param_file, param_array, fmt="%s", delimiter=",", header="parameter,value", comments="")

    print(f"Saved truth model parameters to: {param_file}")


# Execute the script directly when run as a file.
if __name__ == "__main__":
    main()
