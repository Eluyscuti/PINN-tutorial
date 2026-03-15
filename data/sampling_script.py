"""
sampling_script.py
====================
Generates Mass-Spring-Damper data and saves to a .pkl file.
All core functionality is identical to the original sampling script.

Usage examples
--------------
# defaults (replicates original script exactly)
python sampling_script.py

# custom physical parameters
python sampling_script.py --m 2.0 --c 0.3 --k 0.2

# custom sparsity levels
python sampling_script.py --sparsity 50 100 200

# custom noise
python sampling_script.py --noise_sd 0.05

# specific forcing models only
python sampling_script.py --forcing none sinusoidal

# save plots (directory auto-named from parameters)
python sampling_script.py --plots

# custom output path
python sampling_script.py --output /path/to/my_data.pkl

# full custom example
python sampling_script.py --m 1.5 --c 0.2 --k 0.3 --sparsity 50 100 --noise_sd 0.05 --forcing none sinusoidal --plots --output custom_data.pkl
# → saves plots to: plots_m1.5_c0.2_k0.3_sd0.05_s50-100_fnone-sinusoidal_seed42/
"""

import argparse
import os
import numpy as np
import scipy.signal
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle


# ---------------------------
# CLI argument parsing
# ---------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate Mass-Spring-Damper simulation data."
    )

    # Physical parameters
    parser.add_argument('--m',  type=float, default=2.0,
                        help='Mass [kg] (default: 2.0)')
    parser.add_argument('--c',  type=float, default=0.3,
                        help='Damping coefficient [Ns/m] (default: 0.3)')
    parser.add_argument('--k',  type=float, default=0.2,
                        help='Spring constant [N/m] (default: 0.2)')
    parser.add_argument('--Kp', type=float, default=3.0,
                        help='Proportional gain for reference tracking (default: 3.0)')

    # Initial conditions
    parser.add_argument('--x0', type=float, nargs=2, default=[1.0, 0.0],
                        metavar=('POSITION', 'VELOCITY'),
                        help='Initial conditions [position velocity] (default: 1.0 0.0)')

    # Time span
    parser.add_argument('--t_end',   type=float, default=60.0,
                        help='End time [s] (default: 60.0)')
    parser.add_argument('--n_eval',  type=int,   default=600,
                        help='Number of time points for ground truth (default: 600)')

    # Sparsity
    parser.add_argument('--sparsity', type=int, nargs='+', default=[50, 100, 150],
                        help='Sparsity levels (default: 50 100 150)')

    # Noise
    parser.add_argument('--noise_sd',   type=float, default=0.1,
                        help='Gaussian noise std dev (default: 0.1)')
    parser.add_argument('--noise_mean', type=float, default=0.0,
                        help='Gaussian noise mean (default: 0.0)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')

    # Forcing models to include
    parser.add_argument('--forcing', type=str, nargs='+',
                        choices=['none', 'sinusoidal', 'square', 'reference'],
                        default=['none', 'sinusoidal', 'square', 'reference'],
                        help='Forcing models to include (default: all four)')

    # Output
    parser.add_argument('--output', type=str, default='massdamper_data.pkl',
                        help='Output .pkl file path (default: massdamper_data.pkl)')
    parser.add_argument('--plots', action='store_true',
                        help='Generate and save plots (off by default). '
                             'Directory is auto-named from call parameters.')

    return parser.parse_args()


# ---------------------------
# ODE functions
# ---------------------------
def oscillator(t, y, m_true, c_true, k_true, u_t, w_t, Kp):
    x, v = y
    dxdt = v
    if Kp is None:
        u = u_t(t) if callable(u_t) else u_t
    else:
        u = u_t(t, x) if callable(u_t) else u_t
    w = w_t(t) if callable(w_t) else w_t
    dvdt = (-c_true * v - k_true * x) / m_true + (u + w) / m_true
    return [dxdt, dvdt]


def reference_signal(t):
    r_t = np.zeros_like(t)
    r_t[(t < 20.0)] = 0.2
    r_t[(t > 20.0) & (t < 35.0)] = -0.2
    r_t[(t < 45.0) & (t > 35.0)] = 0.2
    r_t[(t < 55.0) & (t > 45.0)] = 0.4
    r_t[t >= 55.0] = 0.0
    return r_t


# ---------------------------
# Main
# ---------------------------
def main():
    args = parse_args()
    np.random.seed(args.seed)

    m_true = args.m
    c_true = args.c
    k_true = args.k
    Kp     = args.Kp
    x0     = args.x0
    t_span = (0, args.t_end)
    t_eval = np.linspace(*t_span, args.n_eval)

    # derived parameters
    delta_val  = c_true / (2 * m_true)
    omega0_val = np.sqrt(k_true / m_true)
    T_val      = 1 / omega0_val

    print(f"Physical parameters: m={m_true}, c={c_true}, k={k_true}")
    print(f"Derived: delta={delta_val:.4f}, omega0={omega0_val:.4f}")
    print(f"Forcing models: {args.forcing}")
    print(f"Sparsity levels: {args.sparsity}")
    print(f"Noise: mean={args.noise_mean}, sd={args.noise_sd}")

    # ---------------------------
    # Define forcing terms
    # ---------------------------
    forcing_map = {
        'none':      (0,                                              None,  'No external force'),
        'sinusoidal':(lambda t: np.sin(2 * np.pi * t),               None,  'sinusoidal force'),
        'square':    (lambda t: scipy.signal.square(2*np.pi*0.5*t),  None,  'square wave force'),
        'reference': (lambda t, x: Kp * (reference_signal(t) - x),  Kp,    'reference tracking with P controller'),
    }

    w_1 = 0
    selected         = [forcing_map[f] for f in args.forcing]
    u_funcs          = [s[0] for s in selected]
    Kp_array         = [s[1] for s in selected]
    u_legend         = [s[2] for s in selected]

    # ---------------------------
    # Ground truth simulation
    # ---------------------------
    sol_inharm = []
    for i in range(len(u_funcs)):
        sol = solve_ivp(oscillator, t_span, x0, t_eval=t_eval,
                        args=(m_true, c_true, k_true, u_funcs[i], w_1, Kp_array[i]))
        sol_inharm.append(sol)

    print(f"Simulated {len(sol_inharm)} forcing models over t={t_span}, {args.n_eval} timepoints.")

    # ---------------------------
    # Ground truth dictionary
    # ---------------------------
    data_groundtruth = {
        'ground_truth_params': {
            'x0': x0, 't_span': t_span, 't_eval': t_eval,
            'm_true': m_true, 'c_true': c_true, 'k_true': k_true,
            'Kp_array': Kp_array, 'delta_val': delta_val,
            'omega0_val': omega0_val, 'T_val': T_val
        }
    }
    for i in range(len(u_funcs)):
        data_groundtruth[f'model: {u_legend[i]}'] = {
            't': sol_inharm[i].t,
            'x': sol_inharm[i].y[0],
            'v': sol_inharm[i].y[1],
        }

    # ---------------------------
    # Evenly spaced samples (no noise)
    # ---------------------------
    data_sampled = {}
    for n_idx in range(len(u_funcs)):
        model_name = u_legend[n_idx]
        data_sampled[model_name] = {}
        t_samples = [np.linspace(0, len(sol_inharm[n_idx].t) - 1, m).astype(int)
                     for m in args.sparsity]
        for m_idx, m in enumerate(args.sparsity):
            subsamples = t_samples[m_idx]
            data_sampled[model_name][f'sparcity level {m}'] = {
                't': sol_inharm[n_idx].t[subsamples],
                'x': sol_inharm[n_idx].y[0][subsamples],
                'v': sol_inharm[n_idx].y[1][subsamples],
            }

    # ---------------------------
    # Random samples (no noise)
    # ---------------------------
    data_sampled_random = {}
    for n_idx in range(len(u_funcs)):
        model_name = u_legend[n_idx]
        data_sampled_random[model_name] = {}
        t_full = sol_inharm[n_idx].t
        x_full = sol_inharm[n_idx].y[0]
        v_full = sol_inharm[n_idx].y[1]
        for m in args.sparsity:
            indices = np.random.choice(len(t_full), size=m, replace=False)
            indices = np.sort(indices)
            data_sampled_random[model_name][str(m)] = {
                't': t_full[indices],
                'x': x_full[indices],
                'v': v_full[indices],
            }

    # ---------------------------
    # Evenly spaced samples with noise
    # ---------------------------
    data_sampled_random_with_noise = {}
    for n_idx in range(len(u_funcs)):
        model_name = u_legend[n_idx]
        data_sampled_random_with_noise[model_name] = {}
        for m in args.sparsity:
            x_clean = data_sampled[model_name][f'sparcity level {m}']['x']
            v_clean = data_sampled[model_name][f'sparcity level {m}']['v']
            x_noise = x_clean + np.random.normal(args.noise_mean, args.noise_sd, size=x_clean.shape)
            v_noise = x_clean + np.random.normal(args.noise_mean, args.noise_sd, size=v_clean.shape)
            data_sampled_random_with_noise[model_name][f'sparcity level {m}'] = {
                't': data_sampled[model_name][f'sparcity level {m}']['t'],
                'x': x_noise,
                'v': v_noise,
            }

    # ---------------------------
    # Collocation points
    # ---------------------------
    data_colloc = {
        f"Collocation points {n}": np.linspace(0, t_span[1], n)
        for n in args.sparsity
    }

    # ---------------------------
    # Plots
    # ---------------------------
    if args.plots:
        # auto-name directory from call parameters
        sparsity_str  = '-'.join(str(s) for s in args.sparsity)
        forcing_str   = '-'.join(args.forcing)
        plot_dir      = (
            f"plots_m{m_true}_c{c_true}_k{k_true}"
            f"_sd{args.noise_sd}"
            f"_s{sparsity_str}"
            f"_f{forcing_str}"
            f"_seed{args.seed}"
        )
        os.makedirs(plot_dir, exist_ok=True)
        print(f"Saving plots to {plot_dir}/")

        def savefig(name):
            path = os.path.join(plot_dir, name)
            plt.savefig(path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  Saved: {path}")

        # 1. Ground truth trajectories — all forcing models
        fig, ax = plt.subplots(figsize=(10, 4))
        for i in range(len(u_funcs)):
            ax.plot(sol_inharm[i].t, sol_inharm[i].y[0], label=f"x(t) – {u_legend[i]}")
        if 'reference' in args.forcing:
            ref_idx = args.forcing.index('reference')
            ax.plot(sol_inharm[ref_idx].t,
                    [reference_signal(t) for t in sol_inharm[ref_idx].t],
                    '--', label='reference r(t)')
        ax.set_xlabel("time t [s]")
        ax.set_ylabel("Displacement x(t) [m]")
        ax.set_title("Ground truth — all forcing models")
        ax.legend(); ax.grid(True); fig.tight_layout()
        savefig("ground_truth.png")

        # 2. Evenly spaced samples — one plot per sparsity level
        for s in args.sparsity:
            fig, ax = plt.subplots(figsize=(10, 4))
            for model_name in u_legend:
                d = data_sampled[model_name][f'sparcity level {s}']
                ax.scatter(d['t'], d['x'], label=model_name, s=4, lw=0)
            ax.set_title(f"Evenly spaced samples — sparsity {s}")
            ax.set_xlabel("t"); ax.set_ylabel("x(t)")
            ax.legend(); ax.grid(True); fig.tight_layout()
            savefig(f"sampled_even_s{s}.png")

        # 3. Random samples — one plot per sparsity level
        for s in args.sparsity:
            fig, ax = plt.subplots(figsize=(10, 4))
            for model_name in u_legend:
                d = data_sampled_random[model_name][str(s)]
                ax.scatter(d['t'], d['x'], label=model_name, s=4)
            ax.set_title(f"Random samples — {s} points")
            ax.set_xlabel("t"); ax.set_ylabel("x(t)")
            ax.legend(); ax.grid(True); fig.tight_layout()
            savefig(f"sampled_random_s{s}.png")

        # 4. Noisy samples — one plot per sparsity level
        for s in args.sparsity:
            fig, ax = plt.subplots(figsize=(10, 4))
            for model_name in u_legend:
                d = data_sampled_random_with_noise[model_name][f'sparcity level {s}']
                ax.scatter(d['t'], d['x'], label=model_name, s=4, lw=0)
            ax.set_title(
                f"Noisy samples (mean={args.noise_mean}, sd={args.noise_sd}) — sparsity {s}"
            )
            ax.set_xlabel("t"); ax.set_ylabel("x(t)")
            ax.legend(); ax.grid(True); fig.tight_layout()
            savefig(f"sampled_noisy_s{s}.png")

        # 5. Per-model overlay: clean vs noisy for each sparsity level
        for model_name in u_legend:
            fig, axes = plt.subplots(1, len(args.sparsity),
                                     figsize=(5 * len(args.sparsity), 4), sharey=True)
            if len(args.sparsity) == 1:
                axes = [axes]
            for ax, s in zip(axes, args.sparsity):
                clean = data_sampled[model_name][f'sparcity level {s}']
                noisy = data_sampled_random_with_noise[model_name][f'sparcity level {s}']
                ax.plot(clean['t'], clean['x'], lw=1.5, label='clean')
                ax.scatter(noisy['t'], noisy['x'], s=6, color='tomato', label='noisy', zorder=3)
                ax.set_title(f"sparsity {s}")
                ax.set_xlabel("t"); ax.grid(True)
                ax.legend(fontsize=8)
            axes[0].set_ylabel("x(t)")
            safe_name = model_name.replace(' ', '_').replace('/', '_')
            fig.suptitle(f"{model_name} — clean vs noisy")
            fig.tight_layout()
            savefig(f"clean_vs_noisy_{safe_name}.png")

        print(f"All plots saved to {plot_dir}/")

    # ---------------------------
    # Save
    # ---------------------------
    with open(args.output, 'wb') as f:
        pickle.dump({
            'data_groundtruth':               data_groundtruth,
            'data_sampled':                   data_sampled,
            'data_sampled_random':            data_sampled_random,
            'data_sampled_random_with_noise': data_sampled_random_with_noise,
            'data_colloc':                    data_colloc,
            'args':                           vars(args),   # store run config
        }, f)

    print(f"Saved to {args.output}")


if __name__ == '__main__':
    main()