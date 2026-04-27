import numpy as np
import sympy as sp
import sympy.physics.mechanics as me


def build_aircraft_model():
    m, J_y, g, t = sp.symbols('m, J_y, g, t')
    gamma, alpha, V, Q, x_ig, x_iv, z_1, z_2 = me.dynamicsymbols(
        'gamma, alpha, V, Q, x_ig, x_iv, z_1, z_2'
    )
    vwp, gwp, vref, gref, tau, q = me.dynamicsymbols('vwp, gwp, vref, gref, tau, q')

    frame_e = me.ReferenceFrame('e')
    frame_w = frame_e.orientnew('w', 'Axis', (gamma, frame_e.y))
    frame_b = frame_w.orientnew('b', 'Axis', (-alpha, frame_w.y))
    W = m * g

    C_La, C_L0, k_CLCD, C_D0, C_M0, C_Me, C_MQ, C_Ma, rho, S = sp.symbols(
        'C_La, C_L0, k_CLCD, C_D0, C_M0, C_Me, C_MQ, C_Ma, rho, S'
    )
    Kpg, Kpv, Kig, Kiv, Kdg = sp.symbols('Kpg, Kpv, Kig, Kiv, Kdg')

    Tv_ref, Tg_ref = sp.symbols('Tv_ref, Tg_ref', positive=True, real=True)
    v0, g0 = sp.symbols('v0, g0', real=True)
    Av, Ag = sp.symbols('Av, Ag', positive=True, real=True)
    mu_q = sp.symbols('mu_q', real=True)

    vwp_dot = sp.Integer(0)
    gwp_dot = sp.Integer(0)
    tau_dot = sp.Integer(1)
    q_dot = sp.Integer(0)

    vref_dot = (vwp - vref) / Tv_ref
    gref_dot = (gwp - gref) / Tg_ref

    e_g = z_1 - gamma
    e_v = vref - V
    delta_g = Kpg * e_g + Kig * x_ig + Kdg * Q
    delta_v = Kpv * e_v + Kiv * x_iv
    T_act, elv_act = me.dynamicsymbols('T_act, elv_act')

    T_cmd = delta_v
    elv_cmd = delta_g

    Tmin, Tmax, dTmax, tau_T = sp.symbols('Tmin, Tmax, dTmax, tau_T', real=True)
    emin, emax, demax, tau_elv = sp.symbols('emin, emax, demax, tau_elv', real=True)

    T_act_dot_raw  = (T_cmd  - T_act) / tau_T
    elv_act_dot_raw = (elv_cmd - elv_act) / tau_elv

    gust = me.dynamicsymbols('gust')
    tau_gust = sp.symbols('tau_gust', positive=True, real=True)
    gust_dot = -gust / tau_gust

    C_L = C_L0 + C_La * alpha
    C_D = C_D0 + k_CLCD * C_L**2
    C_M = C_M0 + C_Ma * alpha + C_Me * elv_act + C_MQ * Q
    q_dyn = rho * V**2 / 2

    F_w = (
        -C_L*q_dyn*S * frame_w.z
        - C_D*q_dyn*S * frame_w.x
        + W * frame_e.z
        + T_act * frame_b.x
    ).express(frame_w).simplify() + gust * frame_w.x * m

    LM = m * V * frame_w.x
    eom_trans = (LM.diff(t, frame_e).simplify() - F_w).to_matrix(frame_w)

    M_b = (C_M*q_dyn*S * frame_b.y).express(frame_w).simplify()
    AM = J_y * frame_b.ang_vel_in(frame_e).express(frame_w).subs(
        gamma.diff(t) - alpha.diff(t), Q
    )
    eom_rot = (AM.diff(t, frame_e) - M_b).to_matrix(frame_b)

    x = sp.Matrix([V, alpha, gamma, Q])
    dx = x.diff(t)
    eoms = sp.Matrix([
        eom_rot[1],
        eom_trans[0],
        eom_trans[2],
        gamma.diff(t) + alpha.diff(t) - Q,
    ])

    sol = sp.solve(eoms, dx, dict=True)[0]
    f4 = sp.Matrix([sol[dxi] for dxi in dx])  # [Vdot, alphadot, gammadot, Qdot]

    zeta, omega = sp.symbols('zeta, omega', positive=True, real=True)
    z1dot = z_2
    z2dot = -2*zeta*omega*z_2 - omega**2*(z_1 - gref)

    f = sp.Matrix.vstack(
        f4,
        sp.Matrix([e_g]),        # x_ig_dot
        sp.Matrix([e_v]),        # x_iv_dot
        sp.Matrix([z1dot]),      # z_1_dot
        sp.Matrix([z2dot]),      # z_2_dot

        sp.Matrix([vwp_dot]),
        sp.Matrix([gwp_dot]),
        sp.Matrix([vref_dot]),
        sp.Matrix([gref_dot]),
        sp.Matrix([tau_dot]),
        sp.Matrix([q_dot]),

        sp.Matrix([T_act_dot_raw]),
        sp.Matrix([elv_act_dot_raw]),

        sp.Matrix([gust_dot]),
    )

    params = {
        m: 1.0, J_y: 1.0, g: 9.8,
        C_La: 3.0, C_L0: 0.1, k_CLCD: 0.1, C_D0: 0.03,
        C_M0: 0.01, C_Me: 0.1, C_MQ: -0.1, C_Ma: -0.1,
        rho: 1.225, S: 1.0,
        Kpg: 4.0, Kig: 0.8, Kdg: -1.2,
        Kpv: 1.75, Kiv: 0.75,

        zeta: 0.9,
        omega: 1.2,

        v0: 6.5,
        g0: 0.0,
        Av: 0.8,
        Ag: 0.06,
        mu_q: 3.99,
        emin: -0.35,
        emax: 0.35,
        demax: 2.5,
        tau_elv: 0.05,
        Tmin: 0.0,
        Tmax: 3.0,
        dTmax: 8.0,
        tau_T: 0.10,
        tau_gust: 2.0,
        Tv_ref: 0.4,
        Tg_ref: 0.3,
    }

    p = list(params.keys())
    p0 = np.array([float(params[key]) for key in p], dtype=float)

    state_syms = [
        V, alpha, gamma, Q, x_ig, x_iv, z_1, z_2,
        vwp, gwp, vref, gref, tau, q,
        T_act, elv_act,
        gust,
    ]
    # existing at the end of build_aircraft_model
    f_eval = sp.lambdify([state_syms, p], f[:], modules="numpy")
    
    idx = {
        "zeta": p.index(zeta),
        "omega": p.index(omega),
        "v0": p.index(v0),
        "g0": p.index(g0),
        "Av": p.index(Av),
        "Ag": p.index(Ag),
        "mu_q": p.index(mu_q),
        "Tv_ref": p.index(Tv_ref),
        "Tg_ref": p.index(Tg_ref),
        "Tmin": p.index(Tmin),
        "Tmax": p.index(Tmax),
        "dTmax": p.index(dTmax),
        "emin": p.index(emin),
        "emax": p.index(emax),
        "demax": p.index(demax),
    }
    
    # NEW: parameter-substituted longitudinal RHS
    f4_num = f4.subs(params)  # params is the dict defined above
    
    return f_eval, p, p0, idx, {
        "f4": f4,              # original symbolic
        "f4_num": f4_num,      # NEW: numeric-parameter version
        "state_syms": state_syms,
        "params": params,
        "p_syms": p,
        "p0": p0,
        "vref_u": vref,
        "gammaref_u": gref,
        "T_act_u": T_act,
        "elv_act_u": elv_act,
        "gust_u": gust,
        "vref_state": vref,
        "gref_state": gref,
    }

 
