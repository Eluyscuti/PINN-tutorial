
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R
from scipy.signal import savgol_filter




INPUT_PATH = "Data_proccessing/plot_data_foxglove_manual_feb6.csv"
OUTPUT_PROCESSED_PATH = "Data_proccessing/manual_processed_feb6_file1.csv"
OUTPUT_WINDOWS_PATH = "Data_proccessing/longitudinal_windows_report_feb6_file1.csv"
OUTPUT_SEGMENTS_PATH = "Data_proccessing/longitudinal_segments_feb6_file1.csv"
OUTPUT_SUMMARY_JSON = "Data_proccessing/longitudinal_summary_feb6_file1.json"
PLOT_TRAJ_PATH = "Data_proccessing/trajectory_longitudinal_windows_feb6_file1.png"
PLOT_DIAG_PATH = "Data_proccessing/trajectory_window_diagnostics_feb6_file1.png"

JOY_PREFIX = "/cub1/joy_serial_status.data"
ODOM_PREFIX = "/cub1/odom.pose.pose"

# =========================
# Windowing and smoothing
# =========================
WIN_S = 5.0
STEP_S = 2.5
MIN_POINTS_PER_WINDOW = 30

SAVGOL_WINDOW = 11   # ~0.11 s at 100 Hz
SAVGOL_POLY = 3

# =========================
# Path-quality thresholds
# =========================
MIN_MEAN_SPEED_3D_MPS = 1.0
MIN_FORWARD_SPEED_MPS = 2.0
MAX_LAT_RATIO = 0.30
MAX_HEADING_STD_DEG = 15.0
MIN_PROGRESS_M = 5.0

# Soft diagnostics only
ROLL_DIAG_WARN_DEG = 25.0

def _clip(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.minimum(np.maximum(x, lo), hi)


def normalize_throttle_ppm(ppm: np.ndarray) -> np.ndarray:
    return _clip((ppm - 1000.0) / 1000.0, 0.0, 1.0)


def normalize_symmetric_ppm(ppm: np.ndarray) -> np.ndarray:
    return _clip((ppm - 1500.0) / 500.0, -1.0, 1.0)


def normalize_elevator_ppm(ppm: np.ndarray) -> np.ndarray:
    return _clip(-(ppm - 1500.0) / 500.0, -1.0, 1.0)


def normalize_mode_ppm(ppm: np.ndarray) -> np.ndarray:
    return _clip((ppm - 1500.0) / 500.0, -1.0, 1.0)


def quat_to_roll_deg(qx, qy, qz, qw) -> np.ndarray:
    qx = np.asarray(qx, dtype=float)
    qy = np.asarray(qy, dtype=float)
    qz = np.asarray(qz, dtype=float)
    qw = np.asarray(qw, dtype=float)

    n = np.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
    n = np.where(n > 1e-12, n, 1.0)
    qx, qy, qz, qw = qx/n, qy/n, qz/n, qw/n

    sinr_cosp = 2.0 * (qw*qx + qy*qz)
    cosr_cosp = 1.0 - 2.0 * (qx*qx + qy*qy)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    return np.rad2deg(roll)


def quat_to_pitch_deg(qx, qy, qz, qw) -> np.ndarray:
    qx = np.asarray(qx, dtype=float)
    qy = np.asarray(qy, dtype=float)
    qz = np.asarray(qz, dtype=float)
    qw = np.asarray(qw, dtype=float)

    n = np.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
    n = np.where(n > 1e-12, n, 1.0)
    qx, qy, qz, qw = qx/n, qy/n, qz/n, qw/n

    sinp = 2.0 * (qw*qy - qz*qx)
    sinp = np.clip(sinp, -1.0, 1.0)
    pitch = np.arcsin(sinp)
    return np.rad2deg(pitch)



def quat_to_euler(qx, qy, qz, qw) -> np.ndarray:
    q = [qx, qy, qz, qw]  # NOTE ORDER!
    r = R.from_quat(q)

    roll, pitch, yaw = r.as_euler('xyz', degrees=False)



def wrap_to_pi(a: np.ndarray) -> np.ndarray:
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def pick_savgol_window(n: int, desired: int = SAVGOL_WINDOW, poly: int = SAVGOL_POLY) -> int:
    w = min(desired, n if n % 2 == 1 else n - 1)
    if w < poly + 2:
        w = poly + 2
    if w % 2 == 0:
        w += 1
    if w > n:
        w = n if n % 2 == 1 else n - 1
    return max(w, poly + 2 + ((poly + 2) % 2 == 0))


def smooth_signal(x: np.ndarray, desired_window: int = SAVGOL_WINDOW, poly: int = SAVGOL_POLY) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 7:
        return x.copy()
    w = pick_savgol_window(n, desired_window, poly)
    if w <= poly:
        return x.copy()
    return savgol_filter(x, window_length=w, polyorder=poly, mode="interp")


def compute_kinematics(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy().sort_values("time").reset_index(drop=True)

    t = out["time"].to_numpy(dtype=float)
    x = out["x"].to_numpy(dtype=float)
    y = out["y"].to_numpy(dtype=float)
    z = out["z"].to_numpy(dtype=float)

    x_s = smooth_signal(x)
    y_s = smooth_signal(y)
    z_s = smooth_signal(z)

    vx = np.gradient(x_s, t)
    vy = np.gradient(y_s, t)
    vz = np.gradient(z_s, t)

    vx_s = smooth_signal(vx)
    vy_s = smooth_signal(vy)
    vz_s = smooth_signal(vz)

    speed_xy = np.sqrt(vx_s**2 + vy_s**2)
    speed_3d = np.sqrt(vx_s**2 + vy_s**2 + vz_s**2)
    chi = np.arctan2(vy_s, vx_s)

    out["x_s"] = x_s
    out["y_s"] = y_s
    out["z_s"] = z_s
    out["vx"] = vx_s
    out["vy"] = vy_s
    out["vz"] = vz_s
    out["speed_xy"] = speed_xy
    out["speed_3d"] = speed_3d
    out["chi_deg"] = np.rad2deg(chi)
    return out


def principal_direction_xy(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    pts = np.column_stack([x, y])
    pts_centered = pts - pts.mean(axis=0, keepdims=True)
    cov = np.cov(pts_centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    direction = eigvecs[:, np.argmax(eigvals)]
    direction = direction / max(np.linalg.norm(direction), 1e-12)
    return direction


def scan_longitudinal_windows(df: pd.DataFrame) -> pd.DataFrame:
    t = df["time"].to_numpy(dtype=float)
    starts = np.arange(float(t[0]), float(t[-1]) - WIN_S + 1e-12, STEP_S)

    rows = []
    for s in starts:
        e = s + WIN_S
        w = df[(df["time"] >= s) & (df["time"] <= e)].copy()
        if len(w) < MIN_POINTS_PER_WINDOW:
            continue

        xw = w["x_s"].to_numpy(dtype=float)
        yw = w["y_s"].to_numpy(dtype=float)
        vxw = w["vx"].to_numpy(dtype=float)
        vyw = w["vy"].to_numpy(dtype=float)
        vzw = w["vz"].to_numpy(dtype=float)
        speed3 = w["speed_3d"].to_numpy(dtype=float)
        speedxy = w["speed_xy"].to_numpy(dtype=float)

        e_long = principal_direction_xy(xw, yw)

        # orient direction so that most progress is positive
        progress_signed = np.dot(np.array([xw[-1] - xw[0], yw[-1] - yw[0]]), e_long)
        if progress_signed < 0:
            e_long = -e_long
        e_lat = np.array([-e_long[1], e_long[0]])

        v_long = vxw * e_long[0] + vyw * e_long[1]
        v_lat = vxw * e_lat[0] + vyw * e_lat[1]

        mean_speed_3d = float(np.mean(speed3))
        mean_speed_xy = float(np.mean(speedxy))
        mean_forward_speed = float(np.mean(v_long))
        rms_forward_speed = float(np.sqrt(np.mean(v_long**2)))
        rms_lat_speed = float(np.sqrt(np.mean(v_lat**2)))
        lat_ratio = rms_lat_speed / max(rms_forward_speed, 1e-8)

        dx = xw[-1] - xw[0]
        dy = yw[-1] - yw[0]
        progress = float(dx * e_long[0] + dy * e_long[1])

        chi_w = np.arctan2(vyw, vxw)
        chi_ref = np.arctan2(e_long[1], e_long[0])
        heading_err = wrap_to_pi(chi_w - chi_ref)
        heading_std_deg = float(np.rad2deg(np.std(heading_err)))

        lateral_disp = (xw - xw[0]) * e_lat[0] + (yw - yw[0]) * e_lat[1]
        lateral_span = float(np.max(lateral_disp) - np.min(lateral_disp))
        forward_disp = (xw - xw[0]) * e_long[0] + (yw - yw[0]) * e_long[1]
        forward_span = float(np.max(forward_disp) - np.min(forward_disp))

        straightness = progress / max(forward_span, 1e-8)

        phi_max = float(np.max(np.abs(w["phi_deg"]))) if "phi_deg" in w.columns else np.nan
        theta_std = float(np.std(w["theta_deg"])) if "theta_deg" in w.columns else np.nan

        is_long = (
            (mean_speed_3d >= MIN_MEAN_SPEED_3D_MPS) and
            (mean_forward_speed >= MIN_FORWARD_SPEED_MPS) and
            (lat_ratio <= MAX_LAT_RATIO) and
            (heading_std_deg <= MAX_HEADING_STD_DEG) and
            (progress >= MIN_PROGRESS_M)
        )

        quality_score = (
            3.0 * lat_ratio
            + 1.5 * (heading_std_deg / max(MAX_HEADING_STD_DEG, 1e-6))
            + 1.0 * max(0.0, MIN_FORWARD_SPEED_MPS - mean_forward_speed)
            + 0.7 * max(0.0, MIN_PROGRESS_M - progress)
            + 0.5 * max(0.0, 1.0 - straightness)
        )

        rows.append({
            "t_start": s,
            "t_end": e,
            "N": len(w),
            "mean_speed_3d_mps": mean_speed_3d,
            "mean_speed_xy_mps": mean_speed_xy,
            "mean_forward_speed_mps": mean_forward_speed,
            "rms_forward_speed_mps": rms_forward_speed,
            "rms_lat_speed_mps": rms_lat_speed,
            "lat_ratio": lat_ratio,
            "heading_std_deg": heading_std_deg,
            "progress_m": progress,
            "forward_span_m": forward_span,
            "lateral_span_m": lateral_span,
            "straightness": straightness,
            "principal_dir_x": float(e_long[0]),
            "principal_dir_y": float(e_long[1]),
            "phi_max_deg": phi_max,
            "theta_std_deg": theta_std,
            "roll_warn": bool(phi_max > ROLL_DIAG_WARN_DEG) if np.isfinite(phi_max) else False,
            "is_longitudinal": bool(is_long),
            "quality_score": quality_score,
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out = out.sort_values(
        by=["is_longitudinal", "quality_score", "lat_ratio", "heading_std_deg", "progress_m"],
        ascending=[False, True, True, True, False],
    ).reset_index(drop=True)
    return out


def merge_good_windows(win_report: pd.DataFrame, min_overlap: float = 0.0) -> pd.DataFrame:
    good = win_report[win_report["is_longitudinal"]].copy()
    if good.empty:
        return pd.DataFrame(columns=["seg_id", "t_start", "t_end", "duration_s", "n_windows"])

    good = good.sort_values("t_start").reset_index(drop=True)

    merged = []
    cur_start = float(good.loc[0, "t_start"])
    cur_end = float(good.loc[0, "t_end"])
    count = 1

    for i in range(1, len(good)):
        s = float(good.loc[i, "t_start"])
        e = float(good.loc[i, "t_end"])
        if s <= cur_end - min_overlap + 1e-12:
            cur_end = max(cur_end, e)
            count += 1
        else:
            merged.append({"t_start": cur_start, "t_end": cur_end, "n_windows": count})
            cur_start, cur_end, count = s, e, 1

    merged.append({"t_start": cur_start, "t_end": cur_end, "n_windows": count})

    segs = pd.DataFrame(merged)
    segs["seg_id"] = np.arange(1, len(segs) + 1)
    segs["duration_s"] = segs["t_end"] - segs["t_start"]
    return segs[["seg_id", "t_start", "t_end", "duration_s", "n_windows"]]


def make_trajectory_plot(df: pd.DataFrame, segs: pd.DataFrame, out_path: str) -> None:
    plt.figure(figsize=(10, 8))
    plt.plot(df["x_s"], df["y_s"], linewidth=1.0, label="Smoothed trajectory")

    if not segs.empty:
        for _, seg in segs.iterrows():
            w = df[(df["time"] >= seg["t_start"]) & (df["time"] <= seg["t_end"])]
            if len(w) > 1:
                plt.plot(w["x_s"], w["y_s"], linewidth=2.5, label=f"Seg {int(seg['seg_id'])}")
                plt.text(float(w["x_s"].iloc[0]), float(w["y_s"].iloc[0]), f"S{int(seg['seg_id'])}")

    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title("Mocap trajectory with candidate longitudinal segments")
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()



def make_trajectory_plot_3d(df: pd.DataFrame, segs: pd.DataFrame, out_path: str) -> None:
    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Full smoothed trajectory
    ax.plot(
        df["x_s"], df["y_s"], df["z_s"],
        linewidth=1.0,
        label="Smoothed trajectory"
    )

    # Candidate segments
    if not segs.empty:
        for _, seg in segs.iterrows():
            w = df[(df["time"] >= seg["t_start"]) & (df["time"] <= seg["t_end"])]
            if len(w) > 1:
                ax.plot(
                    w["x_s"], w["y_s"], w["z_s"],
                    linewidth=3.0,
                    label=f"Seg {int(seg['seg_id'])}"
                )
                ax.text(
                    float(w["x_s"].iloc[0]),
                    float(w["y_s"].iloc[0]),
                    float(w["z_s"].iloc[0]),
                    f"S{int(seg['seg_id'])}"
                )

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.set_title("3D mocap trajectory with candidate longitudinal segments")
    ax.legend(loc="best", fontsize=8)

    # Optional: make aspect less distorted
    try:
        x = df["x_s"].to_numpy()
        y = df["y_s"].to_numpy()
        z = df["z_s"].to_numpy()
        ax.set_box_aspect((
            np.ptp(x) if np.ptp(x) > 0 else 1.0,
            np.ptp(y) if np.ptp(y) > 0 else 1.0,
            np.ptp(z) if np.ptp(z) > 0 else 1.0,
        ))
    except Exception:
        pass

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)

def make_diagnostic_plot(win_report: pd.DataFrame, out_path: str) -> None:
    if win_report.empty:
        return

    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

    t_mid = 0.5 * (win_report["t_start"] + win_report["t_end"])
    mask = win_report["is_longitudinal"].to_numpy(dtype=bool)

    axes[0].plot(t_mid, win_report["mean_forward_speed_mps"], linewidth=1.2)
    axes[0].scatter(t_mid[mask], win_report.loc[mask, "mean_forward_speed_mps"], s=18)
    axes[0].axhline(MIN_FORWARD_SPEED_MPS, linestyle="--", linewidth=1.0)
    axes[0].set_ylabel("Forward speed")

    axes[1].plot(t_mid, win_report["lat_ratio"], linewidth=1.2)
    axes[1].scatter(t_mid[mask], win_report.loc[mask, "lat_ratio"], s=18)
    axes[1].axhline(MAX_LAT_RATIO, linestyle="--", linewidth=1.0)
    axes[1].set_ylabel("Lat ratio")

    axes[2].plot(t_mid, win_report["heading_std_deg"], linewidth=1.2)
    axes[2].scatter(t_mid[mask], win_report.loc[mask, "heading_std_deg"], s=18)
    axes[2].axhline(MAX_HEADING_STD_DEG, linestyle="--", linewidth=1.0)
    axes[2].set_ylabel("Heading std [deg]")

    axes[3].plot(t_mid, win_report["progress_m"], linewidth=1.2)
    axes[3].scatter(t_mid[mask], win_report.loc[mask, "progress_m"], s=18)
    axes[3].axhline(MIN_PROGRESS_M, linestyle="--", linewidth=1.0)
    axes[3].set_ylabel("Progress [m]")
    axes[3].set_xlabel("Window midpoint time [s]")

    for ax in axes:
        ax.grid(True, alpha=0.3)

    fig.suptitle("Window diagnostics for longitudinal-segment selection")
    fig.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


def build_sysid_ready_subset(df: pd.DataFrame, segs: pd.DataFrame) -> pd.DataFrame:
    if segs.empty:
        return pd.DataFrame()

    chunks = []
    for _, seg in segs.iterrows():
        w = df[(df["time"] >= seg["t_start"]) & (df["time"] <= seg["t_end"])].copy()
        if w.empty:
            continue
        w["seg_id"] = int(seg["seg_id"])
        chunks.append(w)

    out = pd.concat(chunks, ignore_index=True)
    return out


def main():
    in_path = Path(INPUT_PATH)
    if not in_path.exists():
        raise FileNotFoundError(f"Missing input file: {in_path.resolve()}")

    raw = pd.read_csv(in_path)

    required_cols = {"elapsed time", "topic", "value"}
    missing = required_cols - set(raw.columns)
    if missing:
        raise ValueError(f"Input CSV missing required columns: {missing}. Found: {list(raw.columns)}")

    raw = raw.rename(columns={"elapsed time": "time"})
    raw["time"] = pd.to_numeric(raw["time"], errors="coerce")
    raw["value"] = pd.to_numeric(raw["value"], errors="coerce")
    raw = raw.dropna(subset=["time", "topic", "value"]).copy()
    raw = raw.sort_values("time").reset_index(drop=True)

    # 1) joystick wide
    joy_mask = raw["topic"].str.startswith(JOY_PREFIX)
    joy_rows = raw.loc[joy_mask].copy()
    m = joy_rows["topic"].str.extract(r"\[(\d+)\]")
    if m.isna().all().all():
        raise ValueError("Could not extract joystick axis indices.")
    joy_rows["axis"] = m[0].astype(int)
    axis_name = {0: "Throttle", 1: "Aileron", 2: "Elevator", 3: "Rudder", 4: "Mode"}
    joy_rows["axis_name"] = joy_rows["axis"].map(axis_name)

    joy_wide = (
        joy_rows.pivot_table(index="time", columns="axis_name", values="value", aggfunc="last")
        .sort_index()
        .ffill()
    )

    # 2) mocap wide
    mocap_map = {
        f"{ODOM_PREFIX}.position.x": "x",
        f"{ODOM_PREFIX}.position.y": "y",
        f"{ODOM_PREFIX}.position.z": "z",
        f"{ODOM_PREFIX}.orientation.x": "quat_x",
        f"{ODOM_PREFIX}.orientation.y": "quat_y",
        f"{ODOM_PREFIX}.orientation.z": "quat_z",
        f"{ODOM_PREFIX}.orientation.w": "quat_w",
    }
    mocap_rows = raw.loc[raw["topic"].isin(mocap_map.keys())].copy()
    if mocap_rows.empty:
        raise ValueError("No mocap pose topics matched expected odom topics.")
    mocap_rows["col"] = mocap_rows["topic"].map(mocap_map)
    mocap_wide = (
        mocap_rows.pivot_table(index="time", columns="col", values="value", aggfunc="last")
        .sort_index()
    )

    # 3) merge
    wide = pd.merge_asof(
        mocap_wide.reset_index().sort_values("time"),
        joy_wide.reset_index().sort_values("time"),
        on="time",
        direction="backward",
    )
    for col in ["Aileron", "Elevator", "Throttle", "Rudder", "Mode"]:
        if col in wide.columns:
            wide[col] = wide[col].ffill().bfill()
        else:
            wide[col] = np.nan

    # 4) normalize controls
    wide["Throttle"] = normalize_throttle_ppm(wide["Throttle"].to_numpy(dtype=float))
    wide["Aileron"] = normalize_symmetric_ppm(wide["Aileron"].to_numpy(dtype=float))
    wide["Elevator"] = normalize_elevator_ppm(wide["Elevator"].to_numpy(dtype=float))
    wide["Rudder"] = normalize_symmetric_ppm(wide["Rudder"].to_numpy(dtype=float))
    wide["Mode"] = normalize_mode_ppm(wide["Mode"].to_numpy(dtype=float))

    # 5) orientation diagnostics
    for c in ["quat_x", "quat_y", "quat_z", "quat_w"]:
        if c not in wide.columns:
            wide[c] = np.nan

    wide["phi_deg"] = quat_to_roll_deg(
        wide["quat_x"].to_numpy(dtype=float),
        wide["quat_y"].to_numpy(dtype=float),
        wide["quat_z"].to_numpy(dtype=float),
        wide["quat_w"].to_numpy(dtype=float),
    )
    wide["theta_deg"] = quat_to_pitch_deg(
        wide["quat_x"].to_numpy(dtype=float),
        wide["quat_y"].to_numpy(dtype=float),
        wide["quat_z"].to_numpy(dtype=float),
        wide["quat_w"].to_numpy(dtype=float),
    )

    """wide["phi_deg"], wide["theta_deg"], yaw = quat_to_euler(wide["quat_x"].to_numpy(dtype=float),
        wide["quat_y"].to_numpy(dtype=float),
        wide["quat_z"].to_numpy(dtype=float),
        wide["quat_w"].to_numpy(dtype=float),)"""

    # 6) kinematics
    wide = compute_kinematics(wide)

    # 7) scan and merge
    win_report = scan_longitudinal_windows(wide)
    segs = merge_good_windows(win_report)

    # 8) save outputs
    processed_cols = [
        "time",
        "Throttle", "Aileron", "Elevator", "Rudder", "Mode",
        "x", "y", "z", "x_s", "y_s", "z_s",
        "vx", "vy", "vz", "speed_xy", "speed_3d", "chi_deg",
        "quat_x", "quat_y", "quat_z", "quat_w",
        "phi_deg", "theta_deg",
    ]
    for c in processed_cols:
        if c not in wide.columns:
            wide[c] = np.nan
    wide[processed_cols].to_csv(OUTPUT_PROCESSED_PATH, index=False)
    win_report.to_csv(OUTPUT_WINDOWS_PATH, index=False)
    segs.to_csv(OUTPUT_SEGMENTS_PATH, index=False)

    make_trajectory_plot(wide, segs, PLOT_TRAJ_PATH)
    make_diagnostic_plot(win_report, PLOT_DIAG_PATH)
    PLOT_TRAJ_3D_PATH = "Data_proccessing/trajectory_longitudinal_windows_feb6_3d.png"
    make_trajectory_plot_3d(wide, segs, PLOT_TRAJ_3D_PATH)

    sysid_subset = build_sysid_ready_subset(wide, segs)
    sysid_subset_path = "Data_proccessing/sysid_ready_longitudinal_subset_feb6.csv"
    sysid_subset.to_csv(sysid_subset_path, index=False)
    sysid_xy_subset = project_xy_plane(sysid_subset)
    print("checking excitation")
    check_segment_excitation(sysid_xy_subset)

    summary = {
        "input_path": str(Path(INPUT_PATH).resolve()),
        "processed_path": str(Path(OUTPUT_PROCESSED_PATH).resolve()),
        "windows_report_path": str(Path(OUTPUT_WINDOWS_PATH).resolve()),
        "segments_path": str(Path(OUTPUT_SEGMENTS_PATH).resolve()),
        "sysid_ready_subset_path": str(Path(sysid_subset_path).resolve()),
        "trajectory_plot_path": str(Path(PLOT_TRAJ_PATH).resolve()),
        "diagnostic_plot_path": str(Path(PLOT_DIAG_PATH).resolve()),
        "n_samples": int(len(wide)),
        "n_windows": int(len(win_report)),
        "n_good_windows": int(win_report["is_longitudinal"].sum()) if not win_report.empty else 0,
        "n_segments": int(len(segs)),
        "segments": segs.to_dict(orient="records"),
        "thresholds": {
            "MIN_MEAN_SPEED_3D_MPS": MIN_MEAN_SPEED_3D_MPS,
            "MIN_FORWARD_SPEED_MPS": MIN_FORWARD_SPEED_MPS,
            "MAX_LAT_RATIO": MAX_LAT_RATIO,
            "MAX_HEADING_STD_DEG": MAX_HEADING_STD_DEG,
            "MIN_PROGRESS_M": MIN_PROGRESS_M,
            "WIN_S": WIN_S,
            "STEP_S": STEP_S,
            "SAVGOL_WINDOW": SAVGOL_WINDOW,
            "SAVGOL_POLY": SAVGOL_POLY,
        },
    }
    Path(OUTPUT_SUMMARY_JSON).write_text(json.dumps(summary, indent=2))

    print("\n=== Robust trajectory-based longitudinal segmentation ===")
    print(f"Samples: {len(wide)}")
    print(f"Windows scanned: {len(win_report)}")
    print(f"Good windows: {int(win_report['is_longitudinal'].sum()) if not win_report.empty else 0}")
    print(f"Merged segments: {len(segs)}")
    if not segs.empty:
        print("\nCandidate segments for SysID:")
        print(segs.to_string(index=False))
    else:
        print("\nNo segment met the current thresholds. Inspect plots and relax thresholds if needed.")

def project_xy_plane(full_sysid_subset):
    print("subsets to project")
   
    data_labels = full_sysid_subset.keys().tolist()
    print(data_labels)

    max_seg_id = int(full_sysid_subset['seg_id'].max())
    print(f"max seg id {max_seg_id}")
    all_segments_list = []

    for i in range(1, max_seg_id+1):


        sysid_subset = full_sysid_subset[full_sysid_subset['seg_id'] == i].copy()

        if sysid_subset.empty:
            continue

        x_data = sysid_subset['x_s']
        y_data = sysid_subset['y_s']
        z_data = sysid_subset['z_s']
        time_data = sysid_subset['time']
        theta = sysid_subset['theta_deg']
        print(f"x: {x_data}")
        along_track_s = get_along_track_distance(x_data, y_data, z_data)
        s, n = get_along_track_distance_pca(x_data,y_data,z_data)

        u = np.gradient(s, time_data)
        gamma = np.arctan2(np.gradient(z_data, time_data), np.gradient(s, time_data))
        alpha = theta - gamma

        sysid_subset['s'] = s
        sysid_subset['n'] = n
        sysid_subset['u_along_track'] = u
        sysid_subset['alpha_deg'] = alpha
        sysid_subset['gamma_deg'] = gamma

        



        plot_projected_flight_path(s,list(z_data))
        plot_projected_flight_velocity(u, list(time_data))

        cols = list(sysid_subset.columns)

        # 2. Remove 'seg_id' from its current position and append it to the end
        if 'seg_id' in cols:
            cols.remove('seg_id')
            cols.append('seg_id')

        # 3. Reorder the DataFrame using the new column list
        sysid_subset = sysid_subset[cols]

        all_segments_list.append(sysid_subset)

    if all_segments_list:
        full_stacked_df = pd.concat(all_segments_list, axis=0, ignore_index=True)

       

        # 9. Save the full Master CSV
        output_filename = "Data_proccessing/sysid_final_processed_feb6.csv"
        full_stacked_df.to_csv(output_filename, index=False)
        
        print(f"Successfully created: {output_filename} with {len(all_segments_list)} segments.")

        return full_stacked_df



    # Save to CSV
    # index=False prevents pandas from writing row numbers as the first column



def get_along_track_distance(x, y ,z):
    dx = x.diff().fillna(0)
    dy = y.diff().fillna(0)
    dz = z.diff().fillna(0)

    # 2. Calculate the 3D step distance for each row
    step_dist = np.sqrt(dx**2 + dy**2 + dz**2)
    return step_dist.cumsum()

def get_along_track_distance_pca(x,y,z):
    print("x mean")
    print(np.mean(x))

    #get centered coords
    x_centered = x - np.mean(x)
    y_centered = y - np.mean(y)

    #use coords to calculate covarience matrix
    coords_stacked = np.vstack([x_centered, y_centered])
    cov_matrix = np.cov(coords_stacked)
    print(f"cov matrix: {cov_matrix}")

    #get eigen values and eignen vectors to find principal directions of 
    eigvals, eigvecs = np.linalg.eigh(cov_matrix)

    # The eigenvector with the LARGEST eigenvalue is our path direction (s)
    # The other eigenvector is our lateral deviation (n)
    idx = np.argmax(eigvals)
    e_long = eigvecs[:, idx]      # Along-track unit vector
    e_lat  = eigvecs[:, 1 - idx]  # Cross-track unit vector (perpendicular)
    
    # 5. Project the centered data onto the new axes
    # s = Progress along the best-fit line
    # n = Deviation (wobble) from that line
    s = x_centered * e_long[0] + y_centered * e_long[1]
    n = x_centered * e_lat[0] + y_centered * e_lat[1]
    
    # Optional: Shift 's' so it starts at 0 for easier plotting
    s = s - s.iloc[0]
    return s, n

def plot_projected_flight_velocity(u,t):
     # 5. Create the 2D Plot
    plt.figure(figsize=(10, 5))
    plt.plot(t, u, color='blue', linewidth=2, label='Flight Path')
    
    # Formatting
    plt.ylabel('Along-Track velcity (u) m/s')
    plt.xlabel('Time (s)')
    plt.title('2D Longitudinal Velocity Profile (PCA Projection)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    
    # Optional: ensure the plot isn't distorted
    # plt.axis('equal') 
    
    plt.tight_layout()
    plt.show()

def plot_projected_flight_path(s,z):
     # 5. Create the 2D Plot
    plt.figure(figsize=(10, 5))
    plt.plot(s, z, color='blue', linewidth=2, label='Flight Path')
    
    # Formatting
    plt.xlabel('Along-Track Distance (s) [meters]')
    plt.ylabel('Altitude (z) [meters]')
    plt.title('2D Longitudinal Vertical Profile (PCA Projection)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    
    # Optional: ensure the plot isn't distorted
    # plt.axis('equal') 
    
    plt.tight_layout()
    plt.show()

def check_segment_excitation(full_sysid_subset):
    max_seg_id = int(full_sysid_subset['seg_id'].max())
    print(f"max seg id {max_seg_id}")
    all_segments_list = []

    for i in range(1, max_seg_id+1):
        
        sysid_subset = full_sysid_subset[full_sysid_subset['seg_id'] == i].copy()

        if sysid_subset.empty:
            continue

        data_labels = sysid_subset.keys().tolist()
       # print(data_labels)

        elevator_input = sysid_subset['Elevator']
        throttle_input = sysid_subset['Throttle']

        alpha = sysid_subset['alpha_deg']
        gamma = sysid_subset['gamma_deg']

        u_along_track = sysid_subset['u_along_track']

        pitch = sysid_subset['theta_deg']
        time = sysid_subset['time']

        # window_length must be odd
        theta_filt = savgol_filter(pitch, window_length=11, polyorder=3)

        # Compute derivative directly
        dt = np.mean(np.diff(time))
        #use savgol filter to calulate pitch rate with noise reduction
        q = savgol_filter(pitch, window_length=11, polyorder=3, deriv=1, delta=dt)

        #sanity check for pitch rate calculation
        plt.plot(time, pitch, label='theta (rad)')
        plt.plot(time, q, label='q (rad/s)')
        plt.legend()
        plt.xlabel('Time (s)')
        plt.show()

        u = [elevator_input, throttle_input]
        y = [u_along_track, alpha, gamma, q]

        phi = np.column_stack((u_along_track, alpha, gamma, q, throttle_input, elevator_input))
        phi = phi - np.mean(phi, axis=0)

        




        #inputs: throttle, elevator 
        #outputs: u_along track, alpha, gamma, pitch rate
        #steps: 1) get regressor matrix of inputs and outputs, 2) rank test
        #3) Fisher information test(eigenvalue and curvature)
        #4) control input variance and correlation check
        #5) input frequency content test
        #6) Signal to Noise ratio test

        ratio_e_threshold = 0.1 #higher ratio is good
        ratio_t_threshold = 0.05



        #implement rank test:

        rank_test_result =  rank_test(phi) #rank must be equal to parameters
        good_rank = rank_test_result[2]
      #  print(f"rank test result: {rank_test_result}")

        #get fisher information matrix

        fisher_result = get_fisher_info_matrix(phi)
        #print(fisher_result)

        good_fisher = fisher_result['min_eig'] >= 1e-3 and fisher_result['cond'] <= 1e5 #high eignvalues not close to zero are good low codition is good
            

        #input variance and correlation test
            
        varience_result = input_tests(throttle_input, elevator_input)

        good_variance_t =  varience_result['ratio_T'] > ratio_t_threshold #
        good_variance_e = varience_result['ratio_e'] > ratio_e_threshold
        good_corr = abs(varience_result['corr']) <0.8 #low correlation is good

        #frequency band test

        freq_e = frequency_test(elevator_input, time)
        freq_T = frequency_test(throttle_input, time)   

        good_e_freq = (
        freq_e["mid"] > 0.1 and #requencies spread accross multiple bands is good
        freq_e["high"] > 0.1 and
        freq_e["low"] > 0.1
        )  
        
        good_t_freq = (
        freq_T["low"] > 0.1
        )

        #print(freq_e)
        #print(freq_T)

        #signal to noise test

        snr_e = snr_test(elevator_input)
        snr_t = snr_test(throttle_input)
        
        good_snr_e = snr_e > 0.65 #high snr is good
        good_snr_t = snr_t > 0.65

        # print(f"rank test result: {good_rank}") 
        # print(f"fisher result: {good_fisher}")
        # print(f"varience result t: {good_variance_t}")
        # print(f"varience result e: {good_variance_e}")
        # print(f"good corr result: {good_corr}")
        # print(f"good elevator frequency: {good_e_freq}")
        # print(f"good throttle frequency: {good_t_freq}")
        # print(f"good snr e: {good_snr_e}")
        # print(f"good snr t: {good_snr_t}")

        results = [good_rank, good_fisher, good_variance_t, good_variance_e, good_corr, good_e_freq, good_t_freq, good_snr_e, good_snr_t]

        true_count = sum(results)

        if (true_count >= 7):
            print(f"segment {i} is good")
            all_segments_list.append(sysid_subset)

    if all_segments_list:
        full_stacked_df = pd.concat(all_segments_list, axis=0, ignore_index=True)

       

        # 9. Save the full Master CSV
        output_filename = "Data_proccessing/sysid_final_excitation_processed_feb6.csv"
        full_stacked_df.to_csv(output_filename, index=False)
        
        print(f"Successfully created: {output_filename} with {len(all_segments_list)} segments.")

        return full_stacked_df 



#check rank of matrix is equal to number of parameters
def rank_test(phi):

    rank = np.linalg.matrix_rank(phi)

    parameters = phi.shape[1]
    return rank, parameters, (rank == parameters)

def get_fisher_info_matrix(phi):
    phi_norm = (phi - np.mean(phi, axis=0)) / np.std(phi, axis=0)
    F = phi_norm.T @ phi_norm
    
    eig_vals = np.linalg.eigvals(F)
    cond_number = np.linalg.cond(F)
    
    return {
        "F": F,
        "eigenvalues": eig_vals,
        "min_eig": np.min(eig_vals),
        "cond": cond_number
    }

def input_tests(delta_T, delta_e):
    var_T = np.var(delta_T)
    var_e = np.var(delta_e)
    
    corr = np.corrcoef(delta_T, delta_e)[0,1]

    range_e = np.max(delta_e) - np.min(delta_e)
    range_T = np.max(delta_T) - np.min(delta_T)

    ratio_e = np.sqrt(var_e) / range_e
    ratio_t = np.sqrt(var_T) / range_T
    
    return {
        "ratio_T": ratio_t,
        "ratio_e": ratio_e,
        "corr": corr
    }

def frequency_test(signal, t):
    dt = np.mean(np.diff(t))
    Fs = 1 / dt
    
    fft_vals = np.abs(np.fft.fft(signal))
    freqs = np.fft.fftfreq(len(signal), dt)
    
    # Only positive frequencies
    pos_mask = freqs > 0
    freqs = freqs[pos_mask]
    fft_vals = fft_vals[pos_mask]
    
    # Energy ratio (high vs total)


    low_energy = np.sum(fft_vals[freqs < 0.1]**2)
    mid_energy = np.sum((fft_vals[(freqs >= 0.1) & (freqs < 1)])**2)
    high_energy = np.sum(fft_vals[freqs >= 1]**2)

    total_energy = low_energy + mid_energy + high_energy
    
    ratio = high_energy / total_energy if total_energy > 0 else 0
    
    return {
        "low": low_energy / total_energy,
        "mid": mid_energy / total_energy,
        "high": high_energy / total_energy
    }

def snr_test(signal):
    # crude noise estimate via smoothing
    smooth = np.convolve(signal, np.ones(5)/5, mode='same')
    
    signal_power = np.var(signal)
    noise_power = np.var(signal - smooth)
    
    snr = signal_power / noise_power if noise_power > 0 else np.inf
    
    return snr



    


if __name__ == "__main__":
    main()