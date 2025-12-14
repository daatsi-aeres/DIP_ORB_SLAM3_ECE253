import matplotlib
matplotlib.use("Agg")  # headless + faster

import polars as pl
import matplotlib.pyplot as plt
import numpy as np
import os
import gc
from datetime import datetime



# -------------------------
# Auto-generated Run Tag
# -------------------------
# Format example: 2025-02-14_12-18PM
now = datetime.now()
run_tag = now.strftime("%Y-%m-%d_%I-%M%p")   # auto-format date + time

#output_dir = f"runs/{run_tag}"
#os.makedirs(output_dir, exist_ok=True)

base_runs_dir = "runs"
os.makedirs(base_runs_dir, exist_ok=True)

temp_output_dir = f"{base_runs_dir}/_tmp_{run_tag}"
final_output_dir = f"{base_runs_dir}/{run_tag}"
success = False

try:
    # Create TEMP directory only
    os.makedirs(temp_output_dir, exist_ok=True)

    output_dir = temp_output_dir   # <-- use temp dir everywhere

    print(f"Temporary run directory created: {output_dir}")

    # =========================
    # ALL YOUR EXISTING CODE
    # =========================
    # (no changes inside needed)

    success = True

except Exception as e:
    print("\n❌ ERROR OCCURRED — RUN ABORTED")
    print(str(e))

finally:
    if success:
        # Rename temp directory to final run name
        os.rename(temp_output_dir, final_output_dir)
        print(f"\n✅ Run completed successfully.")
        print(f"📁 Results saved to: {final_output_dir}")
    else:
        # Remove temp directory if it exists
        if os.path.exists(temp_output_dir):
            import shutil
            shutil.rmtree(temp_output_dir)
            print("🧹 Temporary run directory removed.")


print(f"Run tag generated: {run_tag}")


output_dir = f"runs/{run_tag}"
os.makedirs(output_dir, exist_ok=True)

def process_culling(raw_df):
    # Pivot CREATED/CULLED rows into columns
    pivot = raw_df.pivot_table(
        index="kf_id",
        columns="created",
        values="culled",
        aggfunc="first"
    ).reset_index()

    # Rename
    pivot = pivot.rename(columns={
        "CULLED": "culled",
        "CREATED": "created"
    }).fillna(0)

    # Compute totals
    pivot["total"] = pivot["culled"] + pivot["created"]

    # Avoid division by zero
    pivot["cull_ratio"] = pivot["culled"] / pivot["total"].replace(0, float("nan"))

    # Mean observations per keyframe
    pivot["mean_obs"] = pivot["total"] / 2

    return pivot



def savefig(name):
    plt.savefig(f"{output_dir}/{name}_{run_tag}.png", dpi=300)
    plt.close()
    gc.collect()  # free memory ASAP


# -------------------------
# Optimized CSV Loader
# -------------------------
def load_metrics(metric):
    base = pl.read_csv(f"baseline_good1/{metric}.csv", try_parse_dates=False, low_memory=True)
    dip  = pl.read_csv(f"dip_aggressive/{metric}.csv",      try_parse_dates=False, low_memory=True)
    return base.to_pandas(), dip.to_pandas()


def align_by_shortest_length(df_b, df_d, key="frame_id"):
    """
    Truncate both dataframes to the shortest length
    based on sorted key (default: frame_id).
    """
    df_b = df_b.sort_values(key).reset_index(drop=True)
    df_d = df_d.sort_values(key).reset_index(drop=True)

    min_len = min(len(df_b), len(df_d))

    return df_b.iloc[:min_len], df_d.iloc[:min_len]


# -------------------------
# Map Point Culling Data Processing
# -------------------------
df_b_raw, df_d_raw = load_metrics("map_culling")   # <-- raw CREATED / CULLED rows

# Process raw CREATED/CULLED rows
df_b = process_culling(df_b_raw)
df_d = process_culling(df_d_raw)

# Save processed data for record / reuse
df_b.to_csv(f"{output_dir}/map_culling_processed_baseline_{run_tag}.csv", index=False)
df_d.to_csv(f"{output_dir}/map_culling_processed_dip_{run_tag}.csv", index=False)


# -------------------------
# Map Point Creation Comparison (Baseline vs DIP)
# -------------------------
fig = plt.figure(figsize=(12,6))

plt.plot(
    df_b["kf_id"],
    df_b["created"],
    label="Baseline – Created",
    linewidth=2,
    color="tab:green"
)

plt.plot(
    df_d["kf_id"],
    df_d["created"],
    label="DIP – Created",
    linewidth=2,
    color="tab:green",
    linestyle="--"
)

plt.title("Map Point Creation per Keyframe: Baseline vs DIP")
plt.xlabel("Keyframe ID")
plt.ylabel("Number of Map Points Created")
plt.grid(True)
plt.legend()

fig.text(
    0.5, 0.02,
    "Interpretation: Higher or more consistent map point creation in the DIP pipeline "
    "suggests improved feature visibility and stability, particularly under degraded "
    "visual conditions such as motion blur or low contrast.",
    ha="center",
    va="bottom",
    fontsize=10,
    wrap=True
)

plt.subplots_adjust(bottom=0.25)
savefig("plot_map_points_created_comparison")


# -------------------------
# Map Point Culling Comparison (Baseline vs DIP)
# -------------------------
fig = plt.figure(figsize=(12,6))

plt.plot(
    df_b["kf_id"],
    df_b["culled"],
    label="Baseline – Culled",
    linewidth=2,
    color="tab:red"
)

plt.plot(
    df_d["kf_id"],
    df_d["culled"],
    label="DIP – Culled",
    linewidth=2,
    color="tab:red",
    linestyle="--"
)

plt.title("Map Point Culling per Keyframe: Baseline vs DIP")
plt.xlabel("Keyframe ID")
plt.ylabel("Number of Map Points Culled")
plt.grid(True)
plt.legend()

fig.text(
    0.5, 0.02,
    "Interpretation: Peaks in culled map points correspond to visually degraded frames "
    "where features fail to remain stable. Reduced culling magnitude or frequency in "
    "the DIP pipeline indicates improved robustness and feature persistence.",
    ha="center",
    va="bottom",
    fontsize=10,
    wrap=True
)

plt.subplots_adjust(bottom=0.25)
savefig("plot_map_points_culled_comparison")

# -------------------------
# Rolling Average: Map Point Creation (Baseline vs DIP)
# -------------------------
WINDOW = 10  # tune: 5–20 usually works well

df_b["created_avg"] = df_b["created"].rolling(WINDOW, min_periods=1).mean()
df_d["created_avg"] = df_d["created"].rolling(WINDOW, min_periods=1).mean()

fig = plt.figure(figsize=(12,6))

plt.plot(
    df_b["kf_id"],
    df_b["created_avg"],
    label=f"Baseline – Created (Avg, {WINDOW})",
    linewidth=2.5,
    color="tab:green"
)

plt.plot(
    df_d["kf_id"],
    df_d["created_avg"],
    label=f"DIP – Created (Avg, {WINDOW})",
    linewidth=2.5,
    color="tab:green",
    linestyle="--"
)

plt.title("Rolling Average of Map Point Creation per Keyframe")
plt.xlabel("Keyframe ID")
plt.ylabel("Average Number of Map Points Created")
plt.grid(True)
plt.legend()

fig.text(
    0.5, 0.02,
    "Interpretation: The rolling average of created map points reflects the system’s "
    "ability to consistently identify stable new features. Higher or more stable values "
    "in the DIP pipeline indicate improved feature visibility and robustness.",
    ha="center",
    va="bottom",
    fontsize=10,
    wrap=True
)

plt.subplots_adjust(bottom=0.25)
savefig("plot_map_points_created_rolling_avg")

# -------------------------
# Rolling Average: Map Point Culling (Baseline vs DIP)
# -------------------------
df_b["culled_avg"] = df_b["culled"].rolling(WINDOW, min_periods=1).mean()
df_d["culled_avg"] = df_d["culled"].rolling(WINDOW, min_periods=1).mean()

fig = plt.figure(figsize=(12,6))

plt.plot(
    df_b["kf_id"],
    df_b["culled_avg"],
    label=f"Baseline – Culled (Avg, {WINDOW})",
    linewidth=2.5,
    color="tab:red"
)

plt.plot(
    df_d["kf_id"],
    df_d["culled_avg"],
    label=f"DIP – Culled (Avg, {WINDOW})",
    linewidth=2.5,
    color="tab:red",
    linestyle="--"
)

plt.title("Rolling Average of Map Point Culling per Keyframe")
plt.xlabel("Keyframe ID")
plt.ylabel("Average Number of Map Points Culled")
plt.grid(True)
plt.legend()

fig.text(
    0.5, 0.02,
    "Interpretation: The rolling average of culled map points highlights persistent "
    "tracking difficulty rather than isolated failures. Reduced culling trends in the "
    "DIP pipeline suggest improved feature stability across consecutive frames.",
    ha="center",
    va="bottom",
    fontsize=10,
    wrap=True
)

plt.subplots_adjust(bottom=0.25)
savefig("plot_map_points_culled_rolling_avg")



# -------------------------
# 3. Tracking State Timeline (Stacked)
# -------------------------
df_b, df_d = load_metrics("tracking_state")

fig, axes = plt.subplots(2, 1, figsize=(14,6), sharex=True)

def add_stats_box(ax, df, title_color="black"):
    # Count occurrences of each tracking state
    ok_count = (df["state"] == 2).sum()
    recent_lost_count = (df["state"] == 3).sum()
    lost_count = (df["state"] == 4).sum()
    
    text = (
        f"OK: {ok_count}\n"
        f"Recently Lost: {recent_lost_count}\n"
        f"LOST: {lost_count}"
    )

    # Add the annotation in bottom-right corner
    ax.text(
        0.98, 0.02, text,
        transform=ax.transAxes,
        fontsize=10,
        va='bottom',
        ha='right',
        bbox=dict(
            boxstyle="round,pad=0.4",
            fc="white",
            ec=title_color,
            alpha=0.8
        )
    )

# --- Baseline ---
axes[0].scatter(df_b["frame_id"], df_b["state"], c="blue", s=6)
axes[0].set_title("Tracking State – Baseline")
axes[0].set_yticks([-1,0,1,2,3,4,5])
axes[0].set_yticklabels(["SYS_NOT_RDY","NO_IMG_YET","NOT_INIT","OK", "RECENTLY_LOST", "LOST","OK_KLT"])
axes[0].grid(True)
add_stats_box(axes[0], df_b, title_color="blue")

# --- DIP ---
axes[1].scatter(df_d["frame_id"], df_d["state"], c="orange", s=6)
axes[1].set_title("Tracking State – DIP")
axes[1].set_yticks([-1,0,1,2,3,4,5])
axes[1].set_yticklabels(["SYS_NOT_RDY","NO_IMG_YET","NOT_INIT","OK", "RECENTLY_LOST", "LOST","OK_KLT"])
axes[1].set_xlabel("Frame ID")
axes[1].grid(True)
add_stats_box(axes[1], df_d, title_color="orange")

plt.tight_layout()
savefig("plot_tracking_state_stacked")

# -------------------------
# 4. Untracked Keypoints (Stacked Baseline + DIP)
# -------------------------
df_b, df_d = load_metrics("keypoints")

# Compute untracked keypoints
df_b["untracked"] = df_b["keypoints"] - df_b["tracked"]
df_d["untracked"] = df_d["keypoints"] - df_d["tracked"]

# Rolling window size (tune if needed)
WINDOW = 30

df_b["untracked_avg"] = df_b["untracked"].rolling(WINDOW, min_periods=1).mean()
df_d["untracked_avg"] = df_d["untracked"].rolling(WINDOW, min_periods=1).mean()

fig, axes = plt.subplots(2, 1, figsize=(12,7), sharex=True)

# --- Baseline ---
axes[0].plot(
    df_b["frame_id"],
    df_b["untracked"],
    color="tab:red",
    alpha=0.3,
    linewidth=1,
    label="Untracked (raw)"
)
axes[0].plot(
    df_b["frame_id"],
    df_b["untracked_avg"],
    color="black",
    linewidth=2.5,
    linestyle="--",
    label=f"Rolling Mean ({WINDOW} frames)"
)

axes[0].set_title("Baseline: Untracked Keypoints per Frame")
axes[0].set_ylabel("Count")
axes[0].legend()
axes[0].grid(True)

# --- DIP ---
axes[1].plot(
    df_d["frame_id"],
    df_d["untracked"],
    color="tab:purple",
    alpha=0.3,
    linewidth=1,
    label="Untracked (raw)"
)
axes[1].plot(
    df_d["frame_id"],
    df_d["untracked_avg"],
    color="black",
    linewidth=2.5,
    linestyle="--",
    label=f"Rolling Mean ({WINDOW} frames)"
)

axes[1].set_title("DIP: Untracked Keypoints per Frame")
axes[1].set_ylabel("Count")
axes[1].set_xlabel("Frame ID")
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
savefig("plot_untracked_keypoints_with_average")

# -------------------------
# 4c. Rolling Average Comparison: Untracked Keypoints (Baseline vs DIP)
# -------------------------

# Ensure both datasets are aligned in length
min_len = min(len(df_b), len(df_d))
df_b_aligned = df_b.iloc[:min_len].reset_index(drop=True)
df_d_aligned = df_d.iloc[:min_len].reset_index(drop=True)

plt.figure(figsize=(12,5))

plt.plot(
    df_b_aligned["frame_id"],
    df_b_aligned["untracked_avg"],
    label="Baseline (Rolling Avg)",
    linewidth=3,
    color="tab:red"
)

plt.plot(
    df_d_aligned["frame_id"],
    df_d_aligned["untracked_avg"],
    label="DIP (Rolling Avg)",
    linewidth=3,
    color="tab:green"
)

plt.title("Rolling Average Untracked Keypoints: Baseline vs DIP")
plt.xlabel("Frame ID")
plt.ylabel("Untracked Keypoints (Rolling Average)")
plt.grid(True)
plt.legend()

plt.tight_layout()
savefig("plot_untracked_keypoints_rolling_avg_comparison")


# -------------------------
# Keyframe Creation Frequency (4 Visualizations)
# -------------------------
df_b, df_d = load_metrics("keyframe_frequency")

def compute_kf_intervals(df):
    df = df.sort_values("kf_id").reset_index(drop=True)
    df["frame_delta"] = df["frame_id"].diff().fillna(0)
    if "timestamp" in df.columns:
        df["time_delta"] = df["timestamp"].diff().fillna(0)
    else:
        df["time_delta"] = 0
    return df

df_b = compute_kf_intervals(df_b)
df_d = compute_kf_intervals(df_d)

# Align lengths (important for fair comparison)
min_len = min(len(df_b), len(df_d))
df_b = df_b.iloc[:min_len]
df_d = df_d.iloc[:min_len]

# ============================================================
# 1️⃣ ORIGINAL PLOT — Frames Between Keyframes vs KF ID
# ============================================================
plt.figure(figsize=(10,5))
plt.plot(df_b["kf_id"], df_b["frame_delta"], label="Baseline", marker="o", alpha=0.7)
plt.plot(df_d["kf_id"], df_d["frame_delta"], label="DIP", marker="o", alpha=0.7)

plt.title("Keyframe Creation Frequency (Frames Between Keyframes)")
plt.xlabel("Keyframe ID")
plt.ylabel("Frames Between Keyframes")
plt.grid(True)
plt.legend()
plt.tight_layout()
savefig("plot_keyframe_frequency_raw")

# ============================================================
# 2️⃣ HISTOGRAM — Distribution of Keyframe Intervals
# ============================================================
plt.figure(figsize=(10,5))

plt.hist(
    df_b["frame_delta"],
    bins=40,
    alpha=0.6,
    density=True,
    label="Baseline",
    color="tab:red"
)

plt.hist(
    df_d["frame_delta"],
    bins=40,
    alpha=0.6,
    density=True,
    label="DIP",
    color="tab:green"
)

plt.xlabel("Frames Between Keyframes")
plt.ylabel("Density")
plt.title("Distribution of Keyframe Creation Intervals")
plt.grid(True)
plt.legend()
plt.tight_layout()
savefig("plot_keyframe_frequency_histogram")

# ============================================================
# 3️⃣ CDF — Probability of Early Keyframe Creation
# ============================================================
def plot_cdf(data, label):
    sorted_data = np.sort(data)
    y = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    plt.plot(sorted_data, y, label=label, linewidth=2)

plt.figure(figsize=(10,5))
plot_cdf(df_b["frame_delta"], "Baseline")
plot_cdf(df_d["frame_delta"], "DIP")

plt.xlabel("Frames Between Keyframes")
plt.ylabel("Cumulative Probability")
plt.title("CDF of Keyframe Creation Intervals")
plt.grid(True)
plt.legend()
plt.tight_layout()
savefig("plot_keyframe_frequency_cdf")

# ============================================================
# 4️⃣ SMOOTHED TREND — Rolling Average Stability
# ============================================================
WINDOW = 20  # tune if needed

df_b["frame_delta_avg"] = df_b["frame_delta"].rolling(WINDOW, min_periods=1).mean()
df_d["frame_delta_avg"] = df_d["frame_delta"].rolling(WINDOW, min_periods=1).mean()

plt.figure(figsize=(12,5))
plt.plot(
    df_b["kf_id"],
    df_b["frame_delta_avg"],
    label="Baseline",
    linewidth=2
)
plt.plot(
    df_d["kf_id"],
    df_d["frame_delta_avg"],
    label="DIP",
    linewidth=2
)

plt.xlabel("Keyframe ID")
plt.ylabel(f"Avg Frames Between Keyframes ({WINDOW}-KF window)")
plt.title("Smoothed Keyframe Creation Frequency")
plt.grid(True)
plt.legend()
plt.tight_layout()
savefig("plot_keyframe_frequency_smoothed")


# -------------------------
# Camera Pose Jump Magnitude
# -------------------------

def load_trajectory_txt(path):
    """
    Load ORB-SLAM3 trajectory in TUM format:
    timestamp tx ty tz qx qy qz qw
    """
    cols = ["timestamp","tx","ty","tz","qx","qy","qz","qw"]
    try:
        df = pl.read_csv(path, separator=" ", has_header=False, new_columns=cols)
    except:
        # fallback if PL can't auto-detect space separation
        df = pl.read_csv(path, has_header=False, new_columns=cols)
    return df.to_pandas()


def compute_pose_jumps(df):
    """
    Computes translation jump per frame:
    || position[i] - position[i-1] ||
    """
    pos = df[["tx","ty","tz"]].values
    # difference between consecutive positions
    diffs = np.diff(pos, axis=0)
    # Euclidean norm of each diff
    jumps = np.linalg.norm(diffs, axis=1)
    return jumps


# Load baseline and DIP trajectory text files
cam_base = load_trajectory_txt("baseline/CameraTrajectory.txt")
cam_dip  = load_trajectory_txt("dip/CameraTrajectory.txt")

# Compute pose jumps
jumps_base = compute_pose_jumps(cam_base)
jumps_dip  = compute_pose_jumps(cam_dip)

# --- Plot ---
plt.figure(figsize=(12,5))
plt.plot(jumps_base, label="Baseline Pose Jump Magnitude", linewidth=1.5)
plt.plot(jumps_dip,  label="DIP Pose Jump Magnitude", linewidth=1.5)

plt.title("Camera Pose Jump Magnitude Per Frame")
plt.xlabel("Frame Index")
plt.ylabel("Jump Magnitude (meters)")
plt.grid(True)
plt.legend()
plt.tight_layout()
savefig("plot_camera_pose_jump_magnitude")
plt.close()