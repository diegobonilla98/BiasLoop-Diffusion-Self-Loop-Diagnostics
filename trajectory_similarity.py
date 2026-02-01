import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
from sklearn.decomposition import PCA

embeddings_base = Path("embeddings_flux_dev")
output_base = Path("trajectory_analysis")
output_base.mkdir(parents=True, exist_ok=True)

model_names = ["siglip", "dino"]
eps = 1e-8
plt.style.use("seaborn-v0_8-whitegrid")


def trajectory_stats(emb):
    diffs = emb[1:] - emb[:-1]
    step_norms = np.linalg.norm(diffs, axis=1)
    unit_steps = diffs / (step_norms[:, None] + eps)
    total_length = step_norms.sum()
    displacement = emb[-1] - emb[0]
    displacement_norm = np.linalg.norm(displacement)
    avg_dir = unit_steps.mean(axis=0)
    avg_dir_norm = np.linalg.norm(avg_dir)
    mean_speed = step_norms.mean()
    speed_std = step_norms.std()
    tortuosity = total_length / (displacement_norm + eps)
    if unit_steps.shape[0] > 1:
        cos_turn = np.sum(unit_steps[:-1] * unit_steps[1:], axis=1)
        cos_turn = np.clip(cos_turn, -1.0, 1.0)
        mean_turn_angle = np.mean(np.arccos(cos_turn))
    else:
        mean_turn_angle = 0.0
    return {
        "n_steps": emb.shape[0],
        "total_length": float(total_length),
        "displacement_norm": float(displacement_norm),
        "mean_speed": float(mean_speed),
        "speed_std": float(speed_std),
        "tortuosity": float(tortuosity),
        "directional_consistency": float(avg_dir_norm),
        "mean_turn_angle": float(mean_turn_angle),
    }


def pairwise_metrics(emb_a, emb_b):
    n = min(emb_a.shape[0], emb_b.shape[0])
    a = emb_a[:n]
    b = emb_b[:n]
    diffs_a = a[1:] - a[:-1]
    diffs_b = b[1:] - b[:-1]
    norm_a = np.linalg.norm(diffs_a, axis=1)
    norm_b = np.linalg.norm(diffs_b, axis=1)
    unit_a = diffs_a / (norm_a[:, None] + eps)
    unit_b = diffs_b / (norm_b[:, None] + eps)
    step_cos = np.sum(unit_a * unit_b, axis=1)
    step_cos = np.clip(step_cos, -1.0, 1.0)
    step_cos_mean = float(np.mean(step_cos))

    disp_a = a[-1] - a[0]
    disp_b = b[-1] - b[0]
    disp_cos = float(np.dot(disp_a, disp_b) / ((np.linalg.norm(disp_a) + eps) * (np.linalg.norm(disp_b) + eps)))

    path_a = a - a[0]
    path_b = b - b[0]
    length_a = np.sum(norm_a)
    length_b = np.sum(norm_b)
    path_a = path_a / (length_a + eps)
    path_b = path_b / (length_b + eps)
    diff = path_a - path_b
    rmse = float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))
    shape_sim = float(1.0 / (1.0 + rmse))

    return {
        "step_cosine": step_cos_mean,
        "displacement_cosine": disp_cos,
        "shape_rmse": rmse,
        "shape_similarity": shape_sim,
    }


def per_step_alignment(trajectories):
    min_steps = min(t["emb"].shape[0] for t in trajectories)
    rows = []
    for step in range(min_steps - 1):
        unit_steps = []
        speeds = []
        for t in trajectories:
            emb = t["emb"]
            diff = emb[step + 1] - emb[step]
            norm = np.linalg.norm(diff)
            speeds.append(norm)
            unit_steps.append(diff / (norm + eps))
        unit_steps = np.array(unit_steps)
        mean_dir = unit_steps.mean(axis=0)
        alignment = float(np.linalg.norm(mean_dir))
        mean_speed = float(np.mean(speeds))
        rows.append({
            "step": step,
            "alignment": alignment,
            "mean_speed": mean_speed,
            "num_trajectories": len(trajectories)
        })
    return rows


def get_speed_profile(emb):
    diffs = emb[1:] - emb[:-1]
    return np.linalg.norm(diffs, axis=1)


def plot_similarity_heatmap(pair_df, metric, strength, images, out_path, title):
    n = len(images)
    matrix = np.eye(n)
    img_idx = {img: i for i, img in enumerate(images)}
    subset = pair_df[(pair_df["strength"] == strength) | (pair_df["strength"] == "all" if strength == "all" else False)]
    if strength != "all":
        subset = pair_df[pair_df["strength"] == strength]
    for _, row in subset.iterrows():
        i, j = img_idx.get(row["image_a"]), img_idx.get(row["image_b"])
        if i is not None and j is not None:
            matrix[i, j] = row[metric]
            matrix[j, i] = row[metric]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(matrix, annot=True, fmt=".2f", cmap="RdYlGn", vmin=-1 if "cosine" in metric else 0, 
                vmax=1, xticklabels=images, yticklabels=images, ax=ax, square=True,
                cbar_kws={"label": metric.replace("_", " ").title()})
    ax.set_title(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_step_alignment_over_time(strengths_data, out_path, model_name):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(strengths_data)))
    for (strength, data), color in zip(strengths_data.items(), colors):
        steps = [r["step"] for r in data]
        alignment = [r["alignment"] for r in data]
        speed = [r["mean_speed"] for r in data]
        axes[0].plot(steps, alignment, label=f"s={strength}", color=color, linewidth=2)
        axes[1].plot(steps, speed, label=f"s={strength}", color=color, linewidth=2)
    
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Direction Alignment (0=random, 1=identical)")
    axes[0].set_title(f"{model_name.upper()} - Cross-Image Direction Alignment Over Steps")
    axes[0].legend()
    axes[0].set_ylim(0, 1.05)
    
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Mean Speed (embedding distance)")
    axes[1].set_title(f"{model_name.upper()} - Mean Speed Over Steps")
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_speed_profiles(trajectories, strength, out_path, model_name):
    group = [t for t in trajectories if t["strength_val"] == strength]
    if len(group) == 0:
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = plt.cm.tab20(np.linspace(0, 1, len(group)))
    
    all_speeds = []
    for t, color in zip(group, colors):
        speeds = get_speed_profile(t["emb"])
        all_speeds.append(speeds)
        axes[0].plot(speeds, label=t["image"], color=color, alpha=0.7, linewidth=1.5)
    
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Speed (embedding distance per step)")
    axes[0].set_title(f"{model_name.upper()} - Individual Speed Profiles (s={strength})")
    axes[0].legend(loc="upper right", fontsize=8, ncol=2)
    
    min_len = min(len(s) for s in all_speeds)
    all_speeds = np.array([s[:min_len] for s in all_speeds])
    mean_speed = all_speeds.mean(axis=0)
    std_speed = all_speeds.std(axis=0)
    steps = np.arange(min_len)
    
    axes[1].plot(steps, mean_speed, color="blue", linewidth=2, label="Mean")
    axes[1].fill_between(steps, mean_speed - std_speed, mean_speed + std_speed, alpha=0.3, color="blue", label="±1 Std")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Speed")
    axes[1].set_title(f"{model_name.upper()} - Aggregated Speed Profile (s={strength})")
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_trajectory_stats_comparison(traj_df, out_path, model_name):
    metrics = ["tortuosity", "directional_consistency", "mean_speed", "displacement_norm"]
    strengths = sorted(traj_df["strength"].unique())
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for ax, metric in zip(axes, metrics):
        data = [traj_df[traj_df["strength"] == s][metric].values for s in strengths]
        positions = np.arange(len(strengths))
        bp = ax.boxplot(data, positions=positions, widths=0.6, patch_artist=True)
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(strengths)))
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_xticks(positions)
        ax.set_xticklabels([f"s={s}" for s in strengths])
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_title(f"{model_name.upper()} - {metric.replace('_', ' ').title()} by Strength")
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_summary_bars(summary_df, out_path, model_name):
    metrics = ["step_cosine_mean", "displacement_cosine_mean", "shape_similarity_mean"]
    labels = ["Step Direction\nSimilarity", "Displacement\nSimilarity", "Shape\nSimilarity"]
    
    numeric_strengths = summary_df[summary_df["strength"] != "all"]
    strengths = numeric_strengths["strength"].tolist()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(strengths))
    width = 0.25
    colors = ["#2ecc71", "#3498db", "#9b59b6"]
    
    for i, (metric, label, color) in enumerate(zip(metrics, labels, colors)):
        values = numeric_strengths[metric].values
        errors = numeric_strengths[metric.replace("_mean", "_std")].values
        ax.bar(x + i * width, values, width, label=label, color=color, alpha=0.8, yerr=errors, capsize=4)
    
    ax.set_xlabel("Strength")
    ax.set_ylabel("Similarity Score")
    ax.set_title(f"{model_name.upper()} - Path Similarity Metrics by Strength\n(Higher = More Similar Deformation Across Images)")
    ax.set_xticks(x + width)
    ax.set_xticklabels([f"s={s}" for s in strengths])
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_normalized_trajectories_overlay(trajectories, strength, out_path, model_name):
    group = [t for t in trajectories if t["strength_val"] == strength]
    if len(group) == 0:
        return
    
    all_emb = np.vstack([t["emb"] for t in group])
    pca = PCA(n_components=2, random_state=42)
    pca.fit(all_emb)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    colors = plt.cm.tab20(np.linspace(0, 1, len(group)))
    
    for t, color in zip(group, colors):
        proj = pca.transform(t["emb"])
        axes[0].plot(proj[:, 0], proj[:, 1], color=color, alpha=0.7, linewidth=1.5, label=t["image"])
        axes[0].scatter(proj[0, 0], proj[0, 1], color=color, marker="o", s=80, edgecolors="black", zorder=5)
        axes[0].scatter(proj[-1, 0], proj[-1, 1], color=color, marker="X", s=80, edgecolors="black", zorder=5)
    
    axes[0].set_xlabel("PC1")
    axes[0].set_ylabel("PC2")
    axes[0].set_title(f"{model_name.upper()} - Raw Trajectories in PCA Space (s={strength})")
    axes[0].legend(loc="upper right", fontsize=8, ncol=2)
    
    for t, color in zip(group, colors):
        emb = t["emb"]
        centered = emb - emb[0]
        diffs = emb[1:] - emb[:-1]
        total_len = np.sum(np.linalg.norm(diffs, axis=1))
        normalized = centered / (total_len + eps)
        proj = pca.transform(normalized + emb[0])
        proj = proj - proj[0]
        axes[1].plot(proj[:, 0], proj[:, 1], color=color, alpha=0.7, linewidth=1.5, label=t["image"])
        axes[1].scatter(0, 0, color=color, marker="o", s=80, edgecolors="black", zorder=5)
        axes[1].scatter(proj[-1, 0], proj[-1, 1], color=color, marker="X", s=80, edgecolors="black", zorder=5)
    
    axes[1].set_xlabel("PC1 (normalized)")
    axes[1].set_ylabel("PC2 (normalized)")
    axes[1].set_title(f"{model_name.upper()} - Normalized Trajectories Overlay (s={strength})\n(Start aligned, length normalized)")
    axes[1].legend(loc="upper right", fontsize=8, ncol=2)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_trajectory_clustering(pair_df, strength, images, out_path, model_name):
    n = len(images)
    if n < 3:
        return
    
    dist_matrix = np.zeros((n, n))
    img_idx = {img: i for i, img in enumerate(images)}
    
    subset = pair_df[pair_df["strength"] == strength]
    for _, row in subset.iterrows():
        i, j = img_idx.get(row["image_a"]), img_idx.get(row["image_b"])
        if i is not None and j is not None:
            dist = row["shape_rmse"]
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist
    
    condensed = squareform(dist_matrix)
    Z = linkage(condensed, method="ward")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    dendrogram(Z, labels=images, ax=ax, leaf_rotation=45)
    ax.set_ylabel("Distance (Shape RMSE)")
    ax.set_title(f"{model_name.upper()} - Trajectory Shape Clustering (s={strength})\n(Similar shapes cluster together)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_cumulative_displacement(trajectories, strength, out_path, model_name):
    group = [t for t in trajectories if t["strength_val"] == strength]
    if len(group) == 0:
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.tab20(np.linspace(0, 1, len(group)))
    
    for t, color in zip(group, colors):
        emb = t["emb"]
        displacements = np.linalg.norm(emb - emb[0], axis=1)
        ax.plot(displacements, color=color, linewidth=1.5, alpha=0.8, label=t["image"])
    
    ax.set_xlabel("Step")
    ax.set_ylabel("Distance from Start (embedding space)")
    ax.set_title(f"{model_name.upper()} - Cumulative Displacement from Origin (s={strength})")
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


for model_name in model_names:
    trajectory_dirs = sorted([d for d in embeddings_base.rglob("strength_*") if d.is_dir()])
    trajectories = []

    for traj_dir in trajectory_dirs:
        emb_path = traj_dir / f"{model_name}_embeddings.npy"
        if not emb_path.exists():
            continue
        emb = np.load(emb_path)
        img_name = traj_dir.parent.name
        strength_str = traj_dir.name.replace("strength_", "")
        strength_val = float(strength_str)
        trajectories.append({
            "image": img_name,
            "strength_str": strength_str,
            "strength_val": strength_val,
            "emb": emb
        })

    if len(trajectories) == 0:
        print(f"No embeddings found for {model_name}")
        continue

    model_out = output_base / model_name
    model_out.mkdir(parents=True, exist_ok=True)

    traj_rows = []
    for t in trajectories:
        stats = trajectory_stats(t["emb"])
        traj_rows.append({
            "image": t["image"],
            "strength": t["strength_val"],
            **stats
        })
    traj_df = pd.DataFrame(traj_rows)
    traj_df.to_csv(model_out / "trajectory_stats.csv", index=False)

    pair_rows = []
    strengths = sorted(set(t["strength_val"] for t in trajectories))
    for s in strengths:
        group = [t for t in trajectories if t["strength_val"] == s]
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                metrics = pairwise_metrics(group[i]["emb"], group[j]["emb"])
                pair_rows.append({
                    "strength": s,
                    "image_a": group[i]["image"],
                    "image_b": group[j]["image"],
                    **metrics
                })
        step_rows = per_step_alignment(group)
        pd.DataFrame(step_rows).to_csv(model_out / f"strength_{s}_step_alignment.csv", index=False)

    for i in range(len(trajectories)):
        for j in range(i + 1, len(trajectories)):
            metrics = pairwise_metrics(trajectories[i]["emb"], trajectories[j]["emb"])
            pair_rows.append({
                "strength": "all",
                "image_a": trajectories[i]["image"],
                "image_b": trajectories[j]["image"],
                **metrics
            })

    pair_df = pd.DataFrame(pair_rows)
    pair_df.to_csv(model_out / "pairwise_similarity.csv", index=False)

    summary_rows = []
    for s in list(strengths) + ["all"]:
        subset = pair_df[pair_df["strength"] == s]
        if len(subset) == 0:
            continue
        summary_rows.append({
            "strength": s,
            "step_cosine_mean": float(subset["step_cosine"].mean()),
            "step_cosine_std": float(subset["step_cosine"].std()),
            "displacement_cosine_mean": float(subset["displacement_cosine"].mean()),
            "displacement_cosine_std": float(subset["displacement_cosine"].std()),
            "shape_rmse_mean": float(subset["shape_rmse"].mean()),
            "shape_rmse_std": float(subset["shape_rmse"].std()),
            "shape_similarity_mean": float(subset["shape_similarity"].mean()),
            "shape_similarity_std": float(subset["shape_similarity"].std()),
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(model_out / "summary_by_strength.csv", index=False)

    plots_dir = model_out / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    unique_images = sorted(set(t["image"] for t in trajectories))
    
    plot_trajectory_stats_comparison(traj_df, plots_dir / "trajectory_stats_comparison.png", model_name)
    plot_summary_bars(summary_df, plots_dir / "similarity_summary.png", model_name)
    
    strengths_alignment_data = {}
    for s in strengths:
        group = [t for t in trajectories if t["strength_val"] == s]
        strengths_alignment_data[s] = per_step_alignment(group)
    plot_step_alignment_over_time(strengths_alignment_data, plots_dir / "step_alignment_over_time.png", model_name)
    
    for s in strengths:
        s_str = str(s).replace(".", "_")
        strength_plots = plots_dir / f"strength_{s_str}"
        strength_plots.mkdir(parents=True, exist_ok=True)
        
        strength_images = sorted(set(t["image"] for t in trajectories if t["strength_val"] == s))
        
        plot_speed_profiles(trajectories, s, strength_plots / "speed_profiles.png", model_name)
        plot_normalized_trajectories_overlay(trajectories, s, strength_plots / "trajectories_overlay.png", model_name)
        plot_cumulative_displacement(trajectories, s, strength_plots / "cumulative_displacement.png", model_name)
        
        plot_similarity_heatmap(pair_df, "step_cosine", s, strength_images, 
                               strength_plots / "heatmap_step_cosine.png",
                               f"{model_name.upper()} - Step Direction Similarity (s={s})")
        plot_similarity_heatmap(pair_df, "displacement_cosine", s, strength_images,
                               strength_plots / "heatmap_displacement_cosine.png", 
                               f"{model_name.upper()} - Displacement Similarity (s={s})")
        plot_similarity_heatmap(pair_df, "shape_similarity", s, strength_images,
                               strength_plots / "heatmap_shape_similarity.png",
                               f"{model_name.upper()} - Shape Similarity (s={s})")
        
        plot_trajectory_clustering(pair_df, s, strength_images, strength_plots / "trajectory_clustering.png", model_name)

    print(f"Saved analysis and plots for {model_name} to {model_out}")

print(f"\nAll analysis saved to {output_base}")
