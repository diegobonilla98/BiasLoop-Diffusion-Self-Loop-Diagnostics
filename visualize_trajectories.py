import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from umap import UMAP

embeddings_base = Path("embeddings_flux_dev")
plots_output = Path("plots_flux_dev") / "trajectories"
plots_output.mkdir(parents=True, exist_ok=True)

model_names = ["siglip", "dino"]

for model_name in model_names:
    trajectory_dirs = sorted([d for d in embeddings_base.rglob("strength_*") if d.is_dir()])
    trajectories = []
    all_embeddings = []

    for traj_dir in trajectory_dirs:
        emb_path = traj_dir / f"{model_name}_embeddings.npy"
        if not emb_path.exists():
            continue

        embeddings = np.load(emb_path)
        img_name = traj_dir.parent.name
        strength_str = traj_dir.name.replace("strength_", "")
        strength_val = float(strength_str)
        start_idx = len(all_embeddings)
        all_embeddings.extend(embeddings)

        trajectories.append({
            "image": img_name,
            "strength_str": strength_str,
            "strength_val": strength_val,
            "start_idx": start_idx,
            "end_idx": start_idx + embeddings.shape[0]
        })

    if len(all_embeddings) == 0:
        print(f"No embeddings found for {model_name}")
        continue

    all_embeddings = np.array(all_embeddings)

    umap_2d = UMAP(n_components=2, n_neighbors=15, min_dist=0.1, metric="cosine", random_state=42)
    umap_3d = UMAP(n_components=3, n_neighbors=15, min_dist=0.1, metric="cosine", random_state=42)

    proj_2d = umap_2d.fit_transform(all_embeddings)
    proj_3d = umap_3d.fit_transform(all_embeddings)

    unique_images = sorted(set(t["image"] for t in trajectories))
    unique_strengths = sorted(set(t["strength_val"] for t in trajectories))
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_images)))
    image_color = {img: colors[i] for i, img in enumerate(unique_images)}

    def traj_coords(traj, proj):
        return proj[traj["start_idx"]:traj["end_idx"]]

    for strength_val in unique_strengths:
        strength_trajs = [t for t in trajectories if t["strength_val"] == strength_val]
        if len(strength_trajs) == 0:
            continue

        strength_str = strength_trajs[0]["strength_str"]
        out_dir = plots_output / model_name / f"strength_{strength_str}"
        out_dir.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(12, 9))
        for traj in strength_trajs:
            coords = traj_coords(traj, proj_2d)
            color = image_color[traj["image"]]
            ax.plot(coords[:, 0], coords[:, 1], color=color, linewidth=1.6, alpha=0.85)
            ax.scatter(coords[0, 0], coords[0, 1], color=color, marker="o", s=90, edgecolors="black", linewidths=1.2, zorder=5)
            ax.scatter(coords[-1, 0], coords[-1, 1], color=color, marker="X", s=90, edgecolors="black", linewidths=1.2, zorder=5)

        legend_elements = [plt.Line2D([0], [0], color=image_color[i], linewidth=2, label=f"Image {i}") for i in unique_images]
        legend_elements.append(plt.Line2D([0], [0], marker="o", color="gray", linestyle="", markersize=9, label="Start"))
        legend_elements.append(plt.Line2D([0], [0], marker="X", color="gray", linestyle="", markersize=9, label="End"))
        ax.legend(handles=legend_elements, loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=9)
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        ax.set_title(f"{model_name.upper()} UMAP 2D - Strength {strength_str}")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / "umap_2d_global.png", dpi=150, bbox_inches="tight")
        plt.close()

        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection="3d")
        for traj in strength_trajs:
            coords = traj_coords(traj, proj_3d)
            color = image_color[traj["image"]]
            ax.plot(coords[:, 0], coords[:, 1], coords[:, 2], color=color, linewidth=1.6, alpha=0.85)
            ax.scatter(coords[0, 0], coords[0, 1], coords[0, 2], color=color, marker="o", s=90, edgecolors="black", linewidths=1.2)
            ax.scatter(coords[-1, 0], coords[-1, 1], coords[-1, 2], color=color, marker="X", s=90, edgecolors="black", linewidths=1.2)

        legend_elements = [plt.Line2D([0], [0], color=image_color[i], linewidth=2, label=f"Image {i}") for i in unique_images]
        legend_elements.append(plt.Line2D([0], [0], marker="o", color="gray", linestyle="", markersize=9, label="Start"))
        legend_elements.append(plt.Line2D([0], [0], marker="X", color="gray", linestyle="", markersize=9, label="End"))
        ax.legend(handles=legend_elements, loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=9)
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        ax.set_zlabel("Component 3")
        ax.set_title(f"{model_name.upper()} UMAP 3D - Strength {strength_str}")
        plt.tight_layout()
        plt.savefig(out_dir / "umap_3d_global.png", dpi=150, bbox_inches="tight")
        plt.close()

        for traj in strength_trajs:
            img = traj["image"]
            color = image_color[img]
            coords2d = traj_coords(traj, proj_2d)
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.plot(coords2d[:, 0], coords2d[:, 1], color=color, linewidth=2, alpha=0.9)
            ax.scatter(coords2d[0, 0], coords2d[0, 1], color=color, marker="o", s=120, edgecolors="black", linewidths=1.4, zorder=5)
            ax.scatter(coords2d[-1, 0], coords2d[-1, 1], color=color, marker="X", s=120, edgecolors="black", linewidths=1.4, zorder=5)
            ax.set_xlabel("Component 1")
            ax.set_ylabel("Component 2")
            ax.set_title(f"{model_name.upper()} UMAP 2D - {img} - Strength {strength_str}")
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(out_dir / f"umap_2d_{img}.png", dpi=150, bbox_inches="tight")
            plt.close()

            coords3d = traj_coords(traj, proj_3d)
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection="3d")
            ax.plot(coords3d[:, 0], coords3d[:, 1], coords3d[:, 2], color=color, linewidth=2, alpha=0.9)
            ax.scatter(coords3d[0, 0], coords3d[0, 1], coords3d[0, 2], color=color, marker="o", s=120, edgecolors="black", linewidths=1.4)
            ax.scatter(coords3d[-1, 0], coords3d[-1, 1], coords3d[-1, 2], color=color, marker="X", s=120, edgecolors="black", linewidths=1.4)
            ax.set_xlabel("Component 1")
            ax.set_ylabel("Component 2")
            ax.set_zlabel("Component 3")
            ax.set_title(f"{model_name.upper()} UMAP 3D - {img} - Strength {strength_str}")
            plt.tight_layout()
            plt.savefig(out_dir / f"umap_3d_{img}.png", dpi=150, bbox_inches="tight")
            plt.close()

print(f"\nAll plots saved to {plots_output}")
