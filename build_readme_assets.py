import shutil
from pathlib import Path
from PIL import Image

source_root = Path(r"F:\ImageGenerationTrajectories")
output_dir = Path("readme_assets")
output_dir.mkdir(parents=True, exist_ok=True)

plots_flux = source_root / "plots_flux_dev"
analysis_siglip = source_root / "trajectory_analysis_latent" / "siglip" / "plots"
steps_root = source_root / "output_trajectories_flux_dev"

copy_items = [
    (plots_flux / "siglip_umap_2d_all.png", "siglip_umap_2d_all.png"),
    (plots_flux / "dino_umap_2d_all.png", "dino_umap_2d_all.png"),
    (analysis_siglip / "similarity_summary.png", "similarity_summary_siglip.png"),
    (analysis_siglip / "step_alignment_over_time.png", "step_alignment_over_time_siglip.png"),
    (analysis_siglip / "strength_0_25" / "heatmap_shape_similarity.png", "heatmap_shape_similarity_s025_siglip.png"),
    (analysis_siglip / "strength_0_25" / "trajectories_overlay.png", "trajectories_overlay_s025_siglip.png"),
    (analysis_siglip / "strength_0_25" / "trajectory_clustering.png", "trajectory_clustering_s025_siglip.png"),
    (analysis_siglip / "strength_0_25" / "cumulative_displacement.png", "cumulative_displacement_s025_siglip.png"),
]


def copy_image(src, dst_name):
    if not src.exists():
        return
    shutil.copy2(src, output_dir / dst_name)


def build_step_grid(step_paths, out_path, thumb_size=(256, 256), cols=4, padding=10, bg=(18, 18, 18)):
    images = [Image.open(p).convert("RGB").resize(thumb_size, Image.LANCZOS) for p in step_paths]
    rows = (len(images) + cols - 1) // cols
    w = cols * thumb_size[0] + (cols - 1) * padding
    h = rows * thumb_size[1] + (rows - 1) * padding
    canvas = Image.new("RGB", (w, h), bg)
    for idx, img in enumerate(images):
        r = idx // cols
        c = idx % cols
        x = c * (thumb_size[0] + padding)
        y = r * (thumb_size[1] + padding)
        canvas.paste(img, (x, y))
    canvas.save(out_path)


def step_paths_for(image_id, strength_str, steps):
    base = steps_root / image_id / f"strength_{strength_str}"
    return [base / f"step_{s:03d}.png" for s in steps]


for src, dst_name in copy_items:
    copy_image(src, dst_name)

steps = [0, 1, 2, 5, 10, 15, 20, 25]
grid_a = step_paths_for("00", "0.25", steps)
grid_b = step_paths_for("00", "0.5", steps)

build_step_grid(grid_a, output_dir / "trajectory_steps_00_s025.png")
build_step_grid(grid_b, output_dir / "trajectory_steps_00_s05.png")

print(f"Saved README assets to {output_dir}")
