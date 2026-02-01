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


def build_multi_image_grid(image_ids, strength_str, timesteps, out_path, thumb_size=(256, 256), padding=10, bg=(18, 18, 18)):
    num_images = len(image_ids)
    num_timesteps = len(timesteps)
    
    w = num_timesteps * thumb_size[0] + (num_timesteps - 1) * padding
    h = num_images * thumb_size[1] + (num_images - 1) * padding
    canvas = Image.new("RGB", (w, h), bg)
    
    for img_idx, image_id in enumerate(image_ids):
        base = steps_root / image_id / f"strength_{strength_str}"
        for ts_idx, timestep in enumerate(timesteps):
            img_path = base / f"step_{timestep:03d}.png"
            if img_path.exists():
                img = Image.open(img_path).convert("RGB").resize(thumb_size, Image.LANCZOS)
                x = ts_idx * (thumb_size[0] + padding)
                y = img_idx * (thumb_size[1] + padding)
                canvas.paste(img, (x, y))
    
    canvas.save(out_path)


for src, dst_name in copy_items:
    copy_image(src, dst_name)

image_examples = ["00", "01", "02", "03"]
timesteps = [0, 5, 10, 25]

build_multi_image_grid(image_examples, "0.5", timesteps, output_dir / "trajectory_steps_00_s05.png")

print(f"Saved README assets to {output_dir}")
