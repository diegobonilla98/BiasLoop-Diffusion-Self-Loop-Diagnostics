# BiasLoop: Diffusion Self-Loop Diagnostics

This repo diagnoses and amplifies diffusion model biases by creating a self-loop over the same image at different strengths, then analyzing how trajectories deform across style, content, and abstract structure using embeddings. The goal is to measure whether a model pushes images in a consistent direction, and how that direction changes with loop strength.

## Core Idea
- Loop an image through the same model N times
- Use different strengths to modulate how hard the model overwrites the input
- Embed every step with SigLip2 and DINOv3
- Compare trajectory paths across images to see if deformation is consistent

## Visual Overview

Self-loop snapshots showing 4 image examples across timesteps 0, 5, 10, 25 (FLUX, strength 0.5):

![Trajectory steps s=0.5](readme_assets/trajectory_steps_00_s05.png)

Global trajectory maps (all images, UMAP 2D):

![SigLip UMAP 2D](readme_assets/siglip_umap_2d_all.png)
![DINO UMAP 2D](readme_assets/dino_umap_2d_all.png)

Similarity and consistency plots:

![Similarity summary](readme_assets/similarity_summary_siglip.png)
![Step alignment](readme_assets/step_alignment_over_time_siglip.png)
![Shape similarity heatmap](readme_assets/heatmap_shape_similarity_s025_siglip.png)
![Trajectory overlay](readme_assets/trajectories_overlay_s025_siglip.png)
![Trajectory clustering](readme_assets/trajectory_clustering_s025_siglip.png)
![Cumulative displacement](readme_assets/cumulative_displacement_s025_siglip.png)

## Results
The trajectories show a consistent but weak directional bias across images, and the bias weakens as strength increases. Across SigLip2 embeddings, mean step-direction similarity drops from 0.108 at strength 0.15 to 0.036 at 0.5, while net displacement similarity drops from 0.429 to 0.267. DINOv3 shows the same pattern: step-direction similarity falls from 0.091 to 0.030, and displacement similarity from 0.435 to 0.259. This indicates that higher strength produces more divergent step-by-step motion across inputs, even if the overall deformation still trends in roughly similar directions.

Despite the divergence in step directions, the normalized path shapes remain relatively similar across images. Shape similarity stays high in both encoders (SigLip2: ~0.757–0.760, DINOv3: ~0.746–0.771), suggesting the loop induces comparable trajectory geometries even when exact directions vary. The heatmaps and clustering confirm this: images group by similar path shapes, but alignment is not uniform across all inputs.

In short, the self-loop reveals a measurable global bias: trajectories are not random, but they are only partially aligned. Stronger loops increase deformation magnitude and reduce cross-image agreement, while preserving a shared geometric pattern in the embedding path. This supports using embedding-space trajectory analysis as a diagnostic for systematic diffusion bias.

## Pipeline
1. Generate trajectories (FLUX or Qwen)
2. Extract embeddings for each step
3. Project trajectories into 2D/3D
4. Quantify path similarity and deformation consistency

## Folder Structure
```
input_images/
output_trajectories_flux_dev/
output_trajectories_qwen/
embeddings_flux_dev/
plots_flux_dev/
trajectory_analysis/
readme_assets/
```

## Setup
```
pip install diffusers transformers torch torchvision pillow tqdm pandas numpy umap-learn matplotlib seaborn scipy scikit-learn
```

## 1) Generate FLUX Trajectories
Script: `flux_trajectory.py`

Run:
```
python flux_trajectory.py
```

## 2) Generate Qwen Trajectories
Script: `qwen_trajectory.py`

Run:
```
python qwen_trajectory.py
```

## 3) Extract Embeddings
Script: `extract_embeddings.py`

Run:
```
python extract_embeddings.py
```

## 4) Visualize Trajectories
Script: `visualize_trajectories.py`

Run:
```
python visualize_trajectories.py
```

## 5) Trajectory Similarity + Plots
Script: `trajectory_similarity.py`

Run:
```
python trajectory_similarity.py
```

## 6) Build README Assets
Script: `build_readme_assets.py`

This pulls selected plots and builds step grids from the images stored at `F:\ImageGenerationTrajectories` and saves them to `readme_assets/`.

Run:
```
python build_readme_assets.py
```

## Notes
- `input_images/` must contain `.jpg` files
- Adjust `N`, `num_inference_steps`, `strengths`, and image size in each script
- Embeddings and analysis are generated for FLUX trajectories by default

## Quick Start
```
python flux_trajectory.py
python extract_embeddings.py
python visualize_trajectories.py
python trajectory_similarity.py
python build_readme_assets.py
```
