import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel, AutoImageProcessor, pipeline

device = "cuda" if torch.cuda.is_available() else "cpu"

siglip_ckpt = "google/siglip2-base-patch16-512"
siglip_model = AutoModel.from_pretrained(siglip_ckpt, device_map="auto").eval()
siglip_processor = AutoImageProcessor.from_pretrained(siglip_ckpt)

dino_extractor = pipeline(
    model="facebook/dinov3-vitl16-pretrain-lvd1689m",
    task="image-feature-extraction",
    device=device
)

input_base = Path("output_trajectories_flux_dev")
output_embeddings_base = Path("embeddings_flux_dev")

all_records = []

trajectory_dirs = sorted([d for d in input_base.rglob("strength_*") if d.is_dir()])

for traj_dir in tqdm(trajectory_dirs, desc="Trajectories"):
    img_name = traj_dir.parent.name
    strength_name = traj_dir.name
    
    config_path = traj_dir / "config.json"
    config = {}
    if config_path.exists():
        with open(config_path, "r") as f:
            config = json.load(f)
    
    emb_dir = output_embeddings_base / img_name / strength_name
    emb_dir.mkdir(parents=True, exist_ok=True)
    
    image_files = sorted(traj_dir.glob("step_*.png"))
    
    siglip_embeddings = []
    dino_embeddings = []
    
    for img_path in tqdm(image_files, desc=f"  {img_name}/{strength_name}", leave=False):
        step_name = img_path.stem
        image = Image.open(img_path).convert("RGB")
        
        siglip_inputs = siglip_processor(images=[image], return_tensors="pt").to(siglip_model.device)
        with torch.no_grad():
            siglip_output = siglip_model.vision_model(**siglip_inputs)
            siglip_emb = siglip_output.pooler_output.cpu().numpy()
        siglip_embeddings.append(siglip_emb[0])
        
        dino_features = dino_extractor(image)
        dino_emb = np.array(dino_features[0])
        if dino_emb.ndim > 1:
            dino_emb = dino_emb.mean(axis=0)
        dino_embeddings.append(dino_emb)
        
        step_num = int(step_name.replace("step_", ""))
        record = {
            "image_name": img_name,
            "strength": config.get("strength", ""),
            "step": step_num,
            "image_path": str(img_path),
            "siglip_emb_path": str(emb_dir / "siglip_embeddings.npy"),
            "dino_emb_path": str(emb_dir / "dino_embeddings.npy"),
            "num_inference_steps": config.get("num_inference_steps", ""),
            "seed": config.get("seed", ""),
            "width": config.get("width", ""),
            "height": config.get("height", ""),
            "model": config.get("model", ""),
            "vae": config.get("vae", ""),
            "guidance_scale": config.get("guidance_scale", ""),
            "N": config.get("N", ""),
            "input_image": config.get("input_image", ""),
            "timestamp": config.get("timestamp", "")
        }
        all_records.append(record)
    
    siglip_arr = np.stack(siglip_embeddings, axis=0)
    dino_arr = np.stack(dino_embeddings, axis=0)
    
    np.save(emb_dir / "siglip_embeddings.npy", siglip_arr)
    np.save(emb_dir / "dino_embeddings.npy", dino_arr)
    
    with open(emb_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

df = pd.DataFrame(all_records)
df.to_csv(output_embeddings_base / "all_embeddings_metadata.csv", index=False)

print(f"\nDone! Saved {len(all_records)} records to {output_embeddings_base / 'all_embeddings_metadata.csv'}")
print(f"Embeddings saved to {output_embeddings_base}")
