import json
import torch
from datetime import datetime
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from diffusers import FluxImg2ImgPipeline
from diffusers.image_processor import VaeImageProcessor

dtype = torch.bfloat16
device = "cuda" if torch.cuda.is_available() else "cpu"

pipe = FluxImg2ImgPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=dtype
).to(device)
pipe.enable_attention_slicing()
pipe.vae.enable_slicing()
pipe.vae.enable_tiling()
torch.cuda.empty_cache()

vae_scale_factor = 2 ** (len(pipe.vae.config.block_out_channels) - 1) if hasattr(pipe.vae.config, 'block_out_channels') else 8
image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor * 2)

@torch.inference_mode()
def encode_image_to_latents(image, vae, dtype, device):
    image_tensor = image_processor.preprocess(image)
    image_tensor = image_tensor.to(device=device, dtype=dtype)
    latent_dist = vae.encode(image_tensor).latent_dist
    latents = latent_dist.sample()
    latents = (latents - vae.config.shift_factor) * vae.config.scaling_factor
    return latents

@torch.inference_mode()
def decode_latents_to_image(latents, vae, dtype):
    latents_unscaled = (latents / vae.config.scaling_factor) + vae.config.shift_factor
    image = vae.decode(latents_unscaled, return_dict=False)[0]
    image = image_processor.postprocess(image, output_type="pil")[0]
    return image

@torch.inference_mode()
def run_flux_denoising_on_latents(pipe, latents, strength, num_inference_steps, generator, height, width):
    batch_size = latents.shape[0]
    
    prompt_embeds, pooled_prompt_embeds, text_ids = pipe.encode_prompt(
        prompt="",
        prompt_2=None,
        device=device,
        num_images_per_prompt=1,
    )
    
    num_channels_latents = pipe.transformer.config.in_channels // 4
    vae_latent_height = 2 * (height // (vae_scale_factor * 2))
    vae_latent_width = 2 * (width // (vae_scale_factor * 2))
    packed_height = vae_latent_height // 2
    packed_width = vae_latent_width // 2
    
    latent_image_ids = pipe._prepare_latent_image_ids(
        batch_size, packed_height, packed_width, device, dtype
    )
    
    image_seq_len = packed_height * packed_width
    base_image_seq_len = pipe.scheduler.config.get("base_image_seq_len", 256)
    max_image_seq_len = pipe.scheduler.config.get("max_image_seq_len", 4096)
    base_shift = pipe.scheduler.config.get("base_shift", 0.5)
    max_shift = pipe.scheduler.config.get("max_shift", 1.15)
    m = (max_shift - base_shift) / (max_image_seq_len - base_image_seq_len)
    b = base_shift - m * base_image_seq_len
    mu = image_seq_len * m + b
    
    pipe.scheduler.set_timesteps(num_inference_steps, device=device, mu=mu)
    timesteps, num_inference_steps_actual = pipe.get_timesteps(num_inference_steps, strength, device)
    latent_timestep = timesteps[:1]
    
    noise = torch.randn(latents.shape, generator=generator, device=device, dtype=dtype)
    latents_noisy = pipe.scheduler.scale_noise(latents, latent_timestep, noise)
    latents_packed = pipe._pack_latents(latents_noisy, batch_size, num_channels_latents, vae_latent_height, vae_latent_width)
    
    if pipe.transformer.config.guidance_embeds:
        guidance = torch.tensor([0.0], device=device).expand(batch_size)
    else:
        guidance = None
    
    for i, t in enumerate(timesteps):
        timestep = t.expand(batch_size).to(dtype)
        
        with torch.autocast(device_type="cuda", dtype=dtype):
            noise_pred = pipe.transformer(
                hidden_states=latents_packed,
                timestep=timestep / 1000,
                guidance=guidance,
                pooled_projections=pooled_prompt_embeds,
                encoder_hidden_states=prompt_embeds,
                txt_ids=text_ids,
                img_ids=latent_image_ids,
                return_dict=False,
            )[0]
        
        latents_packed = pipe.scheduler.step(noise_pred, t, latents_packed, return_dict=False)[0]
        del noise_pred
        torch.cuda.empty_cache()
    
    latents_out = pipe._unpack_latents(latents_packed, height, width, vae_scale_factor)
    return latents_out

input_folder = "input_images"
output_base = "output_trajectories_flux_dev_latent"
strengths = [0.5, 0.25, 0.15]
num_inference_steps = 28
N = 100
seed = 42
width = 1024
height = 1024

input_images = sorted(Path(input_folder).glob("*.jpg"))

for img_path in tqdm(input_images, desc="Images"):
    img_name = img_path.stem
    
    for strength in tqdm(strengths, desc=f"  {img_name} strengths", leave=False):
        output_dir = Path(output_base) / img_name / f"strength_{strength}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        init_image = Image.open(img_path).convert("RGB")
        init_image = init_image.resize((width, height), Image.LANCZOS)
        init_image.save(output_dir / "step_000.png")
        
        config = {
            "input_image": str(img_path),
            "strength": strength,
            "num_inference_steps": num_inference_steps,
            "N": N,
            "seed": seed,
            "width": width,
            "height": height,
            "model": "black-forest-labs/FLUX.1-dev",
            "vae": "encode_once_decode_each_step",
            "prompt": None,
            "guidance_scale": 0.0,
            "timestamp": datetime.now().isoformat(),
            "mode": "latent_space_trajectory"
        }
        with open(output_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)
        
        current_latents = encode_image_to_latents(init_image, pipe.vae, dtype, device)
        
        for step in tqdm(range(1, N + 1), desc=f"    Steps (s={strength})", leave=False):
            generator = torch.Generator(device=device).manual_seed(seed)
            
            current_latents = run_flux_denoising_on_latents(
                pipe, current_latents, strength, num_inference_steps, generator, height, width
            )
            
            result_image = decode_latents_to_image(current_latents, pipe.vae, dtype)
            result_image.save(output_dir / f"step_{step:03d}.png")
            torch.cuda.empty_cache()

print("\nAll trajectories completed!")
