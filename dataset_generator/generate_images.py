from diffusers import AutoPipelineForText2Image
from pathlib import Path
import torch


def generate_image(text: str, output_path: str, pipe: AutoPipelineForText2Image) -> bool:
    try:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        image = pipe(prompt=text, num_inference_steps=1, guidance_scale=0.0).images[0]
        image.save(output_path)
        return True
    except Exception as e:
        print(f"  [image error] {e}")
        return False
