# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input, Path
import os
import sys
sys.path.append("/cross-image-attention-hf")
from PIL import Image
from typing import Optional
from config import RunConfig
from pathlib import Path as Path2
from run import run_appearance_transfer
from diffusers.training_utils import set_seed
from appearance_transfer_model import AppearanceTransferModel
from utils.latent_utils import load_latents_or_invert_images
from utils.model_utils import get_stable_diffusion_model


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        self.pipe = get_stable_diffusion_model()

    def main_pipeline(self, app_image_path: str,
                  struct_image_path: str,
                  domain_name: str,
                  seed: int,
                  prompt: Optional[str] = None) -> Image.Image:
        if prompt == "":
            prompt = None
        config = RunConfig(
            app_image_path=Path2(app_image_path),
            struct_image_path=Path2(struct_image_path),
            domain_name=domain_name,
            prompt=prompt,
            seed=seed,
            load_latents=False
        )
        print(config)
        set_seed(config.seed)
        model = AppearanceTransferModel(config=config, pipe=self.pipe)
        latents_app, latents_struct, noise_app, noise_struct = load_latents_or_invert_images(model=model, cfg=config)
        model.set_latents(latents_app, latents_struct)
        model.set_noise(noise_app, noise_struct)
        print("Running appearance transfer...")
        images = run_appearance_transfer(model=model, cfg=config)
        print("Done.")
        return [images[0]]

    def predict(
        self,
        appearance_img: Path = Input(description="Input appearance image"),
        structure_img: Path = Input(description="Input structure image"),
        domain: str = Input(description="Single word input domain", default="photo"),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=None
        ),
    ) -> Path:
        """Run a single prediction on the model"""
        if seed is None:
            seed = int.from_bytes(os.urandom(3), "big")
        print(f"Using seed: {seed}")

        output = self.main_pipeline(str(appearance_img), str(structure_img), domain, seed, None)
        output_path = "/tmp/output.png"
        output[0].save(output_path)
        return Path(output_path)
