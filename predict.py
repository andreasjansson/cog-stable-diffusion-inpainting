import os
from typing import List

import torch
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image
import PIL.ImageOps
from cog import BasePredictor, Input, Path


MODEL_CACHE = "diffusers-cache"


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        print("Loading pipeline...")

        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting",
            cache_dir=MODEL_CACHE,
            local_files_only=True,
            revision="fp16",
            torch_dtype=torch.float16,
        ).to("cuda")

    @torch.inference_mode()
    @torch.cuda.amp.autocast()
    def predict(
        self,
        prompt: str = Input(description="Input prompt", default=""),
        image: Path = Input(
            description="Input image to in-paint. Width and height should both be divisible by 8. If they're not, the image will be center cropped to the nearest width and height divisible by 8",
        ),
        mask: Path = Input(
            description="Black and white image to use as mask. White pixels are inpainted and black pixels are preserved.",
        ),
        invert_mask: bool = Input(
            description="If this is true, then black pixels are inpainted and white pixels are preserved.",
            default=False,
        ),
        num_outputs: int = Input(
            description="Number of images to output. NSFW filter in enabled, so you may get fewer outputs than requested if flagged",
            ge=1,
            le=4,
            default=1,
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps", ge=1, le=500, default=50
        ),
        guidance_scale: float = Input(
            description="Scale for classifier-free guidance", ge=1, le=20, default=7.5
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=None
        ),
    ) -> List[Path]:
        """Run a single prediction on the model"""
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

        image = Image.open(image).convert("RGB")
        mask = Image.open(mask).convert("RGB")

        if invert_mask:
            mask = PIL.ImageOps.invert(mask)

        if image.width % 8 != 0 or image.height % 8 != 0:
            if mask.size == image.size:
                mask = crop(mask)
            image = crop(image)

        if mask.size != image.size:
            print(
                f"WARNING: Mask size ({mask.width}, {mask.height}) is different to image size ({image.width}, {image.height}). Mask will be resized to image size."
            )
            mask = mask.resize(image.size)

        generator = torch.Generator("cuda").manual_seed(seed)
        output = self.pipe(
            prompt=prompt,
            image=image,
            num_images_per_prompt=num_outputs,
            mask_image=mask,
            width=image.width,
            height=image.height,
            guidance_scale=guidance_scale,
            generator=generator,
            num_inference_steps=num_inference_steps,
        )

        samples = [
            output.images[i]
            for i, nsfw_flag in enumerate(output.nsfw_content_detected)
            if not nsfw_flag
        ]

        if len(samples) == 0:
            raise Exception(
                f"NSFW content detected. Try running it again, or try a different prompt."
            )

        if num_outputs > len(samples):
            print(
                f"NSFW content detected in {num_outputs - len(samples)} outputs, returning the remaining {len(samples)} images."
            )
        output_paths = []
        for i, sample in enumerate(samples):
            output_path = f"/tmp/out-{i}.png"
            sample.save(output_path)
            output_paths.append(Path(output_path))

        return output_paths


def crop(image):
    height = (image.height // 8) * 8
    width = (image.width // 8) * 8
    left = int((image.width - width) / 2)
    right = left + width
    top = int((image.height - height) / 2)
    bottom = top + height
    image = image.crop((left, top, right, bottom))
    return image
