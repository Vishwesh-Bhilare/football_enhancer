"""Optional Stable Diffusion inpainting with a CUDA-aware fallback."""

import numpy as np
import torch
from PIL import Image


class SDInpainter:

    def __init__(self):
        self.pipe = None
        self.generator = None

        if not torch.cuda.is_available():
            print("Stable Diffusion disabled: CUDA is unavailable")
            return

        try:
            from diffusers import StableDiffusionInpaintPipeline
        except ImportError as exc:
            print(f"Stable Diffusion disabled: optional dependency unavailable ({exc})")
            return

        print("Loading Stable Diffusion inpainting...")

        try:
            self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
                "runwayml/stable-diffusion-inpainting",
                torch_dtype=torch.float16,
            ).to("cuda")

            try:
                self.pipe.enable_xformers_memory_efficient_attention()
            except (ImportError, ModuleNotFoundError):
                print("xFormers unavailable; using standard attention")

            self.pipe.safety_checker = None
            self.pipe.set_progress_bar_config(disable=True)
            self.generator = torch.Generator(device="cuda").manual_seed(42)
        except (OSError, RuntimeError) as exc:
            self.pipe = None
            self.generator = None
            print(f"Stable Diffusion disabled: model could not be loaded ({exc})")
            return

        print("Stable Diffusion ready")

    @property
    def is_enabled(self):
        return self.pipe is not None

    def inpaint(self, frame, mask):
        """Inpaint a BGR frame, returning it unchanged when SD is disabled."""
        if not self.is_enabled:
            return frame

        image = Image.fromarray(frame[:, :, ::-1])
        mask_img = Image.fromarray((mask * 255).astype(np.uint8))

        result = self.pipe(
            prompt="football field grass stadium",
            image=image,
            mask_image=mask_img,
            generator=self.generator,
            guidance_scale=6.5,
            num_inference_steps=20,
        ).images[0]

        return np.array(result)[:, :, ::-1]
