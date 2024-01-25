from diffusers import (
    StableDiffusionXLPipeline,
    StableDiffusionXLControlNetPipeline,
    StableDiffusionXLImg2ImgPipeline,
    ControlNetModel,
    AutoencoderKL,
)
from diffusers.utils import load_image
import numpy as np
import torch
import datetime
import cv2
import os
from PIL import Image


def run_diffusion():
    image = load_image(
        "https://hf.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/hf-logo.png"
    )

    image = np.array(image)
    image = cv2.Canny(image, 100, 200)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    canny_image = Image.fromarray(image)

    save_image_with_timestamp(canny_image, "canny")
    
    controlnet_conditioning_scale = 0.5  # recommended for good generalization
    controlnet = ControlNetModel.from_pretrained(
        r"D:\raul\models\controlnet-canny-sdxl-1.0",
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
    )
    # controlnet = ControlNetModel.from_pretrained(
    #     "diffusers/controlnet-canny-sdxl-1.0",
    #     torch_dtype=torch.float16,
    #     variant="fp16"
    #     #     use_safetensors=True,
    # ).to("cuda")

    vae = AutoencoderKL.from_single_file(
        r"D:\raul\models\sdxl_vae.safetensors",
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
    ).to("cuda")

    # base = StableDiffusionXLPipeline.from_single_file(
    base = StableDiffusionXLControlNetPipeline.from_single_file(
        r"D:\raul\models\sd_xl_base_1.0.safetensors",
        vae=vae,
        controlnet=controlnet,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
    ).to("cuda")
    # base.enable_vae_tiling()
    # base.enable_vae_slicing()

    prompt = "A majestic lion jumping from a big stone at night"

    base.enable_model_cpu_offload()
    
    image = base(
        prompt=prompt,
        controlnet_conditioning_scale=controlnet_conditioning_scale,
        image=canny_image,
        num_inference_steps=40,
        denoising_end=0.8,
        output_type="latent",
    ).images[0]

    # base.to("cpu")
    # del base

    refiner = StableDiffusionXLImg2ImgPipeline.from_single_file(
        r"D:\raul\models\sd_xl_refiner_1.0.safetensors",
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
    ).to("cuda")
    # refiner.enable_vae_tiling()
    # refiner.enable_vae_slicing()

    image = refiner(
        prompt=prompt,
        num_inference_steps=40,
        denoising_start=0.8,
        image=image,
    ).images[0]

    save_image_with_timestamp(image)


def save_image_with_timestamp(image, prefix = "generated"):
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    filename = f"{prefix}-image-{timestamp}.jpg"
    folder = ".\gen_imgs"

    os.makedirs(folder, exist_ok=True)  # Ensure the directory exists
    path = os.path.join(folder, filename)  # Create the full path for the file

    image.save(path)

# add if main
if __name__ == "__main__":
    run_diffusion()