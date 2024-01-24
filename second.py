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

class ControlNetCannyProcessor:
    @staticmethod
    def process(image_url):
        image = load_image(image_url)

        image = np.array(image)
        image = cv2.Canny(image, 100, 200)

        image = image[:, :, None]
        image = np.concatenate([image, image, image], axis=2)

        canny_image = Image.fromarray(image)
        return canny_image

class DiffusionRunner:
    BASE_PATH = r"D:\raul\models\sd_xl_base_1.0_0.9vae.safetensors"
    # VAE_PATH = r"D:\raul\models\sdxl_vae.safetensors"
    REFINER_PATH = r"D:\raul\models\sd_xl_refiner_1.0_0.9vae.safetensors"
    CANNY_CONTROLNET_PATH = r"D:\raul\models\controlnet-canny-sdxl-1.0"
    
    def __init__(self, use_refiner=False):
        self.use_refiner = use_refiner
        self.controlnet = None
        self.base = None
        self.refiner = None

    def load_controlnet(self):
        self.controlnet = ControlNetModel.from_pretrained(
            self.CANNY_CONTROLNET_PATH,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
        ).to("cuda")

    def load_base(self):
        assert self.controlnet is not None, "Controlnet must be loaded first"
        self.base = StableDiffusionXLControlNetPipeline.from_single_file(
            self.BASE_PATH,
            controlnet=self.controlnet,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
        ).to("cuda")

    # def load_vae(self):
        # self.vae = AutoencoderKL.from_single_file(
        #     self.VAE_PATH,
        #     torch_dtype=torch.float16,
        #     variant="fp16",
        #     use_safetensors=True,
        # ).to("cuda")

    def load_refiner(self):
        self.refiner = StableDiffusionXLImg2ImgPipeline.from_single_file(
            self.REFINER_PATH,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
        ).to("cuda")

    def run(self, prompt, control_image_url, controlnet_conditioning_scale = 0.5):
        if self.controlnet is None:
            self.load_controlnet()

        if self.base is None:
            self.load_base()

        # self.base.enable_model_cpu_offload()
        
        canny_image = ControlNetCannyProcessor.process(control_image_url)
        ImageUtils.save_image_with_timestamp(canny_image, "canny")

        diffusion_args = {
            "prompt": prompt,
            "controlnet_conditioning_scale": controlnet_conditioning_scale,
            "image": canny_image,
            "num_inference_steps": 40
        }

        if self.use_refiner:
            diffusion_args["output_type"] = "latent"
            diffusion_args["denoising_end"] = 0.8

        image = self.base(
            **diffusion_args
        ).images[0]
       
        if self.use_refiner:
            image = self.refiner(
                prompt=self.prompt,
                num_inference_steps=40,
                denoising_start=0.8,
                image=image,
            ).images[0]

        return image

class ImageUtils:
    @staticmethod
    def save_image_with_timestamp(image, suffix = "generated"):
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = f"{timestamp}-{suffix}.jpg"
        folder = ".\gen_imgs"
        os.makedirs(folder, exist_ok=True)  # Ensure the directory exists
        path = os.path.join(folder, filename)  # Create the full path for the file
        image.save(path)

if __name__ == "__main__":
    CONTROL_IMAGE_URL = "https://hf.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/hf-logo.png"
   
    prompt = "A majestic lion jumping from a big stone at night"
    diffusion_runner = DiffusionRunner()
    image = diffusion_runner.run(prompt, CONTROL_IMAGE_URL)

    ImageUtils.save_image_with_timestamp(image)