from diffusers import (
    StableDiffusionXLPipeline,
    StableDiffusionXLControlNetPipeline,
    StableDiffusionXLImg2ImgPipeline,
    ControlNetModel,
    AutoencoderKL,
)
from diffusers.models.attention_processor import AttnProcessor2_0
from diffusers.utils import load_image
import numpy as np
import torch
import datetime
import cv2
import os
from PIL import Image

def pad64(x):
    return int(np.ceil(float(x) / 64.0) * 64 - x)


def safer_memory(x):
    # Fix many MAC/AMD problems
    return np.ascontiguousarray(x.copy()).copy()

# code taken from https://github.com/Mikubill/sd-webui-controlnet
class ControlNetCannyProcessor:
    @staticmethod
    def HWC3(x):
        assert x.dtype == np.uint8
        if x.ndim == 2:
            x = x[:, :, None]
        assert x.ndim == 3
        H, W, C = x.shape
        assert C == 1 or C == 3 or C == 4
        if C == 3:
            return x
        if C == 1:
            return np.concatenate([x, x, x], axis=2)
        if C == 4:
            color = x[:, :, 0:3].astype(np.float32)
            alpha = x[:, :, 3:4].astype(np.float32) / 255.0
            y = color * alpha + 255.0 * (1.0 - alpha)
            y = y.clip(0, 255).astype(np.uint8)
            return y
        
    @staticmethod
    def resize_image_with_pad(input_image, resolution, skip_hwc3=False):
        if skip_hwc3:
            img = input_image
        else:
            img = ControlNetCannyProcessor.HWC3(input_image)
        H_raw, W_raw, _ = img.shape
        k = float(resolution) / float(min(H_raw, W_raw))
        interpolation = cv2.INTER_CUBIC if k > 1 else cv2.INTER_AREA
        H_target = int(np.round(float(H_raw) * k))
        W_target = int(np.round(float(W_raw) * k))
        img = cv2.resize(img, (W_target, H_target), interpolation=interpolation)
        H_pad, W_pad = pad64(H_target), pad64(W_target)
        img_padded = np.pad(img, [[0, H_pad], [0, W_pad], [0, 0]], mode='edge')

        def remove_pad(x):
            return safer_memory(x[:H_target, :W_target])

        return safer_memory(img_padded), remove_pad

    @staticmethod
    def canny(img, res, thr_a, thr_b):
        l, h = thr_a, thr_b

        img, remove_pad = ControlNetCannyProcessor.resize_image_with_pad(img, res)        
        
        result = cv2.Canny(img, l, h)

        return remove_pad(result)
        
        # contours, h = cv2.findContours(canny_img, 
        #                     cv2.RETR_EXTERNAL,
        #                     cv2.CHAIN_APPROX_NONE)
        # contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        # cv2.drawContours(canny_img, contours, -1, (255, 255, 255), thickness = 1)
        
        # result, remove_pad = ControlNetCannyProcessor.resize_image_with_pad(canny_img, res)

        # return remove_pad(canny_img)

    @staticmethod
    def process(image_url, res=512, thr_a=15, thr_b=55):
        image = load_image(image_url)

        image = np.array(image)
        
        image = ControlNetCannyProcessor.canny(image, res, thr_a, thr_b)

        canny_image = Image.fromarray(image)
        return canny_image

class DiffusionRunner:
    BASE_PATH = r"D:\raul\models\juggernautXL_v8Rundiffusion.safetensors"
    # BASE_PATH = r"D:\raul\models\sd_xl_base_1.0_0.9vae.safetensors"
    # VAE_PATH = r"D:\raul\models\sdxl_vae.safetensors"
    REFINER_PATH = r"D:\raul\models\sd_xl_refiner_1.0_0.9vae.safetensors"
    CANNY_CONTROLNET_PATH = r"D:\raul\models\controlnet-canny-sdxl-1.0"
    
    def __init__(self, use_refiner=False):
        self.use_refiner = use_refiner
        self.controlnet = None
        self.pipe = None
        self.refiner = None

    def load_controlnet(self):
        self.controlnet = ControlNetModel.from_pretrained(
            self.CANNY_CONTROLNET_PATH,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
        )#.to("cuda")

    def load_base(self):
        assert self.controlnet is not None, "Controlnet must be loaded first"
        self.pipe = StableDiffusionXLControlNetPipeline.from_single_file(
            self.BASE_PATH,
            controlnet=self.controlnet,
            torch_dtype=torch.float16,
            # variant="fp16",
            use_safetensors=True,
        )

    # def load_vae(self):
        # self.vae = AutoencoderKL.from_single_file(
        #     self.VAE_PATH,
        #     torch_dtype=torch.float16,
        #     variant="fp16",
        #     use_safetensors=True,
        # )

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

        if self.pipe is None:
            self.load_base()

        self.pipe.to("cuda")

        # self.pipe.enable_model_cpu_offload()
        
        # self.pipe.unet.set_attn_processor(AttnProcessor2_0())
            
        # self.pipe.unet.to(memory_format=torch.channels_last)
        # self.pipe.controlnet.to(memory_format=torch.channels_last)

        # self.pipe.unet = torch.compile(self.pipe.unet, mode="reduce-overhead", fullgraph=True)
        # self.pipe.controlnet = torch.compile(self.pipe.controlnet, mode="reduce-overhead", fullgraph=True)
        
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

        image = self.pipe(
            **diffusion_args
        ).images[0]

        # https://github.com/huggingface/diffusers/issues/4657
        if self.use_refiner:
            if self.refiner is None:
                self.load_refiner()

            image = self.refiner(
                prompt=prompt,
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
    CONTROL_IMAGE_URL = r"D:\raul\stuff\objs\obj4\4j.jpg"
   
    prompt = "architectural rendering of houses of same design in german suburb, nice warm, day, sunny, white exterior"
    diffusion_runner = DiffusionRunner()
    image = diffusion_runner.run(prompt, CONTROL_IMAGE_URL)

    ImageUtils.save_image_with_timestamp(image)