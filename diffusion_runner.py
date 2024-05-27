from image_utils import ImageUtils
from diffusers import (
    StableDiffusionXLPipeline,
    StableDiffusionXLControlNetPipeline,
    StableDiffusionXLImg2ImgPipeline,
    ControlNetModel,
    AutoencoderKL,
    DPMSolverMultistepScheduler,
    StableDiffusionUpscalePipeline,
)

# from diffusers.models.attention_processor import AttnProcessor2_0
from diffusers.utils import load_image
from enum import Enum
import numpy as np
import torch
import cv2
import os
from PIL import Image
from compel import Compel, ReturnedEmbeddingsType

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
        img_padded = np.pad(img, [[0, H_pad], [0, W_pad], [0, 0]], mode="edge")

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
    def process(image_url, res=512, thr_a=150, thr_b=200):
        image = load_image(image_url)

        image = np.array(image)

        image = ControlNetCannyProcessor.canny(image, res, thr_a, thr_b)

        canny_image = Image.fromarray(image)
        return canny_image

class ImageProcessor:
    @staticmethod
    def preprocess_control_image(image_path, dimension=1024):
        image = Image.open(image_path)
        
        width, height = image.size
        crop_size = min(width, height)
        
        left = (width - crop_size) // 2
        top = (height - crop_size) // 2
        right = left + crop_size
        bottom = top + crop_size
        
        image = image.crop((left, top, right, bottom))
        image = image.resize((dimension, dimension))

        if image.mode in ("RGBA", "P"): 
            image = image.convert("RGB")
        
        return image

class ControlNetType(Enum):
    CANNY = "canny"
    DEPTH = "depth"
    MISTO = "misto"

class DiffusionRunner:
    BASE_PATH_JUGGERNAUTXL = "/home/raul/codelab/models/juggernautXL_v8Rundiffusion.safetensors"
    BASE_PATH_SDXL = "/home/raul/codelab/models/sd_xl_base_1.0_0.9vae.safetensors"
    # VAE_PATH = "/home/raul/codelab/models/sdxl_vae.safetensors"
    REFINER_PATH = "/home/raul/codelab/models/sd_xl_refiner_1.0_0.9vae.safetensors"
    CANNY_CONTROLNET_PATH = "/home/raul/codelab/models/controlnet-canny-sdxl-1.0"
    DEPTH_CONTROLNET_PATH = "/home/raul/codelab/models/controlnet-depth-sdxl-1.0"
    MISTOLINE_CONTROLNET_PATH = "/home/raul/codelab/models/mistoLine_fp16"
    UPSCALER_MODEL_ID = "stabilityai/stable-diffusion-x4-upscaler"

    def __init__(self, use_sdxl = True, use_refiner=False, controlnet_type="canny"):
        self.use_refiner = use_refiner
        self.controlnet_type = controlnet_type
        self.controlnet = None
        self.pipe = None
        self.refiner = None
        self.compel_proc = None
        self.upscalePipeline = None
        
        self.model_path = self.BASE_PATH_SDXL if use_sdxl else self.BASE_PATH_JUGGERNAUTXL

    def load_controlnet(self):
        if self.controlnet_type == ControlNetType.MISTO:
            controlnet_path = self.MISTOLINE_CONTROLNET_PATH
        elif self.controlnet_type == ControlNetType.DEPTH:
            controlnet_path = self.DEPTH_CONTROLNET_PATH
        elif self.controlnet_type == ControlNetType.CANNY:
            controlnet_path = self.CANNY_CONTROLNET_PATH
        else:
            raise ValueError(f"Unknown controlnet type: {self.controlnet_type}")
        
        print(f"Loading controlnet from: {controlnet_path}")

        self.controlnet = ControlNetModel.from_pretrained(
            controlnet_path,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
        )

    def load_base(self):
        assert self.controlnet is not None, "Controlnet must be loaded first"
        self.pipe = StableDiffusionXLControlNetPipeline.from_single_file(
            self.model_path,
            controlnet=self.controlnet,
            torch_dtype=torch.float16,
            use_safetensors=True,
        )

        self.compel = Compel(
            tokenizer=[self.pipe.tokenizer, self.pipe.tokenizer_2],
            text_encoder=[self.pipe.text_encoder, self.pipe.text_encoder_2],
            returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
            requires_pooled=[False, True],
        )

        # self.pipe.to("cuda")
        self.pipe.enable_model_cpu_offload()

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
        )

        # self.refiner.to("cuda")
        self.refiner.enable_model_cpu_offload()

    def load_upscaler(self):
        self.upscalePipeline = StableDiffusionUpscalePipeline.from_pretrained(
            self.UPSCALER_MODEL_ID,
            torch_dtype=torch.float16
        )

        # self.upscalePipeline.to("cuda")
        self.upscalePipeline.enable_model_cpu_offload()
    
    def run_refiner(self, image):
        if self.refiner is None:
            self.load_refiner()

        return self.refiner(
            prompt=prompt,
            num_inference_steps=40,
            denoising_start=0.8,
            image=image,
        ).images[0]

    def run_upscaler(self, prompt, image):
        if self.upscalePipeline is None:
            self.load_upscaler()

        upscaled_image = self.upscalePipeline(prompt=prompt, image=image).images[0]
        return upscaled_image
   

    @staticmethod
    # used to disable guidance_scale after a certain number of steps
    def callback_dynamic_cfg(pipe, step_index, timestep, callback_kwargs):
        # adjust the batch_size of prompt_embeds according to guidance_scale
        if step_index == int(pipe.num_timesteps * 0.4):
            prompt_embeds = callback_kwargs["prompt_embeds"]
            # prompt_embeds = torch.chunk(prompt_embeds, 2, dim=0)[-1] # prompt_embeds.chunk(2)[-1]

            # update guidance_scale and prompt_embeds
            pipe._guidance_scale = 1.0
            callback_kwargs["prompt_embeds"] = prompt_embeds

        return callback_kwargs

    def run(
        self, prompt, control_image_url, controlnet_conditioning_scale = 0.5, num_inference_steps = 40, guidance_scale = 4.0, 
        seeds=None, num_images=1,
        # not used for now
        upscale=False, prompt_2=None, negative_prompt=None, negative_prompt_2=None,
    ):
        if self.controlnet is None:
            self.load_controlnet()

        if self.pipe is None:
            self.load_base()

        # change scheduler
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config,
                                                                      algorithm_type="sde-dpmsolver++",
                                                                      use_karras_sigmas=False)
        
        processed_image = ImageProcessor.preprocess_control_image(control_image_url)
        ImageUtils.save_image_with_timestamp(processed_image, "preprocessed")
        
        # conditioning, pooled = self.compel(prompt)

        # check https://huggingface.co/docs/diffusers/v0.13.0/en/using-diffusers/reproducibility
        generator = None
        if seeds is not None:
            assert len(seeds) == num_images, "The length of seeds array must be equal to num_images."
            generator = [torch.Generator(device="cuda").manual_seed(seed) for seed in seeds]

        diffusion_args = {
            # "prompt_embeds": conditioning,
            # "pooled_prompt_embeds": pooled,
            "generator": generator,
            "prompt": prompt,
            "prompt_2": prompt_2,
            "negative_prompt": negative_prompt,
            "negative_prompt_2": negative_prompt_2,
            "controlnet_conditioning_scale": controlnet_conditioning_scale,
            "image": processed_image,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "num_images_per_prompt": 1,
        }

        print("Parameters for diffuser:")
        print("prompt:", prompt)
        print("prompt_2:", prompt_2)
        print("negative_prompt:", negative_prompt)
        print("negative_prompt_2:", negative_prompt_2)
        print("controlnet_conditioning_scale:", controlnet_conditioning_scale)
        print("num_inference_steps:", num_inference_steps)
        print("guidance_scale:", guidance_scale)
        # print("seeds:", seeds)

        if self.use_refiner:
            diffusion_args["output_type"] = "latent"
            diffusion_args["denoising_end"] = 0.8

        image = self.pipe(
            **diffusion_args,
            # callback_on_step_end=self.callback_dynamic_cfg,
            # callback_on_step_end_tensor_inputs=['prompt_embeds']
        ).images[0]

        # https://github.com/huggingface/diffusers/issues/4657
        if self.use_refiner:
            image = self.run_refiner(image)

        upscaled_image = None
        if upscale:
            upscaled_image = self.run_upscaler(prompt, image)

        return image, upscaled_image
