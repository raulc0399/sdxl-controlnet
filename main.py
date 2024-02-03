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
import numpy as np
import torch
import datetime
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


class DiffusionRunner:
    BASE_PATH = r"D:\raul\models\juggernautXL_v8Rundiffusion.safetensors"
    # BASE_PATH = r"D:\raul\models\sd_xl_base_1.0_0.9vae.safetensors"
    # VAE_PATH = r"D:\raul\models\sdxl_vae.safetensors"
    REFINER_PATH = r"D:\raul\models\sd_xl_refiner_1.0_0.9vae.safetensors"
    CANNY_CONTROLNET_PATH = r"D:\raul\models\controlnet-canny-sdxl-1.0"
    UPSCALER_MODEL_ID = "stabilityai/stable-diffusion-x4-upscaler"

    def __init__(self, use_refiner=False):
        self.use_refiner = use_refiner
        self.controlnet = None
        self.pipe = None
        self.refiner = None
        self.compel_proc = None
        self.upscalePipeline = None

    def load_controlnet(self):
        self.controlnet = ControlNetModel.from_pretrained(
            self.CANNY_CONTROLNET_PATH,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
        )  # .to("cuda")

    def load_base(self):
        assert self.controlnet is not None, "Controlnet must be loaded first"
        self.pipe = StableDiffusionXLControlNetPipeline.from_single_file(
            self.BASE_PATH,
            controlnet=self.controlnet,
            torch_dtype=torch.float16,
            # variant="fp16",
            use_safetensors=True,
        )

        self.compel = Compel(
            tokenizer=[self.pipe.tokenizer, self.pipe.tokenizer_2],
            text_encoder=[self.pipe.text_encoder, self.pipe.text_encoder_2],
            returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
            requires_pooled=[False, True],
        )

    def load_upscaler(self):
        self.upscalePipeline = StableDiffusionUpscalePipeline.from_pretrained(
            self.UPSCALER_MODEL_ID,
            torch_dtype=torch.float16
        ).to("cuda")

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

    def run_upscaler(self, prompt, image):
        self.pipe.to("cpu")

        if self.upscalePipeline is None:
            self.load_upscaler()

        upscaled_image = self.upscalePipeline(prompt=prompt, image=image).images[0]
        return upscaled_image
    
    def run(
        self, prompt, prompt_2, negative_prompt, negative_prompt_2, control_image_url, upscale=False
    ):
        if self.controlnet is None:
            self.load_controlnet()

        if self.pipe is None:
            self.load_base()

        # self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config,
        #                                                             #   algorithm_type="sde-dpmsolver++",
        #                                                             use_karras_sigmas=True)

        # self.pipe.enable_model_cpu_offload()

        # self.pipe.unet.set_attn_processor(AttnProcessor2_0())

        # self.pipe.unet.to(memory_format=torch.channels_last)
        # self.pipe.controlnet.to(memory_format=torch.channels_last)

        # self.pipe.unet = torch.compile(self.pipe.unet, mode="reduce-overhead", fullgraph=True)
        # self.pipe.controlnet = torch.compile(self.pipe.controlnet, mode="reduce-overhead", fullgraph=True)

        self.pipe.to("cuda")

        canny_image = ControlNetCannyProcessor.process(control_image_url)
        ImageUtils.save_image_with_timestamp(canny_image, "canny")

        conditioning, pooled = self.compel(prompt)

        # check https://huggingface.co/docs/diffusers/v0.13.0/en/using-diffusers/reproducibility
        generator = None
        seed = None  # set to 0 for random, or to a specific seed value
        if seed is not None:
            torch.manual_seed(seed)
            generator = [torch.Generator(device="cuda").manual_seed(seed)]

        diffusion_args = {
            "prompt_embeds": conditioning,
            "pooled_prompt_embeds": pooled,
            "generator": generator,
            # "prompt": prompt,
            # "prompt_2": prompt_2,
            # "negative_prompt": negative_prompt,
            # "negative_prompt_2": negative_prompt_2,
            "controlnet_conditioning_scale": 0.5,
            "image": canny_image,
            "num_inference_steps": 40,
            "guide_scale": 4.0,
        }

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
            if self.refiner is None:
                self.load_refiner()

            image = self.refiner(
                prompt=prompt,
                num_inference_steps=40,
                denoising_start=0.8,
                image=image,
            ).images[0]

        upscaled_image = None
        if upscale:
            upscaled_image = self.run_upscaler(prompt, image)

        return image, upscaled_image


class ImageUtils:
    @staticmethod
    def save_image_with_timestamp(image, suffix="generated"):
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = f"{timestamp}-{suffix}.jpg"
        folder = ".\gen_imgs"
        os.makedirs(folder, exist_ok=True)  # Ensure the directory exists
        path = os.path.join(folder, filename)  # Create the full path for the file
        image.save(path)

if __name__ == "__main__":
    CONTROL_IMAGE_URL = r"D:\raul\stuff\objs\obj4\4j.jpg"

    # prompt = "A realistic image of a modern house in the suburbs of a modern city, showcasing a unique blend of classic architecture with contemporary elements. The house is not on the main street, surrounded by a variety of vegetation. It should display features typical of traditional German houses, such as steep gabled roofs or timber framing, integrated with modern design aspects like geometric (quadratic) shapes and large glass panels. Vary the angle of the image for a different perspective, and alter the sun's position to change the lighting, creating distinctive shadows and highlights. The surrounding environment should have diverse trees and shrubs, reinforcing the house's connection with nature. The scene is captured on a sunny day to accentuate the fusion of architectural styles"
    # prompt = "a 3d rendering of a row of houses with realistic staircase between the floors, sunny, white exterior, warm day, modern city suburb"
    # prompt = "Architecture photography of a row of houses with a staircase between the floors, sunny, white exterior, warm day, modern city"
    # prompt = "Hyperdetailed photography of a row of houses with a staircase between the floors, sunny, white exterior, warm day, modern city"

    prompt = """A 3d rendering of a modern house in the suburbs of a modern city. Stairs between the floors.
The house is not on the main street, nice clean vegetation.
sun's position in the morning, on a warm sunny day. 
"""

    prompt_2 = ""

    negative_prompt = "semi-realistic, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate"
    negative_prompt_2 = ""

    diffusion_runner = DiffusionRunner()
    image, upscaled_image = diffusion_runner.run(
        prompt,
        prompt_2,
        negative_prompt,
        negative_prompt_2,
        CONTROL_IMAGE_URL,
        upscale=True
    )

    ImageUtils.save_image_with_timestamp(image)

    if upscaled_image is not None:
        ImageUtils.save_image_with_timestamp(upscaled_image, "upscaled")
