import os
from upscaler.realesrgan_model import UpscalerRealESRGAN
from diffusers.utils import load_image

input_dir = "./input_imgs"
output_dir = "./gen_imgs"

model_dir = "/home/raul/codelab/models/upscalers"
scale = 4

model_names = ["4x_NMKD-Siax_200k", "DF2K", "4x-UltraSharp"]

for model_name in model_names:
    print(f"Upscaling with {model_name}")

    upscaler = UpscalerRealESRGAN(model_dir, model_name)

    for filename in os.listdir(input_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            print(f"\tUpscaling {filename}")

            input_image = load_image(os.path.join(input_dir, filename))
            
            img = upscaler.do_upscale(input_image)
                
            output_filename = f"image_esrgan_{model_name}_{filename}"
            img.save(os.path.join(output_dir, output_filename))
