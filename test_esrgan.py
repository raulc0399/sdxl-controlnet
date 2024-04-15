from upscaler.esrgan_model import UpscalerESRGAN
from diffusers.utils import load_image

if __name__ == "__main__":
    input_image = load_image(r"D:\raul\stuff\try\gen_imgs\20240203-191525-generated.jpg")

    model_dir = r"D:\raul\models\upscalers"
    scale = 4

    model_name = "4x_NMKD-Siax_200k"
    upscaler = UpscalerESRGAN(model_name, model_dir, scale)
    img = upscaler.do_upscale(input_image)
    
    img.save(rf"D:\raul\stuff\try\gen_imgs\image_esrgan_{model_name}.jpg")    

    model_name = "DF2K"
    upscaler = UpscalerESRGAN(model_name, model_dir, scale)
    img = upscaler.do_upscale(input_image)

    img.save(rf"D:\raul\stuff\try\gen_imgs\image_esrgan_{model_name}.jpg")    
    