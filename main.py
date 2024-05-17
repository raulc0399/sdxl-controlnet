
from diffusion_runner import DiffusionRunner
from image_utils import ImageUtils

from prompts import prompts

def run_diffusion_experiments(diffusion_runner, model, control_image_url, controlnet_conditioning_scale_vals, num_inference_steps_vals, guidance_scale_vals):

    all_experiments_count = len(prompts) * len(controlnet_conditioning_scale_vals) * len(num_inference_steps_vals) * len(guidance_scale_vals)
    experiment_no = 1

    for prompt in prompts:
        for controlnet_conditioning_scale in controlnet_conditioning_scale_vals:
            for num_inference_steps in num_inference_steps_vals:
                for guidance_scale in guidance_scale_vals:
                    print(f"Running experiment {experiment_no} from {all_experiments_count} experiments")
                    print(f"Model: {model}, Controlnet Conditioning Scale: {controlnet_conditioning_scale}, Num Inference Steps: {num_inference_steps}, Guidance Scale: {guidance_scale}, Prompt: {prompt:.20}")

                    image, upscaled_image = diffusion_runner.run(
                        prompt,
                        control_image_url,
                        controlnet_conditioning_scale,
                        num_inference_steps,
                        guidance_scale
                    )
                    
                    params = {
                        "model": model,
                        "controlnet_conditioning_scale": controlnet_conditioning_scale,
                        "num_inference_steps": num_inference_steps,
                        "guidance_scale": guidance_scale,
                        "prompt": prompt,
                        "control_image_url": control_image_url,
                    }

                    ImageUtils.save_image_with_timestamp(image, params=params)

                    if upscaled_image is not None:
                        ImageUtils.save_image_with_timestamp(upscaled_image, "upscaled")

                    experiment_no += 1

                    # if experiment_no == 10:
                    #     exit()

if __name__ == "__main__":
    CONTROL_IMAGE_URL = "/home/raul/codelab/objs/obj4/4j.jpg"

    controlnet_conditioning_scale_vals = [0.5, 1.0, 1.5]
    num_inference_steps_vals = [20, 30, 40]
    guidance_scale_vals = [1.0, 2.0, 4.0, 5.0]

    negative_prompt = ""
    prompt_2 = ""
    negative_prompt_2 = ""

    diffusion_runner = DiffusionRunner(use_sdxl=True)
    run_diffusion_experiments(diffusion_runner, "sdxl", CONTROL_IMAGE_URL, controlnet_conditioning_scale_vals, num_inference_steps_vals, guidance_scale_vals)

    del diffusion_runner

    diffusion_runner = DiffusionRunner(use_sdxl=False)
    run_diffusion_experiments(diffusion_runner, "juggernaut", CONTROL_IMAGE_URL, controlnet_conditioning_scale_vals, num_inference_steps_vals, guidance_scale_vals)