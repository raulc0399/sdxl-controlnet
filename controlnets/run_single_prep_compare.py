import torch
from diffusers.utils import load_image
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel, AutoencoderKL
from diffusers import UniPCMultistepScheduler, DPMSolverMultistepScheduler, DEISMultistepScheduler
from datetime import datetime
import itertools
import json
import os
import sys
from controlnet_aux import CannyDetector, AnylineDetector

ACCESS_TOKEN = os.getenv("HF_TOKEN")

# relative to the starting script
INPUT_FOLDER = "./input_imgs"
OUTPUT_FOLDER = "./gen_imgs"

# xinsir/controlnet-union-sdxl-1.0
MODELS = [
    "diffusers/controlnet-canny-sdxl-1.0",
    "TheMistoAI/MistoLine",
    "xinsir/controlnet-canny-sdxl-1.0",
]

BASE_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
GENERATOR = torch.Generator(device="cuda").manual_seed(87544357)

NEGATIVE_PROMPT = 'low quality, bad quality, sketches'

# PROMPT = """Make a modern professional photo real visualization or Photograph from this clay 3d white model. Keep the Details and proportions from the model and elevations! the style of the architecture should be modern western and new build conditions. The roof with dark glazed roof tiles. the style of the image should late decent afternoon summer sun from side. The Environment style in south germany urban style. interior lights on. long tree shadows from late warm sun. sub urban environment. clean blue sky, desaturated colors and professional grading and postproduction."""
PROMPT = """Make a modern professional photo real visualization or Photograph from this clay 3d white model. Keep the Details and proportions from the model and elevations! the style of the architecture should be modern western and new build conditions. The roof with dark glazed roof tiles. the style of the image should midday, summer sun. The Environment style in south Germany urban style. sub urban environment. clean blue sky, desaturated colors and professional grading and postproduction."""
    
def get_control_images():
    print("\033[96mPreparing control images\033[0m")
    
    """Get both control images - original and preprocessed"""
    # Original c_edges.png
    c_edges_image = load_image(f"{INPUT_FOLDER}/c_edges.png")
    
    # Load cp?.png and apply preprocessing
    cp1_image = load_image(f"{INPUT_FOLDER}/cp1.png")
    cp2_image = load_image(f"{INPUT_FOLDER}/cp2.png")
    
    # Apply CannyDetector and AnylineDetector to cp2.png
    canny_detector = CannyDetector()
    anyline_detector = AnylineDetector.from_pretrained("TheMistoAI/MistoLine", filename="MTEED.pth", subfolder="Anyline")

    cp1_canny = canny_detector(cp1_image)
    cp1_anyline = anyline_detector(cp1_image)

    cp2_canny = canny_detector(cp2_image)
    cp2_anyline = anyline_detector(cp2_image)
    
    # Save the processed images to disk
    cp1_canny.save(f"{OUTPUT_FOLDER}/cp1_canny.png")
    cp1_anyline.save(f"{OUTPUT_FOLDER}/cp1_anyline.png")
    
    cp2_canny.save(f"{OUTPUT_FOLDER}/cp2_canny.png")
    cp2_anyline.save(f"{OUTPUT_FOLDER}/cp2_anyline.png")

    print("\033[96mControl images prepared and saved.\033[0m")

    del canny_detector
    del anyline_detector
    torch.cuda.empty_cache()
    
    print("GPU cleared.")
    
    return [
        ("c_edges.png", c_edges_image),
        ("cp1_canny.png", cp1_canny),
        ("cp1_anyline.png", cp1_anyline),
        ("cp2_canny.png", cp2_canny),
        ("cp2_anyline.png", cp2_anyline)
    ]

def load_pipeline(controlnet_model):
    """Load the pipeline with specified controlnet model"""
    variant_kwargs = {"variant": "fp16"} if "MistoLine" in controlnet_model else {}
    
    controlnet = ControlNetModel.from_pretrained(
        controlnet_model, 
        torch_dtype=torch.float16,
        **variant_kwargs
    ).to("cuda")
    
    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        BASE_MODEL,
        controlnet=controlnet,
        vae=vae,
        torch_dtype=torch.float16,
        token=ACCESS_TOKEN,
    ).to("cuda")
    
    print(f"\nLoaded model: {controlnet_model}")
    print(f"Pipeline device map: {pipe.hf_device_map}")
    print(f"Controlnet device: {controlnet.device}")
    
    return pipe

def generate_image(pipe, control_image, prompt_text, guidance_scale, conditioning_scale, num_steps,
                   image_index, control_image_name, model_name, scheduler):
    """Generate image with specified parameters"""
    width, height = control_image.size

    image = pipe(
        prompt_text,
        negative_prompt=NEGATIVE_PROMPT,
        image=control_image,
        width=width,
        height=height,
        controlnet_conditioning_scale=conditioning_scale,
        num_inference_steps=num_steps,
        guidance_scale=guidance_scale,
        generator=GENERATOR,
    ).images[0]
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Include control image name in the filename
    control_suffix = control_image_name.replace('.png', '').replace('_', '-')
    base_name = f"{timestamp}_{image_index:04d}_{control_suffix}_g{guidance_scale}+c{conditioning_scale}_s{num_steps}_{scheduler}"

    # Save image
    image_path = f"{OUTPUT_FOLDER}/{model_name}/{base_name}.png"
    image.save(image_path)
    
    # Save parameters
    params = {
        "model_name": model_name,
        "conditioning_scale": conditioning_scale,
        "num_steps": num_steps,
        "image_path": image_path,
        "control_image": control_image_name,
        "scheduler": scheduler,
        "prompt": prompt_text
    }
    
    params_path = f"{OUTPUT_FOLDER}/{model_name}/params/{base_name}.json"
    with open(params_path, 'w') as f:
        json.dump(params, f, indent=4, separators=(',\n', ': '))
    
    print(f"Saved image: {image_path}")

def ensure_params_dir(model):
    params_dir = f"{OUTPUT_FOLDER}/{model}/params"
    os.makedirs(params_dir, exist_ok=True)

def main(model_index):
    image_counter = 0
    
    # Parameter combinations - optimized for photorealistic results
    prompts = [PROMPT]
    # guidance_scales = [6.0, 7.0]
    # conditioning_scales = [0.6, 0.75, 1.0]
    # inference_steps = [30, 70, 120]
    guidance_scales = [7.0]
    conditioning_scales = [0.75, 1.0]
    inference_steps = [30, 70, 120]
    # default should be first
    schedulers = ['default', 'dpm2m_sde_karras', 'dpm2m_sde_karras_with_euler_at_final'] # , 'dpm2m_sde_karras_with_lambdas', 'dpm_flow', 'dpm3m_sde_karras']

    # Calculate total combinations (now includes 3 control images)
    total_combinations = (
        len(prompts) *
        len(guidance_scales) *
        len(conditioning_scales) *
        len(inference_steps) *
        len(schedulers) *
        5  # 5 control images: c_edges, cp1_canny, cp1_anyline, cp2_canny, cp2_anyline
    )
    print(f"Total combinations to generate: {total_combinations}")

    # Generate all parameter combinations using itertools
    param_combinations = itertools.product(
        prompts,
        guidance_scales,
        conditioning_scales,
        inference_steps,
        schedulers
    )
    
    model = MODELS[model_index]
    try:
        model_name = model.replace("/", "-")
        ensure_params_dir(model_name)

        control_images = get_control_images()

        pipe = load_pipeline(model)

        orig_config = pipe.scheduler.config
        for prompt_text, guidance_scale, cond_scale, steps, scheduler in param_combinations:
            # Run for each control image
            for control_image_name, control_image in control_images:
                try:
                    if scheduler == 'dpm3m_sde_karras':
                        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                            orig_config,
                            use_karras_sigmas=True, 
                            algorithm_type="sde-dpmsolver++",
                            solver_order=3
                        )                    
                    elif scheduler == 'dmp2m_sde_karras':
                        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                            orig_config,
                            use_karras_sigmas=True, 
                            algorithm_type="sde-dpmsolver++",
                            solver_order=2
                        )
                    elif scheduler == 'dpm2m_sde_karras_with_lambdas':
                        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                            orig_config,
                            use_karras_sigmas=True, 
                            algorithm_type="sde-dpmsolver++",
                            solver_order=2,
                            use_lu_lambdas=True
                        )
                    elif scheduler == 'dpm2m_sde_karras_with_euler_at_final':
                        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                            orig_config,
                            use_karras_sigmas=True, 
                            algorithm_type="sde-dpmsolver++",
                            solver_order=2,
                            euler_at_final=True
                        )
                    elif scheduler == 'dpm_flow':
                        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                            orig_config,
                            use_flow_sigmas=True, 
                            solver_order=2
                        )

                    generate_image(
                        pipe,
                        control_image,
                        prompt_text,
                        guidance_scale,
                        cond_scale,
                        steps,
                        image_counter,
                        control_image_name,
                        model_name,
                        scheduler
                    )

                    image_counter += 1

                except Exception as e:
                    print(f"Error generating image for {model} with control image {control_image_name} and params: {cond_scale}, {steps}")
                    print(f"Error: {str(e)}")

        # clear gpu
        del pipe
        torch.cuda.empty_cache()
                
    except Exception as e:
        print(f"Error while processing {model}")
        print(f"Error: {str(e)}")

def check_model_index(index) -> bool:
    try:
        index = int(index)
        if 0 <= index < len(MODELS):
            return True
        else:
            return False
    except ValueError:
        return False

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python run_single.py <model index>")
        exit()
    
    model_index = sys.argv[1]

    if check_model_index(model_index):
        main(int(model_index))
    else:
        print("Index not valid")
