import torch
from diffusers.utils import load_image
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel, AutoencoderKL
from diffusers import UniPCMultistepScheduler, DPMSolverMultistepScheduler, DEISMultistepScheduler
from datetime import datetime
import itertools
import json
import os
import sys

# relative to the starting script
INPUT_FOLDER = "./input_imgs"
OUTPUT_FOLDER = "./gen_imgs"

# "xinsir/controlnet-union-sdxl-1.0", - to try, separate pipeline https://github.com/xinsir6/ControlNetPlus/blob/main/controlnet_union_test_segment.py
MODELS = [
    "diffusers/controlnet-canny-sdxl-1.0",
    "diffusers/controlnet-depth-sdxl-1.0",
    "TheMistoAI/MistoLine",
    "xinsir/controlnet-canny-sdxl-1.0",
    "xinsir/controlnet-depth-sdxl-1.0",
    "xinsir/controlnet-scribble-sdxl-1.0"
]

BASE_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
GENERATOR = torch.Generator(device="cuda").manual_seed(87544357)

NEGATIVE_PROMPT = 'low quality, bad quality, sketches'

PROMPT = """The image depicts a modern, minimalist two-story residential building with a white exterior. Its cuboid shape features clean lines and sharp angles, creating a sleek look.
Large rectangular windows with dark frames punctuate both floors, some illuminated from within. A small balcony with thin black metal railings extends from the second floor.
An external black metal staircase leads to the upper entrance, adding visual interest. The building is part of a uniform row of similar structures on a gentle slope, ensuring unobstructed views for each unit.
The scene is captured during golden hour, with warm light casting subtle shadows that accentuate the geometric forms and give the white exterior a slight cream tint in places.
Well-maintained landscaping, including a manicured lawn with wildflowers and ornamental grasses, softens the stark architecture and integrates it with the natural surroundings."""
    
PROMPT1 = """Modern minimalist townhouses at sunset, featuring clean white cubic architecture with black metal staircases and railings. The buildings are arranged in a row on a gentle grassy slope. 
Warm evening light casts long shadows across the facades, with illuminated windows creating a cozy glow. The foreground shows wild flowers and ornamental grasses, including white dandelions and pampas grass.
Two people are visible: one climbing the exterior stairs and a child playing in the garden. The scene has a dreamy, nostalgic quality with soft natural lighting and lens flare effects.
Architectural visualization style with photorealistic rendering, shallow depth of field, and warm color grading."""
    
PROMPT2 = """Modern white townhouses arranged on a hillside at sunset. Minimalist cubic architecture with black metal staircases and balconies. Warm glowing windows and wild grasses with dandelions in the foreground.
Natural lens flare and soft evening lighting. Architectural visualization style with photorealistic rendering."""

PROMPT3=""""Architecture photography of a row of houses with a black railings on the balcony, white exterior, warm sunny day, natural lens flare. the houses are on a private street, surronded by a clean lawn"""

def get_control_image(model_name):
    """Select appropriate control image based on model name"""
    control_images = {
        'depth': ("control_image_depth.png", load_image(f"{INPUT_FOLDER}/control_image_depth.png")),
        'canny': ("control_image_edges.png", load_image(f"{INPUT_FOLDER}/control_image_edges.png")),
        'normals': ("control_image_normals.png", load_image(f"{INPUT_FOLDER}/control_image_normals.png"))
    }
    
    if 'depth' in model_name.lower():
        return control_images['depth']
    elif 'canny' in model_name.lower():
        return control_images['canny']
    elif 'lineart' in model_name.lower() or 'mistoline' in model_name.lower() or 'scribble' in model_name.lower():
        return control_images['canny']
    elif 'normals' in model_name.lower():
        return control_images['normals']
    else:
        print(f"Unknown control image for model: {model_name}")
        return None, None

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
        torch_dtype=torch.float16
    ).to("cuda")
    
    print(f"Loaded model: {controlnet_model}")
    print(f"Pipeline device map: {pipe.hf_device_map}")
    print(f"Controlnet device: {controlnet.device}")
    
    return pipe

def generate_image(pipe, control_image, prompt_text, conditioning_scale, num_steps,
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
        generator=GENERATOR,
    ).images[0]
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    base_name = f"{timestamp}_{image_index:04d}_c{conditioning_scale}_s{num_steps}_{scheduler}"

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
    
    # Parameter combinations
    prompts = [PROMPT3]
    conditioning_scales = [0.6, 0.7, 0.8, 0.9, 1.0]
    inference_steps = [20, 30, 40]
    # default should be first
    schedulers = ['default', 'dpmsolver', 'unipc', 'deis']
    # conditioning_scales = [0.8]
    # inference_steps = [30]

    # Calculate total combinations
    total_combinations = (
        len(prompts) *
        len(conditioning_scales) *
        len(inference_steps) *
        len(schedulers)
    )
    print(f"Total combinations to generate: {total_combinations}")

    # Generate all parameter combinations using itertools
    param_combinations = itertools.product(
        prompts,
        conditioning_scales,
        inference_steps,
        schedulers
    )
    
    model = MODELS[model_index]
    try:
        model_name = model.replace("/", "-")
        ensure_params_dir(model_name)

        pipe = load_pipeline(model)
        control_image_name, control_image = get_control_image(model)

        orig_config = pipe.scheduler.config
        for prompt_text, cond_scale, steps, scheduler in param_combinations:
            try:
                # the first value is actually the default scheduler - so we don't need to do anything
                if scheduler == 'dpmsolver':
                    pipe.scheduler = DPMSolverMultistepScheduler.from_config(orig_config,
                                                                             use_karras_sigmas=True, algorithm_type="sde-dpmsolver++")
                elif scheduler == 'unipc':
                    pipe.scheduler = UniPCMultistepScheduler.from_config(orig_config)
                elif scheduler == 'deis':
                    pipe.scheduler = DEISMultistepScheduler.from_config(orig_config)

                generate_image(
                    pipe,
                    control_image,
                    prompt_text,
                    cond_scale,
                    steps,
                    image_counter,
                    control_image_name,
                    model_name,
                    scheduler
                )

                image_counter += 1

            except Exception as e:
                print(f"Error generating image for {model} with params: {cond_scale}, {steps}")
                print(f"Error: {str(e)}")

        # clear gpu
        del pipe
        torch.cuda.empty_cache()
                
    except Exception as e:
        print(f"Error loading model {model}")
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
