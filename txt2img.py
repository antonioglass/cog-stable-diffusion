# Script to run txt2img without Cog

from diffusers import StableDiffusionPipeline
from diffusers import EulerAncestralDiscreteScheduler
from diffusers import DPMSolverMultistepScheduler
import torch

model_id = "./deliberate-v2"

pipe = StableDiffusionPipeline.from_pretrained(model_id)

# change scheduler
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_xformers_memory_efficient_attention()

prompt = "a closeup portrait of a playful maid, undercut hair, apron, amazing body, pronounced feminine feature, busty, kitchen, [ash blonde | ginger | pink hair], freckles, flirting with camera"
negative_prompt = "(deformed, distorted, disfigured:1.3), poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, (mutated hands and fingers:1.4), disconnected limbs, mutation"
height = 1024
width = 768
num_inference_steps = 30
guidance_scale = 6.5
seed = 1804518985

pipe = pipe.to("cuda")
generator = torch.Generator("cuda").manual_seed(seed)

# image = pipe(prompt=prompt, negative_prompt=negative_prompt, height=height, width=width, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale,  generator=generator).images[0]
image = pipe(prompt=prompt, negative_prompt=negative_prompt, height=height, width=width, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale).images[0]
image.save("output.png")