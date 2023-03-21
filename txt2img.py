# Script to run txt2img without Cog

from diffusers import StableDiffusionPipeline
from diffusers import EulerAncestralDiscreteScheduler
import torch

model_id = "XpucT/Deliberate"

pipe = StableDiffusionPipeline.from_pretrained(model_id)

# change scheduler
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
pipe.enable_xformers_memory_efficient_attention()

prompt = "a cute kitten made out of metal, (cyborg:1.1), ([tail | detailed wire]:1.3), (intricate details), hdr, (intricate details, hyperdetailed:1.2), cinematic shot, vignette, centered"
negative_prompt = "(deformed, distorted, disfigured:1.3), poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, (mutated hands and fingers:1.4), disconnected limbs, mutation)
height = 768
width = 512
num_inference_steps = 26
guidance_scale = 6.5
seed = 1791574510

pipe = pipe.to("cuda")
generator = torch.Generator("cuda").manual_seed(seed)

image = pipe(prompt=prompt, negative_prompt=negative_prompt, height=height, width=width, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale,  generator=generator).images[0]
image.save("output_1.png")
