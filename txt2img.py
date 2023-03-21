# Script to run txt2img without Cog
from diffusers import StableDiffusionPipeline
from diffusers import EulerAncestralDiscreteScheduler
from diffusers import DPMSolverMultistepScheduler
import torch
from compel import Compel

model_id = "antonioglass/dlbrt"

pipe = StableDiffusionPipeline.from_pretrained(model_id)
compel = Compel(tokenizer=pipe.tokenizer, text_encoder=pipe.text_encoder)

# change scheduler
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

# enable xformers
pipe.enable_xformers_memory_efficient_attention()

prompt = "a red cat playing with a ball--"
negative_prompt = "deformed, distorted, disfigured, poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, mutated hands and fingers, disconnected limbs, mutation"
height = 512
width = 512
num_inference_steps = 20
guidance_scale = 6.5
seed = 55

conditioning = compel.build_conditioning_tensor(prompt)
negative_conditioning = compel.build_conditioning_tensor(negative_prompt)

pipe = pipe.to("cuda")
generator = torch.Generator("cuda").manual_seed(seed)

image = pipe(prompt_embeds=conditioning, negative_prompt_embeds=negative_conditioning, height=height, width=width, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale,  generator=gener>
image.save("output.png")