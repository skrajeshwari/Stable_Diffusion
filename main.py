# Installing Dependencies

# xformers for memory optimization https://github.com/facebookresearch/xformers

# Run in Bash

#Create and activate virtual environment
#python -m venv proj_venv
#source proj_venv/bin/activate

#pip install -q diffusers==0.11.1
#pip install -q accelerate transformers ftfy bitsandbytes==0.35.0 gradio natsort safetensors xformers
#pip install huggingface-hub>=0.26.0,<1.0
#pip install matplotlib
#pip install jax jaxlib
#pip install transformers -U
# go to the diffusers path to locate the file dynamic_modules_utils.py
# cd /workspaces/Stable_Diffusion
# cd proj_venv/lib/python3.12/site-packages/diffusers/
# changing the file dynamic_modules_utils.py, using following command
# vim /workspaces/Stable_Diffusion/proj_venv/lib/python3.12/site-packages/diffusers/dynamic_modules_utils.py



# Import required packages

from PIL import Image
import matplotlib.pyplot as plt

import torch    #PyTorch
import jax
jax.random.KeyArray = jax.Array           # since jax.random.KeyArray is deprecated
from diffusers import StableDiffusionPipeline

import warnings
warnings.filterwarnings("ignore")

# MODELS

# CompVis/stable-diffusion-v1-4
# We can define with little effort a pipeline to use the Stable Diffusion model, through the StableDiffusionPipeline.
# The checkpoint used here is the 'CompVis/stable-diffusion-v1-4'

# Load SDv1.4
pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
pipe = pipe.to('cuda')  # Use GPU

# Memory optimization, specially on colab
pipe.enable_attention_slicing()
pipe.enable_xformers_memory_efficient_attention()

# runwayml/stable-diffusion-v1-5

# Load SDv1.5
sd15 = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
sd15 = sd15.to('cuda')
sd15.enable_attention_slicing()
sd15.enable_xformers_memory_efficient_attention()

# stabilityai/stable-diffusion-2-1

# sd2 = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1")
# sd2 = sd2.to("cuda")
# sd2.enable_attention_slicing()
# sd2.enable_xformers_memory_efficient_attention()

# nitrosocke/mo-di-diffusion
# Modern Disney style- https://huggingface.co/nitrosocke/mo-di-diffusion

# modi = StableDiffusionPipeline.from_pretrained("nitrosocke/mo-di-diffusion")
# modi = modi.to("cuda")
# modi.enable_attention_slicing()
# modi.enable_xformers_memory_efficient_attention()

# IMAGE GENERATION

#Single Image Generation

prompt = 'a tiger standing on grass land'
img = pipe(prompt).images[0]
img

# Multiple images

num_imgs = 3
prompt = 'photograph of a royal enfield'
imgs = pipe(prompt, num_images_per_prompt= num_imgs).images

# Function for plotting multiple images as one

def grid_img(imgs, rows=1, cols=3, scale=1):
  assert len(imgs) == rows * cols

  w, h = imgs[0].size
  w, h = int(w * scale), int(h * scale) # reduce the size : when s<1 ; 0.75 * 512 = 384

  grid = Image.new('RGB', size = (cols * w, rows * h)) # creating gird (like blank canvas)
  grid_w, grid_h = grid.size

  for i, img in enumerate(imgs): # looping throuhg images
    img = img.resize((w, h), Image.Resampling.LANCZOS) # second parameter preserve the image appearance
    grid.paste(img, box=(i % cols * w, i // cols * h)) # pasting the image on grids

  return grid

grid = grid_img(imgs, rows=1, cols=3, scale=0.75)
grid

# Same image on every run

# Set the seed value to generate same image for same input prompts
seed = 777
generator = torch.Generator('cuda').manual_seed(seed)

prompt = "photograph of a royal enfield'"
num_imgs = 3

imgs = pipe(prompt, num_images_per_prompt=num_imgs, generator=generator).images
grid = grid_img(imgs, rows=1, cols=3, scale=0.75)
grid

### This code cell uses Modern-Disney Stable diffusion
### Restart the runtime and load "nitrosocke/mo-di-diffusion" pipeline ro run this cell.


# prompt = ["albert einstein, modern disney style",
#           "modern disney style rolls-royce in the desert, golden hour",
#           "modern disney style helicopter"]

# seed = 42
# print("Seed: ".format(str(seed)))
# generator = torch.Generator("cuda").manual_seed(seed)
# imgs = modi(prompt, generator=generator).images

# grid = grid_img(imgs, rows=1, cols=3, scale=0.75)
# grid

# TUNING PARAMETERS

#Inference Steps

prompt = "a photograph of an astronaut riding a horse"
seed = 1234 #It is not mandatory to change the seed, you can use the same seed through out
generator = torch.Generator("cuda").manual_seed(seed)
num_inference_steps = 20

img = pipe(prompt, num_inference_steps=num_inference_steps, generator=generator).images[0]
img

# Multiple inference steps

num_inference_steps_lst = [10,20,30,40,50]
plt.figure(figsize=(18,8))
for i in range(len(num_inference_steps_lst)):
    n_steps = num_inference_steps_lst[i]
    generator = torch.Generator('cuda').manual_seed(seed)
    img = pipe(prompt, num_inference_steps=n_steps, generator=generator).images[0]

    plt.subplot(1, 5, i+1)
    plt.title('num_inference_steps: {}'.format(n_steps))
    plt.imshow(img)
    plt.axis('off')
plt.show()

# Guidance scale (CFG)

prompt = "a photograph of an astronaut riding a horse"
seed = 1234 #It is not mandatory to change the seed, you can use the same seed through out
generator = torch.Generator("cuda").manual_seed(seed)
guidance_scale = 10

img = pipe(prompt, guidance_scale=guidance_scale, generator=generator).images[0]
img

# Multiple CFG Values

guidance_scale_lst = [5,6,7,8,9]

plt.figure(figsize=(18,8))

for i in range(len(guidance_scale_lst)):
    n_guidance = guidance_scale_lst[i]
    generator = torch.Generator("cuda").manual_seed(seed)
    img = pipe(prompt, guidance_scale=n_guidance, generator=generator).images[0]

    plt.subplot(1, 5, i+1)
    plt.title('guidance_scale: {}'.format(n_guidance))
    plt.imshow(img)
    plt.axis('off')

plt.show()

# IMAGE DIMENSIONS

#Landscape Mode

seed = 777
prompt = "photograph of a mountain landscape during sunset, stars in the sky"
generator = torch.Generator("cuda").manual_seed(seed)
h, w = 512, 768

img = pipe(prompt, height=h, width=w, generator=generator).images[0]
img

# Portrait Mode

generator = torch.Generator("cuda").manual_seed(seed)
h, w = 768, 512

img = pipe(prompt, height=h, width=w, generator=generator).images[0]
img

# Negative Prompt

num_images = 3
prompt = 'photograph of an old car'
neg_prompt = 'bw photo'                  # black and white photo

imgs = pipe(prompt, negative_prompt = neg_prompt, num_images_per_prompt= num_images).images
grid = grid_img(imgs, rows= 1, cols= 3, scale=0.75)
grid

# Changing the Scheduler

# Displaying the scheduler attributes
sd15.scheduler

# Getting other compatible scheduler with the algorithm
sd15.scheduler.compatibles

# Checking the current config
sd15.scheduler.config

# Changing the scheduler to DDIMScheduler

from diffusers import DDIMScheduler
sd15.scheduler = DDIMScheduler.from_config(sd15.scheduler.config)

generator = torch.Generator(device = 'cuda').manual_seed(seed)
img = sd15(prompt, negative_prompt=neg_prompt, generator=generator).images[0]
img
