# What are Diffusion Models?

Diffusion Models are generative models, meaning that they are used to generate data similar to the data on which they are trained. 
Fundamentally, Diffusion Models work by destroying training data through the successive addition of Gaussian noise, and then learning to recover the data by reversing this noising process. 
After training, we can use the Diffusion Model to generate data by simply passing randomly sampled noise through the learned denoising process.

# How does it work?

The image is asymptotically transformed to pure Gaussian noise. The goal of training a diffusion model is to learn the reverse process.

# Diffusion models 

These are a class of generative models that simulate the process of data generation through a gradual diffusion process.

# Stable Diffusion

This is a specific implementation of a diffusion model developed for high-quality image generation. 
It enhances the standard diffusion model framework with techniques like latent space representation, allowing for more efficient processing and the generation of high-resolution images.
Stable Diffusion also incorporates elements like conditioning on textual input, enabling it to create images based on textual descriptions.

More details:

Stable Diffusion is a deep learning model for image generation, released in 2022. It is based on a particular type of diffusion model called Latent Diffusion, proposed in the paper High-Resolution Image Synthesis with Latent Diffusion Models. 
They are generative models, designed to create new data similar to the data it was trained on. In the case of Stable Diffusion, the data is images and trained to reduce random Gaussian noise.

The Stable Diffusion algorithm was developed by Compvis (the Computer Vision research group at the Ludwig Maximilian University of Munich) and sponsored primarily by the startup Stability AI.

The algorithm is based on ideas from DALL-E 2 (developed by Open AI, which is also the creator of ChatGPT), Imagen (from Google), and others image generation models.

# Types of image generation through diffusion models:

Unconditional image generation: the model generates images without any additional condition like text or image. 
You will get images similar to those provided in the training set.

The generation of images conditioned by text is known as text-to-image, or text2img. 
The prompt is converted into embeddings that are used to condition the model to generate an image from noise.

The generation of images based on other image is known as image-to-image, or img2img. 
In addition to the text prompt, it allows sending an initial image to condition the generation of new images. You have more control over the final composition.

Inpainting allows selecting a specific part of the image to change the class/concept, or even removing it from the scene.

# Main components of Stable diffusion

The main components include:

1. Autoencoder (VAE)
2. U-Net
3. Text-encoder
4. CLIP (Contrastive Language-Image Pre-training)

* Stable Diffusion receives a latent seed and a text prompt as input.
* The seed is used to generate random representations of latent images of size 64x64.
* The text prompt is transformed into text embeddings of size 77x768 using CLIP text encoder.
* U-Net iteratively reduces noise from the random latent image representations while conditioning on the text embeddings.
* The U-Net output (noise residual) is used to compute a denoised latent image representation using a scheduler algorithm.
* The denoising process is repeated  x  times to recover the best latent image representations.
* Once completed, the latent image representation is decoded by the decoder part of the VAE (Variational Autoencoder).

# Parameters

There are many other parameters like generator that can be passed inside the StableDiffusionPipeline to improve the result.

For example:

1. Inference steps (num_inference_steps)
2. Guidance scale (guidance_scale)
3. Image size (height and width dimensions)
4. Negative prompt (negative_prompt)

# Inference steps (num_inference_steps)
The more the steps, the better the results but the longer it takes to generate the image. It is also known as denoising steps, as it indicates the number of steps required to turn the image from complete noise (initial state) into the result.

Stable Diffusion works very well with a relatively small number of steps, so we recommend using the default value of 50 inference steps. The more steps the better the result, but there comes a point where the image stops improving.
For faster results, use a smaller number.
The defaut number of steps varies according to the scheduler algorithm

# Guidance scale (CFG)
The classifier-free guidance (CFG - also known as guidance scale) is a way to increase the adherence to the conditional prompt that guides the generation, as well as the overall quality of the image. It controls how much the prompt will be taken into account for conditioning the diffusion process.

Smaller values: the more the prompt is ignored. For example, if the value is set to 0 then the image generation is unconditioned.
Higher values: returns images that better represent the prompt
Choosing the best value

Values between 7 and 8.5 are generally good choices. The default value is 7.5

In general, you can keep the value in the range from 5 to 9. Less realistic images will be returned if the values are too low or too high

The best value depends on the desired results and the complexity of the prompt.

# Image size (dimensions)
The generated images are 512 x 512 pixels

Recommendations in case you want other dimensions:

make sure the height and width are multiples of 8
less than 512 will result in lower quality images
exceeding 512 in both directions (width and height) will repeat areas of the image ("global coherence" is lost)

# Negative prompt (negative_prompt)
The negative prompt is an additional way of telling what you don't want in the image, what you'd like to avoid.

It could be used to remove objects from the image or fix defects.
It is optional in the first versions of Stable Diffusion; however, in the latest versions it has become important to generate quality images.
Some images can only be generated using negative prompts.
During the text-to-image conditioning step, the prompt is converted into embedding vectors, which will feed the U-Net noise predictor.

There are two sets of embedding vectors: one for the positive prompt and one for the negative prompt.
Both the positive and negative prompts have 77 tokens.

# Changing the Scheduler (Sampler)
The model is usually not trained to directly predict a slightly less noisy image, but rather to predict the 'noise residual,' which is the difference between a less noisy image and the input image or, similarly, the gradient between the two time steps.

To do the denoising process, a specific noise scheduling algorithm is thus necessary and "wrap" the model to define how many diffusion steps are needed for inference as well as how to compute a less noisy image from the model's output. Here is where the different schedulers of the diffusers library come into play.

Scheduler algorithms (also called samplers) calculate the predicted denoised image representation from the previous noise representation and the predicted noise residual. Determines how the image is calculated. There are several different algorithms.

Some examples commonly used with Stable Diffusion :

PNDM (default)
DDIM Scheduler
K-LMS Scheduler
Euler Ancestral Discrete Scheduler (Euler A)
DPM Scheduler
Since these samplers are different mathematically, they will diverge and can eventually converge if the settings are correct.

For technical details on theory and mathematics of the scheduler algorithms, refer to this paper(https://arxiv.org/abs/2206.00364): Elucidating the Design Space of Diffusion-Based Generative Models.

For comparison of different samplers, refer ( https://www.artstation.com/blogs/kaddoura/pBPo/stable-diffusion-samplers ).

To know more about available schedulers on HuggingFace, refer (https://huggingface.co/docs/diffusers/using-diffusers/schedulers#schedulers-summary).

Default is PNDMScheduler( https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_pndm.py).
