# Stable_Diffusion
Using pre-trained stable diffusion model to generate images from the text prompt and work with different parameters to generate refined images.

# Diffusion Explainer

Diffusion Explainer is an interactive visualization tool designed to help anyone learn how Stable Diffusion transforms text prompts into images. It runs in your browser, allowing you to experiment with several preset prompts without any installation, coding skills, or GPUs. Try Diffusion Explainer at https://poloclub.github.io/diffusion-explainer and watch a demo video on YouTube https://youtu.be/Zg4gxdIWDds!

# Research Paper

Diffusion Explainer: Visual Explanation for Text-to-image Stable Diffusion. Seongmin Lee, Benjamin Hoover, Hendrik Strobelt, Zijie J. Wang, ShengYun Peng, Austin Wright, Kevin Li, Haekyu Park, Haoyang Yang, Duen Horng Chau. Short paper, IEEE VIS 2024.

# To run locally

git clone https://github.com/poloclub/diffusion-explainer.git
cd diffusion-explainer
python -m http.server 8000

Then, on your web browser, access http://localhost:8000. You can replace 8000 with other port numbers you want to use.

# Credits

Led by Seongmin Lee, Diffusion Explainer is created by Machine Learning and Human-computer Interaction researchers at Georgia Tech and IBM Research. The team includes Seongmin Lee, Benjamin Hoover, Hendrik Strobelt, Jay Wang, ShengYun (Anthony) Peng, Austin Wright, Kevin Li, Haekyu Park, Alex Yang, and Polo Chau.
