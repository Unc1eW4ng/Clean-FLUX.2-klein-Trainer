import torch
from diffusers import Flux2KleinPipeline
from diffusers.utils import load_image


pipe = Flux2KleinPipeline.from_pretrained("black-forest-labs/FLUX.2-klein-4B", torch_dtype=torch.bfloat16)
print("Loaded base model successfully.")
pipe.load_lora_weights("pytorch_lora_weights.safetensors")
print("Loaded LoRA weights successfully.")
pipe.to("cuda")
print("Pipeline moved to CUDA successfully.")
input_image1 = load_image("../guitar.jpg")
input_image2 = load_image("../singer.jpg")

print("Input image loaded successfully.")
input_image1 = input_image1.resize((512,512))  # Resize the image to match the model's input size
input_image2 = input_image2.resize((512,512))  

image = pipe(
  image=[input_image1,input_image2],
  prompt="An undivided, seamless, and harmonious picture with two objects. On the beach , A toy singer and A guitar with an expressive sound are placed together",
  guidance_scale=1,
  width = 512,
  height = 512,
).images[0]

image.save("output.png")
