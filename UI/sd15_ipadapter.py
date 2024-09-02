import torch
import os
from diffusers import AutoPipelineForText2Image, DDIMScheduler
from transformers import CLIPVisionModelWithProjection
from diffusers.utils import load_image
from datetime import datetime

image_encoder = CLIPVisionModelWithProjection.from_pretrained(
    "h94/IP-Adapter",
    subfolder="models/image_encoder",
    torch_dtype=torch.float16,
)

pipeline = AutoPipelineForText2Image.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
    image_encoder=image_encoder,
    safety_checker = None,
)
pipeline = pipeline.to("cuda")

pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
# pipeline.load_ip_adapter(
#   "h94/IP-Adapter",
#   subfolder="sdxl_models",
#   weight_name=["ip-adapter-plus_sdxl_vit-h.safetensors", "ip-adapter-plus-face_sdxl_vit-h.safetensors"]
# )

pipeline.load_ip_adapter(
  "h94/IP-Adapter",
  subfolder="models",
  weight_name=["ip-adapter_sd15.safetensors", "ip-adapter-plus-face_sd15.safetensors"]
)
pipeline.set_ip_adapter_scale([0.7, 0.3])
pipeline.enable_model_cpu_offload()

input_data_path = "./uploads/input_file"
style_data_path = "./uploads/style_files"
output_data_path = "./static/gen_images"

prompt = "a person"
with open("prompt.txt", "r", encoding="utf-8") as prompt_txt:
    prompt = prompt_txt.read()

face_image = [load_image(f"{input_data_path}/{image_name}") for image_name in os.listdir(input_data_path)][0]
style_images = [load_image(f"{style_data_path}/{image_name}") for image_name in os.listdir(style_data_path)]

generator = torch.Generator(device="cpu").manual_seed(0)

for i in range(4):
    image = pipeline(
        prompt=prompt,
        ip_adapter_image=[style_images, face_image],
        negative_prompt="low quality",
        num_inference_steps=50, num_images_per_prompt=1,
        generator=generator,
    ).images[0]

    image.save(f"{output_data_path}/{datetime.now().date()}_{datetime.now().time()}.jpg")