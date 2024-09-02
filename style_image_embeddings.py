import torch
import os
from diffusers import AutoPipelineForText2Image, DDIMScheduler
from transformers import CLIPVisionModelWithProjection
from diffusers.utils import load_image

# 1. 이미지 인코더를 불러옵니다.
image_encoder = CLIPVisionModelWithProjection.from_pretrained(
    "h94/IP-Adapter",
    subfolder="models/image_encoder",
    torch_dtype=torch.float16,
)

# 2. 기반 모델을 불러옵니다.
# 여기서는 Text-to-Image 모델인 Stable Diffusion XL Base 모델을 불러옵니다.
pipeline = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    image_encoder=image_encoder
)

# 시간이 오래 걸리는 일은 아니라서 CUDA ID를 따로 지정하지 않았습니다.
pipeline = pipeline.to("cuda")

# 3. 스케줄러를 불러옵니다.
pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)

# 4. IP Adapter를 불러옵니다.
# 화풍 처리용 가중치를 사용합니다.
pipeline.load_ip_adapter(
  "h94/IP-Adapter",
  subfolder="sdxl_models",
  weight_name="ip-adapter-plus_sdxl_vit-h.safetensors"
)

# 가중치에 얼마나 비중을 둘것인지를 설정합니다.
pipeline.set_ip_adapter_scale(0.6)

# CPU 오프로딩을 사용해서 최적화합니다.
pipeline.enable_model_cpu_offload()

# 화풍 이미지가 저장된 경로를 명시합니다.
style_folder = "./sports_no_line"

# 이미지를 로드합니다.
# 전체 이미지를 사용하려고 했지만, 메모리 문제 때문에 일부만 사용하게 되었습니다.
style_images = [load_image(f"{style_folder}/{image_name}") for image_name in os.listdir(style_folder)][:15]

# 5. 임베딩을 생성합니다.

# ip_adapter_image는 사용할 이미지입니다.
# device는 CUDA로 사용합니다.
# num_images_per_prompt는 1로 지정했습니다. (튜토리얼의 설정을 따랐습니다.)
# do_classifier_free_guidance는 부정 프롬프트와 관련이 있습니다. 
# 해당 패러미터를 False로 지정할 시, 임베딩 파일을 사용할때 문제가 발생할수 있습니다.

image_embeds = pipeline.prepare_ip_adapter_image_embeds(
    ip_adapter_image=[style_images],
    ip_adapter_image_embeds=None,
    device="cuda",
    num_images_per_prompt=1,
    do_classifier_free_guidance=True,
)

# 생성한 임베딩을 ipadpt 파일로 저장합니다.
torch.save(image_embeds, "sports_image_embeds.ipadpt")

