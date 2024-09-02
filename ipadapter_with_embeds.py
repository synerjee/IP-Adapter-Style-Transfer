import torch
import os
from diffusers import AutoPipelineForText2Image, DDIMScheduler
from transformers import CLIPVisionModelWithProjection
from diffusers.utils import load_image
from input_image_embeddings import save_input_embeddings

# 1. 이미지 인코더를 불러옵니다.
# 이미 이미지 임베딩이 있어도, 이미지 인코더를 불러와야 하는 이유는
# 저희의 기반 모델이 Text-to-Image 모델이어서 이미지 프롬프팅 기능을 추가하기 위해
# 이미지 인코더를 불러오는 것입니다.
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
    image_encoder=image_encoder,
)
# 저희가 GPU를 다른 팀과 공유하고 있기 때문에 CUDA:0으로 지정했습니다.
pipeline = pipeline.to("cuda:0")

# 3. 스케줄러를 불러옵니다.
pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)

# 4. IP Adapter를 불러옵니다.
# 이번에 사용할 가중치는 두 개인데, 
# 첫번째는 화풍 이미지 임베딩을, 두번째는 입력 이미지 임베딩을 처리하기 위함입니다.
# 임베딩 파일은 이미 있으니 image_encoder_folder는 None으로 지정합니다.
pipeline.load_ip_adapter(
  "h94/IP-Adapter",
  subfolder="sdxl_models",
  weight_name=["ip-adapter-plus_sdxl_vit-h.safetensors", "ip-adapter-plus-face_sdxl_vit-h.safetensors"],
  image_encoder_folder = None
)

# 가중치에 얼마나 비중을 둘것인지를 설정합니다.
# 첫번째 가중치에 좀 더 비중을 둔 것을 확인하실수 있습니다.
pipeline.set_ip_adapter_scale([0.7, 0.3])

# CPU 오프로딩을 사용해서 최적화합니다.
pipeline.enable_model_cpu_offload()

# 입력 이미지의 파일 이름을 불러옵니다.
# 예를 들어, "슬램덩크.jpg" 에서 "슬램덩크" 부분을 추출합니다.
faces = [filename.split(".")[0] for filename in os.listdir("./input_data")]

# 각 입력 이미지에 대응하는 프롬프트입니다.
prompt_dict = {
    "slamdunk": "basketball player",
    "homer_simpson": "Homer Simpson",
    "faker": "Korean guy with glasses"
}

# 미리 만들어 둔 화풍 이미지의 임베딩 파일을 불러옵니다.
style_image_embeds = torch.load("sports_image_embeds.ipadpt")

# 5. 입력 이미지마다 화풍 변환 작업을 수행합니다.
for face in faces:
    # 입력 이미지가 저장된 경로를 명시합니다.
    face_image_path = f"./input_data/{face}/{face}.jpg"

    # 입력 이미지의 임베딩 파일을 생성합니다.
    # 생성한 파일은 "./embed_10"이라는 폴더에 저장합니다. 폴더는 미리 만드셔야 합니다.
    # 폴더명 및 경로는 수정하실수 있습니다.
    # 해당 함수의 리턴 값은 입력 이미지 임베딩 파일이 저장된 경로입니다.
    face_image_embeds_path = save_input_embeddings(face, face_image_path, "./embed_10")

    # 입력 이미지 임베딩 파일을 불러옵니다.
    face_image_embeds = torch.load(face_image_embeds_path)

    # 생성 이미지를 저장할 경로를 지정합니다.
    save_dir = f"./saved_images_10/{face}"

    # 난수 생성자를 정의합니다.
    generator = torch.Generator(device="cpu").manual_seed(0)

    # 여기서는 10장씩 생성하는데, 수정하셔도 됩니다.
    for i in range(10):

        # prompt와 negative_prompt는 각각 긍정, 부정 프롬프트입니다.

        # ip_adapter_image_embeds로 이미지 임베딩 파일을 불러옵니다.
        # 각 임베딩 파일은 파이썬 배열안에 한 개의 Tensor 오브젝트가 있는 식으로 구성되어 있습니다. (즉, [ Tensor() ])
        # 여기서는 두 개의 IP Adapter 가중치를 사용하므로, 두 개의 Tensor가 필요합니다. (즉, [ Tensor(), Tensor() ])

        # num_inference_steps는 추론 스텝입니다. 50이 적정한 정도여서 사용했습니다.

        # 원래는 num_images_per_prompt를 사용해서 여러 장의 이미지를 한꺼번에 생성하려 했지만,
        # 메모리 오류가 발생해서 for 문으로 문제를 우회했습니다.

        # generator는 방금 정의한 난수 생성기입니다.
        
        image = pipeline(
            prompt=f"{prompt_dict[face]}",
            ip_adapter_image_embeds=[style_image_embeds[0], face_image_embeds[0]],
            negative_prompt="low quality",
            num_inference_steps=50, num_images_per_prompt=1,
            generator=generator,
        ).images[0]

        # 이렇게 생성한 이미지를 저장합니다.
        image.save(f"{save_dir}/{face}_{i+1}.jpg")


