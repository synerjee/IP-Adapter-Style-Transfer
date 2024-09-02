import os
import json
import shutil
import subprocess

# FID 점수를 계산하는 함수입니다.

def show_fid(count):
    """
    
    생성한 이미지의 횟수를 입력받아, FID 점수를 출력하는 함수입니다.

    Args:
        count: 이미지 생성 횟수입니다. 예를 들어, 10장을 생성하셨다면 10으로 입력하시면 됩니다.

    """

    # 생성한 이미지 횟수를 보여줍니다.
    print(f"FID Score / # of Gen. Img. = {count}")

    # 사용했던 입력 이미지의 이름들입니다.
    faces = ["slamdunk", "homer_simpson", "faker"]

    # 입력 이미지가 저장된 폴더입니다.
    input_dir = "./input_data"

    # 생성 이미지가 저장된 폴더입니다.
    # 저는 예를 들어 10장을 생성하면, saved_images_10 이라는 폴더에 저장했습니다.
    # saved_images_10 폴더 안에는 각 입력 이미지 이름으로 된 서브폴더를 만들어서 
    # 알맞는 이미지를 저장했습니다.
    output_dir = f"./saved_images_{count}"

    # FID 점수 계산을 용이하게 하기위해 temp 폴더를 만들었습니다.
    temp_dir = f"./saved_images_{count}/temp"

    # FID 점수를 저장할 딕셔너리입니다.
    fid = {}

    # 각 입력 이미지마다 FID 점수를 계산합니다.
    for face in faces:

        # 딕셔너리에 입력 이미지 이름이 없다면 초기화합니다.
        if (face not in fid):
            fid[face] = 0

        # 입력 이미지가 있는 폴더입니다.
        face_dir = f"{input_dir}/{face}"

        # 입력 이미지에 대응하는 생성 이미지가 있는 폴더입니다.
        gen_dir = f"{output_dir}/{face}"
        
        for image in os.listdir(gen_dir):
            # FID 점수는 두 이미지 데이터셋의 크기가 동일해야 하기 때문에, 
            # 저는 한 장씩 비교하도록 했습니다.
            # 생성 이미지 한 장씩 temp 폴더로 복사했습니다.
            image_path = f"{gen_dir}/{image}"
            shutil.copy2(image_path, temp_dir)

            # Subprocess 라이브러리로 명령어를 수행할수 있습니다.
            # FID 점수 계산이 끝나면 temp 폴더 내의 파일을 삭제합니다.
            command = f"python -m pytorch_fid {face_dir} {temp_dir}; rm {temp_dir}/{image}"
            ret = subprocess.run(command, capture_output=True, shell=True)

            # FID 점수를 합산합니다.
            fid[face] += float(ret.stdout.decode().split("FID:  ")[1])
        
        # FID의 평균을 구하기 위해 합산 값을 생성 이미지 수로 나눕니다.
        fid[face] /= count
    
    # FID 점수를 출력합니다.
    print(fid)

# 저는 10장, 20장, 30장 씩 생성했으므로, 각 생성 숫자마다 FID 값을 계산합니다.
counts = [10, 20, 30]

for count in counts:
    show_fid(count)
        