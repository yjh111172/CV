import torch

# torch.cuda.amp.autocast가 호출될 때, 강제로 일반 autocast로 연결
class DeviceFriendlyAutocast(torch.amp.autocast):
    def __init__(self, enabled=True, **kwargs):
        # 강제로 device_type을 'cpu'나 'mps'로 고정
        if torch.cuda.is_available():
            super().__init__(device_type='cuda', enabled=enabled, **kwargs)
        elif torch.backends.mps.is_available():
            super().__init__(device_type='mps', enabled=enabled, **kwargs)
        else:
            super().__init__(device_type='cpu', enabled=enabled, **kwargs)

# 기존 CUDA autocast를 로컬 환경에 맞게 변환
torch.cuda.amp.autocast = DeviceFriendlyAutocast

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images

# Device 설정 : CUDA, MPS, CPU
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print(f"Using device: {device}")

# dtype 설정: MPS는 보통 float16 사용
if device == "cuda":
    # Ampere GPU 이상이면 bfloat16, 아니면 float16
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
elif device == "mps":
    dtype = torch.float16
else:
    dtype = torch.float32

print(f"Using dtype: {dtype}")

# 모델 초기화 및 가중치 로드
model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)

print(model)

# 이미지 로드 및 전처리
image_names = ["data/IMG_5175.png", "data/IMG_5176.png", "data/IMG_5177.png", "data/IMG_5178.png"]
images = load_and_preprocess_images(image_names).to(device)

print(f"Loaded {len(image_names)} images")

with torch.no_grad():
    # Autocast 설정: torch.cuda.amp 대신 범용 torch.autocast 사용해서 여러 device 지원
    # PyTorch 2.0 이상부터 device_type='mps' 지원
    #with torch.autocast(device_type=device, dtype=dtype):
    with torch.amp.autocast(device_type=device, dtype=dtype):
        # 3D 속성 예측 (카메라, 깊이 맵, 포인트 맵 등)
        predictions = model(images)

print(predictions)