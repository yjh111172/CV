from transformers import pipeline
from PIL import Image
import requests
import matplotlib.pyplot as plt

# Transformer의 Attention 메커니즘을 이용해 이미지의 전역적 문맥을 파악하여 깊이를 추정하는 실습

# 이미지 불러오기
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

# Depth Anything 모델 사용
# Hugging Face Hub에서 가중치 다운로드
print("Depth Estimation 모델 로딩...")
depth_estimator = pipeline(task="depth-estimation", model="LiheYoung/depth-anything-small-hf")

# 추론 및 깊이 맵 저장
result = depth_estimator(image)
depth_map = result["depth"]

# 결과 시각화
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].imshow(image)
ax[0].set_title("Input RGB Image")
ax[0].axis("off")

ax[1].imshow(depth_map, cmap='inferno')
ax[1].set_title("Predict Depth Map")
ax[1].axis("off")
plt.show()