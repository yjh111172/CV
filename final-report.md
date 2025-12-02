# 🚦 교통 표지판 인식을 통한 교통 안전 보조 시스템  
### 최종 프로젝트 보고서  
---

# 1. 프로젝트 개요

* 수행 학기: 2학기  
* 작성 날짜: 2025/11/11  
* 프로젝트명: **교통 표지판 인식을 통한 교통 안전 보조 시스템**

구분 | 성명 | 학번 | 소속학과 | GitHub
------|-------|-------|-------------|---------
1 | 김진형 | 20201708 | 데이터사이언스학과 | hsmu-jinhyeong
2 | 한영재 | 20191717 | 데이터사이언스학과 | yjh111172

* 지도교수: 데이터사이언스학과 김정은 교수님  

---

# 2. 프로젝트 내용

## 2.1 서론

교통 표지판은 운전자의 안전한 운전을 위해 필수적인 정보지만,  
- 운전자의 부주의  
- 피로  
- 시야 불량  
- 복잡한 도로 환경  

등으로 인해 표지판을 놓치는 일이 잦습니다.

본 프로젝트는:

- **딥러닝 기반 교통 표지판 이미지 분류 모델 개발**
- **실시간 카메라 영상 입력에 대한 인식 시스템 구축**
- **운전자 보조를 위한 경고/안내 기능 프로토타입 구현**

을 목표로 합니다.

---

# 2.2 추진 배경

## 2.2.1 개발 배경 및 필요성

- 교통사고의 주요 원인이 “시각적 정보 놓침”
- 초보·고령 운전자의 표지판 인식률이 낮음
- 상용 ADAS 시스템은 고가 옵션으로 접근성이 낮음
- 오픈소스 딥러닝 기반 저비용 교통 안전 보조 기술의 필요성 증가

## 2.2.2 선행 기술 및 사례 분석

- **Tesla Autopilot**: 카메라 기반 표지판 인식 기능  
- **Mobileye**: 도로 표지판 탐지 및 인식 분야 선도  
- **GTSRB 벤치마크**: ResNet 기반 모델로 98% 이상 정확도 보고  
- **YOLO 계열 모델**: 탐지(Detection) + 분류(Classification) 가능  

---

# 2.3 프로젝트 목표 및 수행 내용

## 2.3.1 프로젝트 목표

1. **GTSRB 데이터셋 기반 교통 표지판 인식 모델 구현**
2. **90% 이상의 분류 정확도 달성**
3. **실시간 영상 기반 표지판 인식 가능하도록 구현**
4. **시각적 안내 기능 제공**

---

## 2.3.2 전체 개발 흐름

[Dataset 수집]

[데이터 전처리 / 증강]

[ResNet18 기반 딥러닝 모델 학습]

[Test 성능 평가]

[실시간 카메라 인식 시스템 구현]

---


# 3. 데이터셋(GTSRB) 소개

- 43개 클래스 (제한속도, 금지, 경고 등)
- 약 50,000개 이미지
- Kaggle 제공 버전 사용
- 클래스별 폴더 구조 → `ImageFolder`로 바로 사용 가능


---

# 4. 이미지 전처리 및 데이터 증강

## 4.1 학습용 전처리

```
train_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])
```

## 4.2 테스트용 전처리
```
test_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])
```

---

# 5. 모델 구축(ResNet18 Fine-Tuning)
교통 표지판 인식 문제는 이미지 분류(Classification) 문제이므로, 다양한 CNN 기반 모델 중  
높은 정확도와 계산 효율성을 모두 갖춘 ResNet18을 기반으로 모델을 설계하였습니다.

## 5.1 모델 정의

```
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
in_features = model.fc.in_features
model.fc = nn.Linear(in_features, 43)

```
---

# 6. 모델 학습 과정
해당 단계는 교통 표지판 이미지로부터 유효한 특징을 스스로 추출하도록 만들기 위한 절차이며,GTSRB 데이터셋은 조명·회전·거리 등 다양한 변형이 존재해서 모델이 이러한 변화를 견디도록 충분한 학습이 필수적이라고 생각합니다.
## 6.1 학습 설정
```
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
```

## 6.2 학습 반복
```
for images, labels in train_loader:
    optimizer.zero_grad()
    outputs = model(images)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
```

---

# 7. 테스트 및 성능 평가
| 지표                  | 값          |
| ------------------- | ---------- |
| Train Accuracy      | 98~99%     |
| Validation Accuracy | 95~98%     |
| Test Accuracy       | **96~99%** |

---

# 8. 실시간 교통표지판 인식 시스템

## 8.1 처리 흐름

웹캠으로 프레임 캡처

전처리 및 Tensor 변환

ResNet18 모델로 분류

가장 높은 확률의 라벨 표시

OpenCV Overlay로 사용자에게 시각 표시

## 8.2 주요 코드

cv2.putText(
    frame, label_text, (20, 40),
    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2
)

## 8.3 결과

프레임 당 약 0.1~0.3초 속도로 실시간 인식

제한속도, 금지, 경고표지 등 대부분 정확하게 분류

---

# 9. 기대효과
## 9.1 사회적 효과

운전 중 표지판 인식 실패 감소 → 교통사고 감소

초보/고령 운전자 지원

## 9.2 경제적 효과

상용 ADAS 대비 저비용

연구·교육용 활용 가능

## 9.3 기술적 효과

딥러닝 기반 실시간 영상 인식 시스템 파이프라인 이해

실제 동작하는 TSR 기능 구현 경험 확보


---

# 10. 업무 분담
김진형 - 데이터셋 확보, 이미지 전처리

한영재 - 딥러닝 모델 구축 및 학습, 테스트 및 성능 평가

공통 - 실시간 입력, 영상 적용





# 참고 문헌

GTSRB Dataset

K. Stallkamp et al., “The German Traffic Sign Recognition Benchmark”

PyTorch

OpenCV

ResNet
