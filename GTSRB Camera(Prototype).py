import cv2
import torch
from torchvision import transforms, models
from torch import nn
from pathlib import Path
import numpy as np


MODEL_PATH = Path("../models/gtsrb_resnet18_final.pth")


# 모델 로드


def load_model(device):
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    class_names = checkpoint["class_names"]
    img_size = checkpoint["img_size"]

    model = models.resnet18(weights=None) 
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    preprocess = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5]),
    ])

    return model, preprocess, class_names


@torch.no_grad()
def predict_frame(model, preprocess, frame_bgr, device):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    img = preprocess(
        cv2_to_pil_like(frame_rgb)
    )
    img = img.unsqueeze(0).to(device)  # (1, C, H, W)

    outputs = model(img)
    probs = torch.softmax(outputs, dim=1)
    conf, pred = torch.max(probs, dim=1)
    return pred.item(), conf.item()


def cv2_to_pil_like(img_rgb):
    from torchvision.transforms.functional import to_pil_image
    return to_pil_image(img_rgb)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model, preprocess, class_names = load_model(device)

    cap = cv2.VideoCapture(0)  # 0번 웹캠

    if not cap.isOpened():
        print("❌ 카메라를 열 수 없습니다.")
        return

    print("실시간 교통 표지판 인식 시작! (종료: q 키)")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("프레임을 읽을 수 없습니다.")
            break

        pred_id, conf = predict_frame(model, preprocess, frame, device)

        label_text = f"{class_names[pred_id]} ({conf*100:.1f}%)"

        cv2.putText(frame, label_text, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        cv2.imshow("Traffic Sign Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
