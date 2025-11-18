import cv2 as cv
import mediapipe as mp

# MediaPipe 모듈 설정
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils # 검출 결과 그리는 데 사용할 모듈
mp_styles = mp.solutions.drawing_styles # 그리는 유형 지정

pose = mp_pose.Pose(
    static_image_mode=False,
    enable_segmentation=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# 파일 경로 지정
video_file_path = 'data/purchacing.mp4'
cap = cv.VideoCapture(video_file_path)

# 파일 열기 성공 여부 확인
if not cap.isOpened():
    print(f"오류: 동영상 파일 '{video_file_path}'을 열 수 없습니다.")
    exit()

while True:
    ret, frame = cap.read()

    # 동영상의 끝에 도달했을 때의 처리
    if not ret:
        print('동영상의 마지막 프레임에 도달하여 루프를 종료합니다.')
        break  # 파일의 끝에 도달하면 루프 종료

    # 자세 추정 수행
    res = pose.process(cv.cvtColor(frame, cv.COLOR_BGR2RGB)) # 프레임 처리 (BGR -> RGB 변환)
    #print(res.pose_landmarks)

    # 랜드마크 그리기
    mp_drawing.draw_landmarks(
        frame,
        res.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_styles.get_default_pose_landmarks_style()
    )

    cv.imshow('MediaPipe pose', frame)


    if cv.waitKey(1) == ord('q'):
        # 'q' 키를 눌렀을 때 3D 랜드마크 플롯 생성
        if res.pose_world_landmarks:
            mp_drawing.plot_landmarks(res.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)
        break

# 작업 후 해제 및 창 닫기
cap.release()
#cv.waitKey()
cv.destroyAllWindows()