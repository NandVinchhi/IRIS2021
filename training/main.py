import cv2
import mediapipe
drawingModule = mediapipe.solutions.drawing_utils
handsModule = mediapipe.solutions.hands

print(handsModule.HandLandmark.__dict__.keys())
poseModule = mediapipe.solutions.pose
mp_drawing = mediapipe.solutions.drawing_utils
mp_drawing_styles = mediapipe.solutions.drawing_styles
capture = cv2.VideoCapture(0)
frameWidth = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
frameHeight = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)

hands = handsModule.Hands(static_image_mode=False, min_detection_confidence=0.7, min_tracking_confidence=0.7, max_num_hands=2)
pose = poseModule.Pose(min_detection_confidence=0.5,min_tracking_confidence=0.5)

while (True):
    ret, frame = capture.read()
    final_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results_hands = hands.process(final_frame)
    results_pose = pose.process(final_frame)

    if results_hands.multi_hand_landmarks:
      for hand_landmarks in results_hands.multi_hand_landmarks:

        
        mp_drawing.draw_landmarks(
            frame,
            hand_landmarks,
            handsModule.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())

    mp_drawing.draw_landmarks(frame, results_pose.pose_landmarks, poseModule.POSE_CONNECTIONS, landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
    cv2.imshow('Sign Language Translator', frame)
    if cv2.waitKey(1) == 27:
        break
cv2.destroyAllWindows()
capture.release()