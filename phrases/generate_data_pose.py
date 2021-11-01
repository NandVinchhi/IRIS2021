import cv2
import mediapipe
import time
import math
from csv import writer

poseModule = mediapipe.solutions.pose
handsModule = mediapipe.solutions.hands
mp_drawing = mediapipe.solutions.drawing_utils
mp_drawing_styles = mediapipe.solutions.drawing_styles
capture = cv2.VideoCapture(0)
frameWidth = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
frameHeight = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)

hands = handsModule.Hands(static_image_mode=False, min_detection_confidence=0.7, min_tracking_confidence=0.7, max_num_hands=2)
pose = poseModule.Pose(min_detection_confidence=0.5,min_tracking_confidence=0.5)

print(handsModule.HandLandmark.__dict__.keys())

def calculateAngle(a, b, c):
    x1 = a.x
    y1 = a.y
    x2 = b.x
    y2 = b.y
    x3 = c.x
    y3 = c.y
    angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
    if angle < 0:
        angle += 360
    return angle

time.sleep(2)
print("Starting training cycle...")
print("Enter pose name: ")
name = input()

print("Starting video capture in 5 seconds")
time.sleep(5)

final_arr = []
frame_counter = 0

while True:
    ret, frame = capture.read()
    final_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results_hands = hands.process(final_frame)
    results_pose = pose.process(final_frame)

    angles_data = [0, 0, 0, 0]
    try:
        LEFT_SHOULDER = results_pose.pose_landmarks.landmark[poseModule.PoseLandmark.LEFT_SHOULDER]
        RIGHT_SHOULDER = results_pose.pose_landmarks.landmark[poseModule.PoseLandmark.RIGHT_SHOULDER]
        LEFT_ELBOW = results_pose.pose_landmarks.landmark[poseModule.PoseLandmark.LEFT_ELBOW]
        RIGHT_ELBOW = results_pose.pose_landmarks.landmark[poseModule.PoseLandmark.RIGHT_ELBOW]
        LEFT_WRIST = results_pose.pose_landmarks.landmark[poseModule.PoseLandmark.LEFT_WRIST]
        RIGHT_WRIST = results_pose.pose_landmarks.landmark[poseModule.PoseLandmark.RIGHT_WRIST]

        angles_data = [
            calculateAngle(LEFT_WRIST, LEFT_ELBOW, LEFT_SHOULDER),
            calculateAngle(LEFT_ELBOW, LEFT_SHOULDER, RIGHT_SHOULDER),
            calculateAngle(LEFT_SHOULDER, RIGHT_SHOULDER, RIGHT_ELBOW),
            calculateAngle(LEFT_SHOULDER, RIGHT_SHOULDER, RIGHT_ELBOW),
        ]

        
    except Exception as e:
        pass

    hands_data = []

    if results_hands.multi_hand_landmarks:
        for hand_landmarks in results_hands.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                handsModule.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())

    # final_arr.append(angles_data + [0] * (80 - len(hands_data)) + hands_data + [name])
    final_arr.append(angles_data + [name])

    mp_drawing.draw_landmarks(frame, results_pose.pose_landmarks, poseModule.POSE_CONNECTIONS, landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
    cv2.imshow('Sign Language Translator', frame)

    if cv2.waitKey(1) == 27:
        break
    frame_counter += 1

    if frame_counter % 10 == 0:
        print(frame_counter)

    if frame_counter >= 250:
        break
    
cv2.destroyAllWindows()
capture.release()

def get_average(k):
    n = k[0][4]
    a = 0
    b = 0
    c = 0
    d = 0
    count = 0
    for i in k:
        count += 1
        a += i[0]
        b += i[1]
        c += i[2]
        d += i[3]
    return [a/count, b/count, c/count, d/count, n]


with open('angles_data.csv', 'a') as f_object:
    writer_object = writer(f_object)

    
    writer_object.writerow(get_average(final_arr))
  
    #Close the file object
    f_object.close()