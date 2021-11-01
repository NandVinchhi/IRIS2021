import numpy as np
from fastdtw import fastdtw
import cv2
import mediapipe
import time
import math
from csv import writer, reader
import numpy as np
from scipy.spatial.distance import euclidean


poseModule = mediapipe.solutions.pose
handsModule = mediapipe.solutions.hands
mp_drawing = mediapipe.solutions.drawing_utils
mp_drawing_styles = mediapipe.solutions.drawing_styles
# capture = cv2.VideoCapture(0)

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

hands = handsModule.Hands(static_image_mode=False, min_detection_confidence=0.7, min_tracking_confidence=0.7, max_num_hands=2)
pose = poseModule.Pose(min_detection_confidence=0.5,min_tracking_confidence=0.5)


def get_data(path):

    capture = cv2.VideoCapture(path)

    frameWidth = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    frameHeight = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)

    final_data = []

    num_frames = 0

    init = time.time()
    while capture.isOpened():
        ret, frame = capture.read()
        num_frames += 1

        try:
            final_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results_hands = hands.process(final_frame)
            results_pose = pose.process(final_frame)

            angles_data = [180, 90, 90, 180]
            

            hands_data = []

            if results_hands.multi_hand_landmarks:

                counter = 1
                for hand_landmarks in results_hands.multi_hand_landmarks:
                    all_landmarks = [
                        handsModule.HandLandmark.THUMB_CMC, 
                        handsModule.HandLandmark.THUMB_MCP, 
                        handsModule.HandLandmark.THUMB_IP, 
                        handsModule.HandLandmark.THUMB_TIP, 
                        handsModule.HandLandmark.INDEX_FINGER_MCP, 
                        handsModule.HandLandmark.INDEX_FINGER_PIP, 
                        handsModule.HandLandmark.INDEX_FINGER_DIP, 
                        handsModule.HandLandmark.INDEX_FINGER_TIP, 
                        handsModule.HandLandmark.MIDDLE_FINGER_MCP, 
                        handsModule.HandLandmark.MIDDLE_FINGER_PIP, 
                        handsModule.HandLandmark.MIDDLE_FINGER_DIP, 
                        handsModule.HandLandmark.MIDDLE_FINGER_TIP, 
                        handsModule.HandLandmark.RING_FINGER_MCP, 
                        handsModule.HandLandmark.RING_FINGER_PIP, 
                        handsModule.HandLandmark.RING_FINGER_DIP, 
                        handsModule.HandLandmark.RING_FINGER_TIP, 
                        handsModule.HandLandmark.PINKY_MCP, 
                        handsModule.HandLandmark.PINKY_PIP,
                        handsModule.HandLandmark.PINKY_DIP,
                        handsModule.HandLandmark.PINKY_TIP
                    ]

                    standard_distance = math.sqrt(abs(hand_landmarks.landmark[handsModule.HandLandmark.WRIST].x - hand_landmarks.landmark[handsModule.HandLandmark.THUMB_CMC].x) ** 2 + abs(hand_landmarks.landmark[handsModule.HandLandmark.WRIST].y - hand_landmarks.landmark[handsModule.HandLandmark.THUMB_CMC].y) ** 2)

                    for j in all_landmarks:
                        hands_data.append((hand_landmarks.landmark[handsModule.HandLandmark.WRIST].x - hand_landmarks.landmark[j].x)/standard_distance)
                        hands_data.append((hand_landmarks.landmark[handsModule.HandLandmark.WRIST].y - hand_landmarks.landmark[j].y)/standard_distance)

                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        handsModule.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

            

            mp_drawing.draw_landmarks(frame, results_pose.pose_landmarks, poseModule.POSE_CONNECTIONS, landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
            cv2.imshow('Signly', frame)

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

            final_arr = angles_data + [0] * (80 - len(hands_data)) + hands_data

            if cv2.waitKey(1) == 32:
                final_data.append(final_arr)
                print("yes")
            
        except Exception as e:
            print(e)
            break

        if cv2.waitKey(1) == 27:
            break



    cv2.destroyAllWindows()
    capture.release()
    return final_data

data1 = get_data("hello.mp4")
data2 = get_data("thankyou.mp4")
data3 = get_data("hello.mp4")
data4 = get_data("welcome.mp4")

distance1, path = fastdtw(data1, data2, dist=euclidean)
distance1 = distance1/((len(data1) + len(data2))/2)

distance2, path = fastdtw(data1, data3, dist=euclidean)
distance2 = distance2/((len(data1) + len(data3))/2)

distance3, path = fastdtw(data1, data4, dist=euclidean)
distance3 = distance3/((len(data1) + len(data4))/2)


print(distance1)
print(distance2)
print(distance3)