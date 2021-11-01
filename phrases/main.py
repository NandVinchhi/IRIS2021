import cv2
import mediapipe
import time
import math
from csv import writer, reader
from joblib import dump, load
import numpy as np
import os
from gtts import gTTS
from playsound import playsound
import threading


clf = load('model.joblib')

angles_csv = list(reader(open("angles_data.csv")))

pose2_csv = list(reader(open("pose2.csv")))
pose3_csv = list(reader(open("pose3.csv")))

poseModule = mediapipe.solutions.pose
handsModule = mediapipe.solutions.hands
mp_drawing = mediapipe.solutions.drawing_utils
mp_drawing_styles = mediapipe.solutions.drawing_styles
capture = cv2.VideoCapture(0)
frameWidth = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
frameHeight = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)

hands = handsModule.Hands(static_image_mode=False, min_detection_confidence=0.7, min_tracking_confidence=0.7, max_num_hands=2)
pose = poseModule.Pose(min_detection_confidence=0.5,min_tracking_confidence=0.5)

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

def get_pose(k):
    csvreader = reader(angles_csv)

    p = "None"

    s = 180
    for row in angles_csv:
        a = abs(float(row[0]) - k[0])
        b = abs(float(row[1]) - k[1])
        c = abs(float(row[2]) - k[2])
        d = abs(float(row[3]) - k[3])

        if a > 180:
            a = 360 - a
        if b > 180:
            b = 360 - b
        if c > 180:
            c = 360 - c
        if d > 180:
            d = 360 - d
            
        score = (a + b + c + d)/4
        

        if a < 20 and b < 20 and c < 20 and d < 20 and score < s:
            s = score
            p = row[4]
    
    return p

def parsed(s):
    final = []
    d = {
        "HELLO": ["hello1", "hello2"],
        "THANK YOU": ["thankyou1", "thankyou2"],
        "PLEASE": ["please"],
        "HOW ARE YOU": ["howareyou"],
        "I LOVE YOU": ["iloveyou"]
    }
    for i in range(0, len(s)):
        for j in d:
            try:
                if d[j] == [s[i]] or d[j] == [s[i], s[i + 1]]:

                    if len(final) == 0:
                        final.append(j)
                    else:
                        if final[-1] != j:
                            final.append(j)
                    break
            except:
                pass


    return final

frame_counter = 0

last_sign = "none"

sentence = []

final_parsed_sentence = []
parsed_sentence = []


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

    # final_arr = angles_data + [0] * (80 - len(hands_data)) + hands_data
    final_arr = angles_data

    mp_drawing.draw_landmarks(frame, results_pose.pose_landmarks, poseModule.POSE_CONNECTIONS, landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
    frame = cv2.putText(frame, " ".join(parsed_sentence), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                   1, (0, 234, 255), 3, cv2.LINE_AA)
    cv2.imshow('Signly', frame)

    if cv2.waitKey(1) == 27:
        break
    frame_counter += 1
    
    if frame_counter % 5 == 0:
        final_pose = get_pose(angles_data)

        final_sign = "none"
        if final_pose == "pose1":
            final_sign = "hello1"
        elif final_pose == "pose2":
            predict_data = [0] * (80 - len(hands_data)) + hands_data

            score1 = np.mean(np.absolute(np.subtract(predict_data, list(map(float, pose2_csv[1][0:80])))))
            score2 = np.mean(np.absolute(np.subtract(predict_data, list(map(float, pose2_csv[2][0:80])))))

            final_sign = "hello2"
            
        elif final_pose == "pose3":
            predict_data = [0] * (80 - len(hands_data)) + hands_data

            score1 = np.mean(np.absolute(np.subtract(predict_data, list(map(float, pose3_csv[1][0:80])))))
            score2 = np.mean(np.absolute(np.subtract(predict_data, list(map(float, pose3_csv[2][0:80])))))

            if score1 < score2:
                final_sign = "name"
            else:
                final_sign = "howareyou"
                
        elif final_pose == "pose4":
            final_sign = "thankyou1"
        elif final_pose == "pose5":
            final_sign = "thankyou2"
            
        if final_sign != "none":
            if final_sign != last_sign:

                if final_sign == "thankyou2":
                    threading.Thread(target=playsound, args=('thankyou.mp3',), daemon=True).start()
                elif final_sign == "howareyou":
                    threading.Thread(target=playsound, args=('howareyou.mp3',), daemon=True).start()
                elif final_sign == "hello2":
                    threading.Thread(target=playsound, args=('hello.mp3',), daemon=True).start()
                sentence.append(final_sign)
                final_parsed_sentence = parsed(sentence)

                if len(final_parsed_sentence) > len(parsed_sentence):
                    parsed_sentence = final_parsed_sentence
                    
                last_sign = final_sign

cv2.destroyAllWindows()
capture.release()



