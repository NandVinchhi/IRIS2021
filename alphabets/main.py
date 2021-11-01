import cv2
import mediapipe
import time
import math
from csv import writer, reader
import numpy as np

data = list(reader(open("alphabets.csv")))[1::]
poseModule = mediapipe.solutions.pose
handsModule = mediapipe.solutions.hands
mp_drawing = mediapipe.solutions.drawing_utils
mp_drawing_styles = mediapipe.solutions.drawing_styles
capture = cv2.VideoCapture(0)
frameWidth = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
frameHeight = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)

hands = handsModule.Hands(static_image_mode=False, min_detection_confidence=0.7, min_tracking_confidence=0.7, max_num_hands=1)

frame_counter = 0
drawn_alphabet = ""

while True:
    ret, frame = capture.read()
    final_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results_hands = hands.process(final_frame)
    
    hands_data = []

    

    if results_hands.multi_hand_landmarks:
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

    frame = cv2.putText(frame, drawn_alphabet, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                   2, (0, 234, 255), 3, cv2.LINE_AA)
    cv2.imshow('Signly', frame)

    if cv2.waitKey(1) == 27:
        break
    frame_counter += 1

    if frame_counter % 10 == 0:
        to_predict = [0] * (40 - len(hands_data)) + hands_data

        final_alphabet = "None"
        final_score = 1000

        for k in data:
            
            score = np.mean(np.absolute(np.subtract(to_predict, list(map(float, k[0:40])))))


            if score < final_score:
                final_score = score
                final_alphabet = k[-1]


        drawn_alphabet = final_alphabet


final_arr = final_arr / 250
cv2.destroyAllWindows()
capture.release()