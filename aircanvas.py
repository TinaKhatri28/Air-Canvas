import cv2 as cv
import numpy as np
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

cap = cv.VideoCapture(0)
finger_tips = [4, 8, 12, 16, 20]
colors = [(255, 0, 0), (128, 0, 52), (255, 0, 255), (0, 0, 0)]
color_index = 0
current_tool = "Draw"
prev_x, prev_y = 0, 0
canvas = None
tool_delay = 20
delay_counter = 0

def count_fingers(hand_landmarks):
    fingers = []
    
    fingers.append(hand_landmarks.landmark[finger_tips[0]].x < hand_landmarks.landmark[finger_tips[0] - 1].x)
    
    for tip in finger_tips[1:]:
        fingers.append(hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y)
    return fingers

def draw_color_palette(frame, colors, selected_index):
    x_start = 10
    y_start = 10
    box_size = 40
    gap = 10

    for i, color in enumerate(colors):
        top_left = (x_start + i * (box_size + gap), y_start)
        bottom_right = (top_left[0] + box_size, top_left[1] + box_size)
        cv.rectangle(frame, top_left, bottom_right, color, -1)
        if i == selected_index:
            cv.rectangle(frame, top_left, bottom_right, (255, 255, 255), 2)

with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5) as hands:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv.flip(frame, 1)
        h, w, _ = frame.shape
        if canvas is None:
            canvas = np.zeros_like(frame)

        rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                fingers = count_fingers(hand_landmarks)
                total_fingers = fingers.count(True)

                index_finger = hand_landmarks.landmark[8]
                x, y = int(index_finger.x * w), int(index_finger.y * h)

                if delay_counter == 0:
                    if fingers == [True, True, False, False, False]:
                        current_tool = "Switch"
                        delay_counter = tool_delay
                    elif fingers == [False, True, False, False, False]:
                        current_tool = "Draw"
                    elif fingers == [True, True, True, True, True]:
                        current_tool = "Change color"
                        color_index = (color_index + 1) % len(colors)
                        delay_counter = tool_delay
                    elif fingers == [True, False, False, False, False]:
                        current_tool = "Erase"
                        delay_counter = tool_delay
                    elif fingers == [False, True, True, True, False]:
                     
                        current_tool = "Cleared"
                    delay_counter = tool_delay

                if current_tool == "Draw" and fingers[1]:
                    if prev_x == 0 and prev_y == 0:
                        prev_x, prev_y = x, y
                    cv.line(canvas, (prev_x, prev_y), (x, y), colors[color_index], 5)
                    prev_x, prev_y = x, y
                elif current_tool == "Erase" and fingers[0]:
                    if prev_x == 0 and prev_y == 0:
                        prev_x, prev_y = x, y
                    cv.line(canvas, (prev_x, prev_y), (x, y), (0, 0, 0), 30)
                    prev_x, prev_y = x, y
                else:
                    prev_x, prev_y = 0, 0

        if delay_counter > 0:
            delay_counter -= 1

        
        mask = cv.cvtColor(canvas, cv.COLOR_BGR2GRAY)
        _, mask = cv.threshold(mask, 20, 255, cv.THRESH_BINARY)
        mask_inv = cv.bitwise_not(mask)
        frame_bg = cv.bitwise_and(frame, frame, mask=mask_inv)
        canvas_fg = cv.bitwise_and(canvas, canvas, mask=mask)
        combined = cv.add(frame_bg, canvas_fg)

        
        draw_color_palette(combined, colors, color_index)
        cv.putText(combined, f"Tool : {current_tool}", (10, 70), cv.FONT_HERSHEY_SIMPLEX, 0.7, (50, 50, 50), 2)

        cv.imshow("Air Canvas", combined)
        cv.imshow("Canvas", canvas)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv.destroyAllWindows()
