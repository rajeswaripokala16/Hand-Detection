import cv2
import numpy as np
import mediapipe as mp

# Initialize hand detection model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Drawing colors and variables
colors = [(255, 0, 255), (255, 0, 0), (0, 255, 0), (0, 0, 255)]
color_index = 0
draw_color = colors[color_index]
prev_x, prev_y = 0, 0
canvas = None

# Open webcam
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    if canvas is None:
        canvas = np.zeros_like(frame)

    # Process frame with MediaPipe
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)
    lm_list = []

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            for i, lm in enumerate(hand_landmarks.landmark):
                h, w, _ = frame.shape
                lm_list.append((int(lm.x * w), int(lm.y * h)))
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        if lm_list:
            index_finger_tip = lm_list[8]
            thumb_tip = lm_list[4]
            # Calculate distance between thumb and index finger tip
            dist = np.hypot(index_finger_tip[0] - thumb_tip[0], index_finger_tip[1] - thumb_tip[1])
            # Color selection (top boxes)
            if index_finger_tip[1] < 50:
                if 50 < index_finger_tip[0] < 150:
                    color_index = 0
                elif 170 < index_finger_tip[0] < 270:
                    color_index = 1
                elif 290 < index_finger_tip[0] < 390:
                    color_index = 2
                elif 410 < index_finger_tip[0] < 510:
                    color_index = 3
                draw_color = colors[color_index]
            # Draw if fingers are pinched
            elif dist < 40:
                if prev_x == 0 and prev_y == 0:
                    prev_x, prev_y = index_finger_tip
                cv2.line(canvas, (prev_x, prev_y), index_finger_tip, draw_color, 10)
                prev_x, prev_y = index_finger_tip
            else:
                prev_x, prev_y = 0, 0

    # Merge canvas and webcam frame
    gray_canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray_canvas, 20, 255, cv2.THRESH_BINARY)
    inv_mask = cv2.bitwise_not(mask)
    frame_bg = cv2.bitwise_and(frame, frame, mask=inv_mask)
    frame_fg = cv2.bitwise_and(canvas, canvas, mask=mask)
    frame = cv2.add(frame_bg, frame_fg)

    # Draw color boxes for selection
    for i, color in enumerate(colors):
        cv2.rectangle(frame, (50+120*i, 1), (150+120*i, 50), color, -1)
    cv2.putText(frame, 'Air Notepad: Pinch to draw, Touch box to select color, ESC to exit', (10, 470),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120, 120, 120), 2)

    cv2.imshow("Air Notepad", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
