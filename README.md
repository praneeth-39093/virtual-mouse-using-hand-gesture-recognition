# virtual-mouse-using-hand-gesture-recognition
import cv2
import mediapipe as mp
import pyautogui
import numpy as np

# Initialize hand tracking module
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8)

# Get screen size
screen_w, screen_h = pyautogui.size()
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue
    
    frame = cv2.flip(frame, 1)  # Mirror image
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            landmarks = hand_landmarks.landmark
            
            # Extract finger tip coordinates
            index_finger_tip = landmarks[8]  # Index finger tip
            middle_finger_tip = landmarks[12] # Middle finger tip
            thumb_tip = landmarks[4] # Thumb tip
            
            index_x, index_y = int(index_finger_tip.x * w), int(index_finger_tip.y * h)
            middle_x, middle_y = int(middle_finger_tip.x * w), int(middle_finger_tip.y * h)
            thumb_x, thumb_y = int(thumb_tip.x * w), int(thumb_tip.y * h)
            
            # Convert to screen coordinates
            screen_x = np.interp(index_x, [0, w], [0, screen_w])
            screen_y = np.interp(index_y, [0, h], [0, screen_h])
            
            # Mouse movement
            pyautogui.moveTo(screen_x, screen_y, duration=0.1)
            
            # Left click: Index up, Middle down
            if index_y < middle_y:
                pyautogui.click()
                cv2.putText(frame, "Left Click", (index_x, index_y-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Right click: Both index and middle up
            if index_y < middle_y and abs(index_y - middle_y) < 10:
                pyautogui.rightClick()
                cv2.putText(frame, "Right Click", (index_x, index_y-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Scroll up and down
            if abs(index_y - middle_y) > 40:
                direction = "Up" if index_y < middle_y else "Down"
                pyautogui.scroll(10 if direction == "Up" else -10)
                cv2.putText(frame, f"Scroll {direction}", (index_x, index_y-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            # Drag and drop (pinch gesture)
            pinch_distance = np.linalg.norm([index_x - thumb_x, index_y - thumb_y])
            if pinch_distance < 30:
                pyautogui.mouseDown()
                cv2.putText(frame, "Dragging", (index_x, index_y-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            else:
                pyautogui.mouseUp()
    
    cv2.imshow("Virtual Mouse", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

