import cv2
import cvzone
from cvzone.HandTrackingModule import HandDetector
import google.generativeai as genai
from PIL import Image
import numpy as np
import time

# Initialize Gemini
GEMINI_API_KEY = "AIzaSyDlZwVHqq8w6WPVBiNpC-l_TJvsCyjquWU"  # Replace with your actual key
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

# Initialize hand detector
detector = HandDetector(
    maxHands=1,
    detectionCon=0.8,
    minTrackCon=0.5,
    modelComplexity=1
)

# Create canvas
canvas = np.zeros((480, 640, 3), dtype=np.uint8)
prev_pos = None
output_text = ""
drawing_mode = False

# Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # Width
cap.set(4, 480)  # Height

print("Gesture Controls:")
print("1. 👆 Index finger up - Draw")
print("2. 👍 Thumb up - Clear canvas")
print("3. ✋ Open hand - Solve math")

try:
    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)

        # Detect hands
        hands, _ = detector.findHands(frame, draw=True)

        # Create a copy of canvas for display
        display_canvas = canvas.copy()

        if hands:
            hand = hands[0]
            fingers = detector.fingersUp(hand)
            lmList = hand["lmList"]

            # Get index finger tip position (landmark 8)
            current_pos = (lmList[8][0], lmList[8][1])

            # Show cursor on canvas (small circle at finger position)
            cv2.circle(display_canvas, current_pos, 8, (0, 255, 255), -1)  # Yellow cursor

            # Drawing logic
            if fingers == [0, 1, 0, 0, 0]:  # Index finger up
                drawing_mode = True
                if prev_pos:
                    # Draw line from previous position to current
                    cv2.line(canvas, prev_pos, current_pos, (255, 0, 255), 8)
                prev_pos = current_pos
            else:
                drawing_mode = False
                prev_pos = None

            # Clear canvas (thumb up)
            if fingers == [1, 0, 0, 0, 0]:
                canvas.fill(0)
                output_text = ""

            # Solve math (open hand)
            if fingers == [1, 1, 1, 1, 1]:
                pil_img = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
                try:
                    response = model.generate_content(["Solve this math problem", pil_img])
                    output_text = response.text[:100]  # Limit text length
                except Exception as e:
                    output_text = f"Error: {str(e)}"

        # Add instruction text
        if drawing_mode:
            cv2.putText(display_canvas, "Drawing...", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Add solution text
        if output_text:
            cv2.putText(display_canvas, output_text, (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Display both frames
        cv2.imshow("Webcam", frame)
        cv2.imshow("Math Canvas", display_canvas)

        # Exit on 'q' key or ESC key
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # 27 is the ESC key
            break

except KeyboardInterrupt:
    print("Stopping...")
finally:
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()