import cv2
import numpy as np
import time

# Initialize webcam
cap = cv2.VideoCapture(0)

print("Warming up camera...")
time.sleep(2)
print("Press 's' to sample cloak color...")

background = None
hsv_lower = None
hsv_upper = None

# --------- STEP 1: Color Sampling ---------
while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        continue

    frame = cv2.flip(frame, 1)

    # Draw bigger ROI box
    cv2.rectangle(frame, (240, 190), (280, 230), (0, 255, 0), 2)
    cv2.putText(frame, "Place cloak in box & press 's'", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Sampling", frame)

    key = cv2.waitKey(1)
    if key == ord('s'):
        roi = frame[190:230, 240:280]  # Bigger box
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        hue, sat, val = cv2.split(hsv_roi)

        h = int(np.median(hue))
        s = int(np.median(sat))
        v = int(np.median(val))

        # Slightly wider range
        hsv_lower = np.array([max(h - 20, 0), max(s - 80, 50), max(v - 80, 50)])
        hsv_upper = np.array([min(h + 20, 180), min(s + 80, 255), min(v + 80, 255)])
        print("HSV Lower:", hsv_lower)
        print("HSV Upper:", hsv_upper)
        break

cv2.destroyAllWindows()

# --------- STEP 2: Capture Background ---------
print("Capturing background... Please move out of the frame.")
for i in range(60):
    ret, background = cap.read()
    if not ret or background is None:
        continue
    background = cv2.flip(background, 1)

print("Background captured. Now wear your cloak!")

# --------- STEP 3: Start Invisibility ---------
kernel = np.ones((5, 5), np.uint8)

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        continue

    frame = cv2.flip(frame, 1)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create mask
    mask = cv2.inRange(hsv, hsv_lower, hsv_upper)

    # Clean the mask
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=3)
    mask = cv2.dilate(mask, kernel, iterations=2)

    inverse_mask = cv2.bitwise_not(mask)

    # Create output
    cloak_area = cv2.bitwise_and(background, background, mask=mask)
    non_cloak_area = cv2.bitwise_and(frame, frame, mask=inverse_mask)
    final_output = cv2.addWeighted(cloak_area, 1, non_cloak_area, 1, 0)

    # Optional blur to smooth transitions
    final_output = cv2.GaussianBlur(final_output, (5, 5), 0)

    cv2.imshow("Harry Potter Invisibility Cloak", final_output)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
