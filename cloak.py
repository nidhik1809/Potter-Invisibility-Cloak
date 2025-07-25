import cv2
import numpy as np
import time

# Webcam chalu karo
cap = cv2.VideoCapture(0)

print("Warming up camera...")  # Camera ko thoda time do warm up hone ke liye
time.sleep(2)
print("Press 's' to sample cloak color...")  # Jab ready ho, cloak ka color sample karne 's' dabao

background = None
hsv_lower = None
hsv_upper = None

# --------- Color Sampling ---------
while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        continue

    frame = cv2.flip(frame, 1)  # Mirror jaise flip kar rahe frame ko

    # ROI box draw kar rahe jahan cloak rakhoge chota box sa
    cv2.rectangle(frame, (240, 190), (280, 230), (0, 255, 0), 2)
    cv2.putText(frame, "Place cloak in box & press 's'", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Sampling", frame)

    key = cv2.waitKey(1)
    if key == ord('s'):
        roi = frame[190:230, 240:280]  
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        hue, sat, val = cv2.split(hsv_roi)

        h = int(np.median(hue))
        s = int(np.median(sat))
        v = int(np.median(val))

        # Color ka thoda wide range le rahe tohki zyada accurate hoga
        hsv_lower = np.array([max(h - 20, 0), max(s - 80, 50), max(v - 80, 50)])
        hsv_upper = np.array([min(h + 20, 180), min(s + 80, 255), min(v + 80, 255)])
        print("HSV Lower:", hsv_lower)
        print("HSV Upper:", hsv_upper)
        break

cv2.destroyAllWindows()

# --------- Background Capture ---------
print("Capturing background... Please move out of the frame.")  # Ab tu frame se nikal ja 
for i in range(60):
    ret, background = cap.read()
    if not ret or background is None:
        continue
    background = cv2.flip(background, 1)  # Flip kar rahe taaki orientation match hojaye

print("Background captured. Now wear your cloak!")  # Ab cloak pehen le bhai

# --------- Invisibility Effect ---------
kernel = np.ones((5, 5), np.uint8)  # Mask smooth karne ke liye kernel bana liya

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        continue

    frame = cv2.flip(frame, 1)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, hsv_lower, hsv_upper)

    # Mask ko clean kar rahe using morphology ( basically, noise hata rahe)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=3)
    mask = cv2.dilate(mask, kernel, iterations=2)

    inverse_mask = cv2.bitwise_not(mask)  # Jo cloak nahi hai, uska mask banaya lol

    # Output ka setup - cloak area background se aur baaki real frame se
    cloak_area = cv2.bitwise_and(background, background, mask=mask)
    non_cloak_area = cv2.bitwise_and(frame, frame, mask=inverse_mask)
    final_output = cv2.addWeighted(cloak_area, 1, non_cloak_area, 1, 0)

    # Smooth transitions ke liye thoda blur
    final_output = cv2.GaussianBlur(final_output, (5, 5), 0)

    cv2.imshow("Harry Potter Invisibility Cloak", final_output) 

    if cv2.waitKey(1) == ord('q'):  # Jab chaho exit karo 'q' se
        break

cap.release()
cv2.destroyAllWindows()
