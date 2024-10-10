import cv2
import time
import math
import numpy as np
import hand


# Set up video capture
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# Initialize hand tracker
detector = hand.HandTracker(detectionCon=0.7)

# Initialize time variables for FPS calculation
pTime, cTime = 0, 0

# Initialize volume variables
vol, volBar, volPer = 0, 0, 0

while True:
    success, img = cap.read()
    if not success:
        break

    img = detector.handsFinder(img)
    lmList = detector.positionFinder(img, draw=False)

    if lmList:
        x1, y1 = lmList[4][1], lmList[4][2]
        x2, y2 = lmList[8][1], lmList[8][2]
        ifx, ify = lmList[8][1], lmList[8][2]
        wx, wy = lmList[0][1], lmList[0][2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        # Draw landmarks and lines
        cv2.circle(img, (x1, y1), 5, (0, 255, 0), cv2.FILLED)
        cv2.circle(img, (x2, y2), 5, (0, 255, 0), cv2.FILLED)
        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.circle(img, (cx, cy), 5, (0, 255, 0), cv2.FILLED)

        # Calculate the length between two points
        length = math.hypot(x2 - x1, y2 - y1)
        refrenceLength = (math.hypot(wx - ifx, wy - ify) * 0.85)
        
        # Convert length to volume values

        volBar = np.interp(length, [0, refrenceLength], [400, 150])
        volPer = np.interp(length, [0, refrenceLength], [-10, 100])

        # Draw volume bar
        cv2.rectangle(img, (50, 150), (85, 400), (0, 255, 0), 3)
        cv2.rectangle(img, (50, int(volBar)), (85, 400), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, f'{int(volPer)} %', (40, 450), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)

    # Calculate and display FPS
    cTime = time.time()
    fps = int(1 // (cTime - pTime))
    pTime = cTime
    cv2.putText(img, f'FPS: {fps}', (40, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 0), 2)

    # Show the image
    cv2.imshow('Image', img)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()