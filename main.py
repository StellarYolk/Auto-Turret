import cv2

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("error, can't open the video stream")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print("error, can't receive frames")
        break

    cv2.imshow("live video", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()