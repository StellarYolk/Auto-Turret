import cv2
from turret_class import TurretController
import numpy as np
import json

turret = TurretController(port="/dev/ttyACM0")
cap = cv2.VideoCapture(1)

pan, tilt = 90, 90
turret.send(pan, tilt, laser_on=True)

calibration_points = []  # will store (pixel_x, pixel_y, pan, tilt)

print("Controls: WASD to jog, SPACE to record current laser position, ESC to finish")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.putText(frame, f"Pan: {pan}  Tilt: {tilt}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow("calibration", frame)

    key = cv2.waitKey(1) & 0xFF
    step = 1

    if key == ord('a'):
        pan -= step
    elif key == ord('d'):
        pan += step
    elif key == ord('w'):
        tilt -= step
    elif key == ord('s'):
        tilt += step
    elif key == ord(' '):
        # click on the frame to mark where the laser dot actually landed
        print("Click on the laser dot in the frame window, then press any key")
        clicked = {}
        def on_click(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                clicked['x'], clicked['y'] = x, y
        cv2.setMouseCallback("calibration", on_click)
        cv2.waitKey(0)
        if 'x' in clicked:
            calibration_points.append((clicked['x'], clicked['y'], pan, tilt))
            print(f"Recorded: pixel=({clicked['x']},{clicked['y']}) pan={pan} tilt={tilt}")
    elif key == 27:  # ESC
        break

    turret.send(pan, tilt, laser_on=True)

cap.release()
cv2.destroyAllWindows()
turret.close()

print(calibration_points)


def fit_calibration(calibration_points=calibration_points):
    pixels = np.array([[p[0], p[1], 1]for p in calibration_points])
    pan = np.array([p[2] for p in calibration_points])
    tilt = np.array([p[3] for p in calibration_points])

    pan_coeffs, _, _, _ = np.linalg.lstsq(pixels, pan, rcond=None)
    tilt_coeffs, _, _, _ = np.linalg.lstsq(pixels, tilt, rcond=None)

    return pan_coeffs, tilt_coeffs

pan_coeffs, titl_coeffs = fit_calibration(calibration_points)
print("Pan coeffs (a,b,c): ", pan_coeffs)
print("tilt coeffs (a,b,c): ", titl_coeffs)

data = {
    "pan_coeffs": pan_coeffs,
    "tilt_coeffs": titl_coeffs
}

with open("calibration.json", "w") as file:
    json.dump(data, file, indent=4)