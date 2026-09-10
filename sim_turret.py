import pybullet as p
import pybullet_data
import numpy as np
import cv2
import random
import time

# --- Setup simulation ---
physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.8)

planeId = p.loadURDF("plane.urdf")
turretId = p.loadURDF("turret.urdf", basePosition=[0, 0, 0], useFixedBase=True)

# --- Identify joint indices ---
num_joints = p.getNumJoints(turretId)
joint_name_to_index = {}
for i in range(num_joints):
    info = p.getJointInfo(turretId, i)
    joint_name_to_index[info[1].decode('utf-8')] = i

PAN_JOINT = joint_name_to_index["pan_joint"]
TILT_JOINT = joint_name_to_index["tilt_joint"]
CAMERA_LINK = joint_name_to_index["camera_joint"]

print("Joint indices:", joint_name_to_index)

# --- Target object (respawns at random locations) ---
target_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.05, 0.05, 0.05], rgbaColor=[1, 1, 0, 1])
target_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.05, 0.05, 0.05])
targetId = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=target_collision,
                              baseVisualShapeIndex=target_visual, basePosition=[1.5, 0.0, 0.3])

# region within which targets can spawn (roughly within the turret's FOV/reach)
SPAWN_X = 1.5           # fixed depth in front of turret
SPAWN_Y_RANGE = (-0.6, 0.6)
SPAWN_Z_RANGE = (0.1, 0.6)

def respawn_target():
    new_y = random.uniform(*SPAWN_Y_RANGE)
    new_z = random.uniform(*SPAWN_Z_RANGE)
    p.resetBasePositionAndOrientation(targetId, [SPAWN_X, new_y, new_z], [0, 0, 0, 1])
    print(f"Target respawned at y={new_y:.2f}, z={new_z:.2f}")

respawn_target()

# --- Camera rendering ---
def get_camera_image(turret_id, camera_link_index, width=640, height=480):
    link_state = p.getLinkState(turret_id, camera_link_index, computeForwardKinematics=True)
    cam_pos = link_state[0]
    cam_orn = link_state[1]

    rot_matrix = np.array(p.getMatrixFromQuaternion(cam_orn)).reshape(3, 3)
    forward_vec = rot_matrix.dot(np.array([1, 0, 0]))
    up_vec = rot_matrix.dot(np.array([0, 0, 1]))
    cam_target = np.array(cam_pos) + forward_vec

    view_matrix = p.computeViewMatrix(cam_pos, cam_target.tolist(), up_vec.tolist())
    proj_matrix = p.computeProjectionMatrixFOV(fov=60, aspect=width / height, nearVal=0.05, farVal=10)

    img = p.getCameraImage(width, height, view_matrix, proj_matrix, renderer=p.ER_BULLET_HARDWARE_OPENGL)
    rgb = np.reshape(img[2], (height, width, 4))[:, :, :3]
    frame_bgr = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2BGR)
    return frame_bgr

# --- Servo control ---
def set_turret_angles(turret_id, pan_deg, tilt_deg):
    pan_rad = np.deg2rad(pan_deg)
    tilt_rad = np.deg2rad(tilt_deg)
    p.setJointMotorControl2(turret_id, PAN_JOINT, p.POSITION_CONTROL, targetPosition=pan_rad, force=0.5, positionGain=0.1, velocityGain=0.7)
    p.setJointMotorControl2(turret_id, TILT_JOINT, p.POSITION_CONTROL, targetPosition=tilt_rad, force=0.5, positionGain=0.1, velocityGain=0.7)

def get_current_turret_angles(turret_id):
    pan_state = p.getJointState(turret_id, PAN_JOINT)
    tilt_state = p.getJointState(turret_id, TILT_JOINT)
    pan_deg = np.rad2deg(pan_state[0])
    tilt_deg = np.rad2deg(tilt_state[0])
    return pan_deg, tilt_deg

# --- Control loop state ---
pan_cmd, tilt_cmd = 0.0, 0.0
SMOOTHING_FACTOR = 0.25
GAIN = 0.05                 # proportional gain, pixel error -> degrees
LOCK_THRESHOLD_PX = 12       # how close to center counts as "reached"
LOCK_HOLD_FRAMES = 5         # must stay locked this many consecutive frames before respawn
lock_counter = 0

while True:
    frame = get_camera_image(turretId, CAMERA_LINK)
    img_h, img_w = frame.shape[:2]
    center_x, center_y = img_w // 2, img_h // 2

    # --- simple color-blob detection (yellow target) ---
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (20, 100, 100), (35, 255, 255))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    target_found = False

    if contours:
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) > 20:  # ignore tiny noise blobs
            target_found = True
            x, y, w, h = cv2.boundingRect(largest)
            cx, cy = x + w // 2, y + h // 2

            # green detection box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # --- proportional control toward target ---
            error_x = cx - center_x
            error_y = cy - center_y

            pan_cmd -= SMOOTHING_FACTOR * error_x * GAIN
            tilt_cmd += SMOOTHING_FACTOR * error_y * GAIN

            pan_cmd = float(np.clip(pan_cmd, -85, 85))
            tilt_cmd = float(np.clip(tilt_cmd, -40, 40))

            set_turret_angles(turretId, pan_cmd, tilt_cmd)

            # --- check if we've locked onto the target (aim point near target center) ---
            dist = np.hypot(error_x, error_y)
            if dist < LOCK_THRESHOLD_PX:
                lock_counter += 1
            else:
                lock_counter = 0

            if lock_counter >= LOCK_HOLD_FRAMES:
                lock_counter = 0
                respawn_target()

    # --- red dot: where the turret is ACTUALLY pointing right now ---
    # since the rendered frame is taken from the camera's own forward direction,
    # the turret's aim point is always the center of the frame by definition.
    # we draw it slightly distinctly (bigger, filled) so it's visually clear
    # even when it overlaps the green box.
    cv2.circle(frame, (center_x, center_y), 6, (0, 0, 255), -1)
    cv2.circle(frame, (center_x, center_y), 9, (0, 0, 255), 1)  # thin ring for visibility

    status = "TRACKING" if target_found else "SEARCHING"
    cv2.putText(frame, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow("Turret Camera View", frame)
    p.stepSimulation()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

p.disconnect()
cv2.destroyAllWindows()