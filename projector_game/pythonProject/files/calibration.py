import cv2
import numpy as np

# List to store points clicked by the user
points_camera = []

def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points_camera.append((x, y))
        print(f"Point captured: ({x}, {y})")
        cv2.circle(param, (x, y), 5, (0, 255, 0), -1)

def capture_calibration_points(camera_index):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Error: Could not open camera with index {camera_index}")
        return None, None

    points_projector = np.array([
        [0, 0], [0, 799], [599, 0], [599, 799]  # Adjust according to your grid size
    ], dtype='float32')

    global points_camera
    points_camera = []

    while len(points_camera) < 4:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break

        cv2.imshow('Calibration', frame)
        cv2.setMouseCallback('Calibration', click_event, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if len(points_camera) < 4:
        print("Error: Not enough points captured")
        return None, None

    points_camera = np.array(points_camera, dtype='float32')
    return points_projector, points_camera

# Manually set the DroidCam index to 1 for testing
camera_index = 0

# Capture calibration points using the manually set camera index
points_projector, points_camera = capture_calibration_points(camera_index)

if points_projector is not None and points_camera is not None:
    M = cv2.getPerspectiveTransform(points_camera, points_projector)
    np.savez('calibration_data.npz', M=M)
    print("Calibration data saved.")
else:
    print("Calibration failed.")
