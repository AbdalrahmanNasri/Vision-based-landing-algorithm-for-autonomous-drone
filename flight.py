import sys
import cv2
import imutils
import time
from pymavlink import mavutil
from yoloDet import YoloTRT

# Function to draw the grid on the image
def draw_grid(image, grid_size, color=(255, 0, 0), thickness=1):
    img_height, img_width = image.shape[:2]
    cell_height = img_height // grid_size[0]
    cell_width = img_width // grid_size[1]

    # Draw horizontal lines
    for i in range(grid_size[0] + 1):
        cv2.line(image, (0, i * cell_height), (img_width, i * cell_height), color, thickness)

    # Draw vertical lines
    for j in range(grid_size[1] + 1):
        cv2.line(image, (j * cell_width, 0), (j * cell_width, img_height), color, thickness)

    return image

# Function to apply shading to a specific cell in the image
def apply_shading_to_cell(image, top_left, bottom_right, color, alpha=0.4):
    start_y = max(0, top_left[1])
    end_y = min(image.shape[0], bottom_right[1])
    start_x = max(0, top_left[0])
    end_x = min(image.shape[1], bottom_right[0])

    if end_y > start_y and end_x > start_x:
        overlay = image.copy()
        cv2.rectangle(overlay, top_left, bottom_right, color, cv2.FILLED)
        cv2.addWeighted(overlay[start_y:end_y, start_x:end_x], alpha, 
                        image[start_y:end_y, start_x:end_x], 1 - alpha, 0, 
                        image[start_y:end_y, start_x:end_x])

# Function to mark occupied cells on the image
def mark_occupied_cells_with_shading(image, detections, grid_size=(2, 2), shading_color=(0, 0, 255), alpha=0.4):
    occupied = set()
    image_with_grid = draw_grid(image, grid_size)
    img_height, img_width = image_with_grid.shape[:2]
    cell_height = img_height // grid_size[0]
    cell_width = img_width // grid_size[1]

    for detection in detections:
        x, y, width, height = detection['box']

        start_col = int(x / cell_width)
        end_col = int((x + width) / cell_width)
        start_row = int(y / cell_height)
        end_row = int((y + height) / cell_height)

        for row in range(start_row, end_row + 1):
            for col in range(start_col, end_col + 1):
                if 0 <= col < grid_size[1] and 0 <= row < grid_size[0]:
                    occupied.add((row, col))
                    top_left = (col * cell_width, row * cell_height)
                    bottom_right = ((col + 1) * cell_width, (row + 1) * cell_height)
                    apply_shading_to_cell(image_with_grid, top_left, bottom_right, shading_color, alpha)

    return image_with_grid, occupied

# Connect to the Vehicle
print("Connecting to vehicle on: /dev/ttyUSB0 at 57600 baud")
mav = mavutil.mavlink_connection('/dev/ttyTHS1', baud=57600)

# Wait for a heartbeat to confirm the connection
print("Waiting for device...")
#mav.wait_heartbeat()
print("Heartbeat from system (system %u component %u)" % (mav.target_system, mav.target_component))

# Function to send NED velocity commands
def send_ned_velocity(velocity_x, velocity_y, velocity_z, duration):
    """
    Move vehicle in direction based on specified velocity vectors.
    """

    msg = mav.mav.set_position_target_local_ned_encode(
        0,  # time_boot_ms (not used)
        mav.target_system, mav.target_component,
        mavutil.mavlink.MAV_FRAME_LOCAL_NED,  # frame
        0b0000111111000111,  # type_mask (only speeds enabled)
        0, 0, 0,  # x, y, z positions (not used)
        velocity_x, velocity_y, velocity_z,  # x, y, z velocity in m/s
        0, 0, 0,  # x, y, z acceleration (not supported yet, ignored in GCS_Mavlink)
        0, 0)     # yaw, yaw_rate (not supported yet, ignored in GCS_Mavlink)
    
    # Use the lower-level send function if send_mavlink is not available
    for _ in range(duration):
        if hasattr(mav.mav, 'send'):
            mav.mav.send(msg)
        elif hasattr(mav, 'send_mavlink'):
            mav.send_mavlink(msg)
        else:
            print("The MAVLink connection does not have send capabilities.")
        time.sleep(1)


# Function to command the drone to land
def land_drone():
    """
    Commands the drone to land.
    """
    mav.mav.command_long_send(
        mav.target_system, mav.target_component,
        mavutil.mavlink.MAV_CMD_NAV_LAND, 
        0, 0, 0, 0, 0, 0, 0, 0)

# Initialize the YOLO model
model = YoloTRT(library="yolov5/new/libmyplugins.so", engine="yolov5/new/yolov5s.engine", conf=0.5, yolo_ver="v5")

# Start capturing video from the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    sys.exit()

grid_size = (2, 2)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    frame = imutils.resize(frame, width=600)

    try:
        detections, t = model.Inference(frame)
    except Exception as e:
        print("Error during inference: ", str(e))
        continue

    # Prepare detections for marking the grid
    formatted_detections = [{'box': (obj['box'][0], obj['box'][1], obj['box'][2], obj['box'][3])} for obj in detections]

    # Mark the occupied cells and get the set of occupied cells
    processed_frame, occupied = mark_occupied_cells_with_shading(frame, formatted_detections, grid_size=grid_size)

    # Identify free cells
    free_cells = [(row, col) for row in range(grid_size[0]) for col in range(grid_size[1]) if (row, col) not in occupied]

    # Display free cells
    print("Free Cells:", free_cells)
       
    

    if free_cells:
        target_cell = free_cells[0]
        print("Target Cell for Motion:", target_cell)
        
        # Calculate target position in the image
        img_height, img_width = frame.shape[:2]
        cell_height = img_height // grid_size[0]
        cell_width = img_width // grid_size[1]
        target_point = (target_cell[1] * cell_width + cell_width // 2, target_cell[0] * cell_height + cell_height // 2)
        
        # Draw a circle on the target cell
        cv2.circle(processed_frame, target_point, 10, (0, 255, 0), -1)
        cv2.putText(processed_frame, "Target", (target_point[0] + 10, target_point[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Calculate motion commands to reach the target cell's center
        motion_x = (target_cell[1] - 0.5 * (grid_size[1] - 1)) * cell_width
        motion_y = (target_cell[0] - 0.5 * (grid_size[0] - 1)) * cell_height
        
        # Normalize the motion vector
        norm = (motion_x**2 + motion_y**2)**0.5
        motion_x /= norm
        motion_y /= norm
        
        # Set a moderate speed (m/s) and short duration (s)
        speed = 0.5  # m/s
        duration = 1  # s
        
        
        # Send motion command
        print(f"Sending motion command: ({motion_x * speed}, {motion_y * speed}, 0)")
        send_ned_velocity(motion_x * speed, motion_y * speed, 0, duration)
        
        # Stop the drone after moving to allow any corrections
        send_ned_velocity(0, 0, 0, 1)

        # Land the drone after reaching the target
        print("Landing the drone")
        land_drone()

        cv2.imshow("Processed Frame", processed_frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
          break
    break

    
fps = 1 / t
print(f"FPS: {fps:.2f}")

    # Display the processed frame
    #cv2.imshow("Processed Frame", processed_frame)



# Clean up
cap.release()
cv2.destroyAllWindows()
mav.close()
print("Vehicle Disconnected")