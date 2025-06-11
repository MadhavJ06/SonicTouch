# collect_data.py
import csv
import time
import threading
from collections import deque
from pythonosc import dispatcher, osc_server

# --- Configuration ---
IP = "0.0.0.0"
PORT = 6448
RECORD_SECONDS = 1.5
CSV_FILE = 'gestures.csv'

# --- Predefined Gestures ---
GESTURE_OPTIONS = [
    'rest',
    'crescendo', 
    'stir',
    'wave',
    'tap',
    'circle',
    'shake',
    'swipe_left',
    'swipe_right',
    'point'
]

# --- Data Storage ---
# Use a deque for efficient appending. This will hold our temporary recording.
recording_buffer = deque()
is_recording = False
current_gesture_name = ""

# --- OSC Handler ---
def sensor_handler(address, x, y, z):
    if is_recording:
        timestamp = time.time()
        # We flatten the data into a single stream for simplicity
        recording_buffer.append((timestamp, address, x, y, z))

# --- Menu Display ---
def display_gesture_menu():
    print("\n=== Gesture Recording Menu ===")
    for i, gesture in enumerate(GESTURE_OPTIONS, 1):
        print(f"{i}. {gesture}")
    print("c. Custom gesture (type your own)")
    print("q. Quit")
    print("=" * 30)

# --- Main Logic ---
def record_gesture(gesture_name):
    global is_recording, recording_buffer
    recording_buffer.clear()
    is_recording = True
    print(f"Recording '{gesture_name}' in 3... 2... 1...")
    time.sleep(3)
    print("...RECORDING NOW...")
    
    # Let the recording run for the specified duration
    time.sleep(RECORD_SECONDS)
    
    is_recording = False
    print(f"...Done. Recorded {len(recording_buffer)} data points.")
    save_to_csv(gesture_name)

def save_to_csv(gesture_name):
    with open(CSV_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        for timestamp, address, x, y, z in recording_buffer:
            # We add the gesture name as the first column for every row
            writer.writerow([gesture_name, timestamp, address, x, y, z])
    print(f"Saved '{gesture_name}' to {CSV_FILE}")

if __name__ == "__main__":
    # Setup CSV file with header if it doesn't exist
    try:
        with open(CSV_FILE, 'x', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['gesture', 'timestamp', 'sensor', 'x', 'y', 'z'])
    except FileExistsError:
        pass # File already exists

    # Setup OSC server in a separate thread
    disp = dispatcher.Dispatcher()
    disp.map("/accelerometer", sensor_handler)
    disp.map("/gyroscope", sensor_handler)
    server = osc_server.ThreadingOSCUDPServer((IP, PORT), disp)
    server_thread = threading.Thread(target=server.serve_forever)
    server_thread.daemon = True
    server_thread.start()
    print(f"Listening for sensor data on port {PORT}")

    # Main loop for user interaction
    while True:
        display_gesture_menu()
        choice = input("Select option: ").strip().lower()
        
        if choice == 'q':
            break
        elif choice == 'c':
            custom_gesture = input("Enter custom gesture name: ").strip()
            if custom_gesture:
                record_gesture(custom_gesture)
        else:
            try:
                choice_num = int(choice)
                if 1 <= choice_num <= len(GESTURE_OPTIONS):
                    gesture_name = GESTURE_OPTIONS[choice_num - 1]
                    record_gesture(gesture_name)
                else:
                    print(f"Invalid choice. Please select 1-{len(GESTURE_OPTIONS)}, 'c', or 'q'.")
            except ValueError:
                print(f"Invalid input. Please select 1-{len(GESTURE_OPTIONS)}, 'c', or 'q'.")
    
    server.shutdown()
    print("Data collection finished.")