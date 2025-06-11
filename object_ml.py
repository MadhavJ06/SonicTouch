import argparse
import numpy as np
import time
import pickle
import joblib
import pandas as pd
from pythonosc import dispatcher
from pythonosc import osc_server
from pythonosc import udp_client
from collections import deque
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

class GestureRecognizer:
    def __init__(self, input_port=12000, processing_port=12001, ableton_port=11000,
                 phone_port=6448, confidence_threshold=0.75, gesture_cooldown=1.0, min_movement=0.2):
        # OSC communication
        self.client_processing = udp_client.SimpleUDPClient("127.0.0.1", processing_port)
        self.client_ableton = udp_client.SimpleUDPClient("127.0.0.1", ableton_port)
        
        # Object tracking data storage (now supports multiple objects)
        self.max_objects = 4
        self.position_histories = [deque(maxlen=30) for _ in range(self.max_objects)]
        self.last_gesture_times = [time.time() for _ in range(self.max_objects)]
        self.gesture_cooldown = gesture_cooldown  # Seconds between gesture recognition attempts
        self.confidence_threshold = confidence_threshold  # Minimum confidence for gesture detection
        self.min_movement = min_movement  # Minimum movement for gesture detection
        self.current_tracks = list(range(self.max_objects))  # Each object controls its own track
        
        # Phone sensor data storage (from run_conductor.py)
        self.live_data_buffer = deque()
        self.last_prediction_time = 0
        self.prediction_cooldown_seconds = 2.0  # Wait 2s between predictions
        self.analysis_window_seconds = 1.5  # Must be same as recording duration
        
        # Load object tracking ML model (if it exists) or create an empty one
        try:
            with open('gesture_model.pkl', 'rb') as f:
                self.object_model = pickle.load(f)
                self.object_scaler = pickle.load(f)
                print("Loaded existing object gesture model")
                self.object_model_exists = True
        except FileNotFoundError:
            self.object_model = RandomForestClassifier(n_estimators=50)
            self.object_scaler = StandardScaler()
            self.object_model_exists = False
            print("No object gesture model found. Use training mode to create one.")
        
        # Load phone sensor ML model (from run_conductor.py)
        try:
            self.sensor_model = joblib.load('conductor_model.joblib')
            self.sensor_scaler = joblib.load('scaler.joblib')
            print("Loaded existing sensor gesture model")
            self.sensor_model_exists = True
        except FileNotFoundError:
            self.sensor_model = None
            self.sensor_scaler = None
            self.sensor_model_exists = False
            print("No sensor gesture model found.")
        
        # Training data storage (for object tracking)
        self.training_data = []
        self.training_labels = []
        self.recording_gesture = False
        self.current_gesture_label = None
        self.current_training_object = 0  # Which object we're currently training with
        self.training_feedback = {}  # Store feedback about recent training samples
        
        # Track playback state for proper play/pause toggle
        self.is_playing = False
        self.last_play_state_query = 0
        
        # Set up OSC server with combined handlers
        self.dispatcher = dispatcher.Dispatcher()
        # Object tracking handlers
        self.dispatcher.map("/tracking/position", self.position_handler)
        self.dispatcher.map("/tracking/track", self.track_handler)
        self.dispatcher.map("/training/start", self.start_recording_handler)
        self.dispatcher.map("/training/stop", self.stop_recording_handler)
        self.dispatcher.map("/training/save", self.save_model_handler)
        # Phone sensor handlers (from run_conductor.py)
        self.dispatcher.map("/accelerometer", self.sensor_handler)
        self.dispatcher.map("/gyroscope", self.sensor_handler)
        
        # Set up parameter query response listeners
        self.setup_parameter_response_listeners()
        
        # Define input and output addresses separately
        self.input_address = ("0.0.0.0", phone_port)  # Listen on all interfaces for all OSC data
        self.processing_address = ("127.0.0.1", processing_port)
        self.ableton_address = ("127.0.0.1", ableton_port)

        # Start OSC server using the defined input address
        self.server = osc_server.ThreadingOSCUDPServer(
            self.input_address, self.dispatcher)
        print(f"Starting combined OSC server on {self.input_address[0]}:{self.input_address[1]}")
        print(f"Sending to Processing on {self.processing_address[0]}:{self.processing_address[1]}")
        print(f"Sending to Ableton on {self.ableton_address[0]}:{self.ableton_address[1]}")
        print(f"Recognition settings: confidence={confidence_threshold}, cooldown={gesture_cooldown}s, min_movement={min_movement}")
    def position_handler(self, unused_addr, object_index, x, y, detected):
        """Handle incoming position updates from Processing for multiple objects"""
        if object_index >= self.max_objects:
            return  # Ignore invalid object indices
            
        if not detected:
            self.position_histories[object_index].clear()
            return
            
        # Add position to history for this specific object
        self.position_histories[object_index].append((x, y))
        
        # If we're recording a gesture for training
        if self.recording_gesture and self.current_gesture_label and self.current_training_object == object_index:
            if len(self.position_histories[object_index]) >= 15:  # Wait until we have enough positions
                # Check for significant movement
                positions = np.array(list(self.position_histories[object_index]))
                total_movement = self.calculate_total_movement(positions)
                
                # Only record samples with significant movement
                if total_movement > self.min_movement:
                    # Extract features and record
                    features = self.extract_gesture_features(object_index)
                    if features is not None:
                        self.training_data.append(features)
                        self.training_labels.append(self.current_gesture_label)
                        print(f"Recorded sample for gesture '{self.current_gesture_label}' with object {object_index} (movement: {total_movement:.3f})")
                        
                        # Send feedback to Processing
                        self.client_processing.send_message("/training/sample", 
                                                           [self.current_gesture_label, total_movement, object_index])
        
        # If we have enough positions and not in training mode, analyze for gesture
        elif len(self.position_histories[object_index]) >= 15 and self.object_model_exists:
            current_time = time.time()
            if current_time - self.last_gesture_times[object_index] > self.gesture_cooldown:
                self.analyze_gesture(object_index)
                self.last_gesture_times[object_index] = current_time
    
    def calculate_total_movement(self, positions):
        """Calculate total movement in a sequence of positions"""
        if len(positions) < 2:
            return 0.0
            
        displacements = np.diff(positions, axis=0)
        total_movement = np.sum(np.sqrt(np.sum(displacements**2, axis=1)))
        return total_movement
    
    def track_handler(self, unused_addr, object_index, track_number):
        """Handle track change messages for specific objects"""
        if object_index < self.max_objects:
            self.current_tracks[object_index] = track_number
            print(f"Object {object_index} track set to {track_number}")
        
    def start_recording_handler(self, unused_addr, gesture_label, object_index=0):
        """Start recording samples for a specific gesture with a specific object"""
        self.recording_gesture = True
        self.current_gesture_label = gesture_label
        self.current_training_object = object_index
        if object_index < self.max_objects:
            self.position_histories[object_index].clear()
        print(f"Started recording samples for gesture: {gesture_label} with object {object_index}")
        
    def stop_recording_handler(self, unused_addr):
        """Stop recording samples"""
        self.recording_gesture = False
        self.current_gesture_label = None
        self.current_training_object = 0
        print("Stopped recording samples")
    
    def save_model_handler(self, unused_addr):
        """Train and save the model with recorded samples"""
        if len(self.training_data) < 10:
            print("Not enough training samples. Record more gestures first.")
            self.client_processing.send_message("/training/status", ["error", "Need at least 10 samples"])
            return
            
        # Print statistics about the training data
        labels, counts = np.unique(self.training_labels, return_counts=True)
        print("\nTraining data summary:")
        for label, count in zip(labels, counts):
            print(f"  {label}: {count} samples")
            
        # Train the model
        X = np.array(self.training_data)
        y = np.array(self.training_labels)
        
        # Scale the features
        self.object_scaler.fit(X)
        X_scaled = self.object_scaler.transform(X)
        
        # Train the model
        self.object_model.fit(X_scaled, y)
        self.object_model_exists = True
        
        # Save the model
        with open('gesture_model.pkl', 'wb') as f:
            pickle.dump(self.object_model, f)
            pickle.dump(self.object_scaler, f)
        
        print(f"Model trained and saved with {len(self.training_data)} samples.")
        self.client_processing.send_message("/training/status", ["success", f"Model saved with {len(self.training_data)} samples"])
    
    def extract_gesture_features(self, object_index):
        """Extract features from position history for gesture recognition"""
        # Convert deque to numpy array for easier processing
        positions = np.array(list(self.position_histories[object_index]))
        
        # Calculate total movement - skip if too small
        total_movement = self.calculate_total_movement(positions)
        if total_movement < self.min_movement:
            return None
            
        # Calculate displacement between consecutive points
        displacements = np.diff(positions, axis=0)
        
        # Basic statistics of the trajectory
        features = []
        
        # Direction changes in X and Y
        direction_changes_x = np.sum(np.diff(np.sign(displacements[:, 0])) != 0)
        direction_changes_y = np.sum(np.diff(np.sign(displacements[:, 1])) != 0)
        
        # Mean and std of velocities
        mean_velocity_x = np.mean(np.abs(displacements[:, 0]))
        mean_velocity_y = np.mean(np.abs(displacements[:, 1]))
        std_velocity_x = np.std(displacements[:, 0])
        std_velocity_y = np.std(displacements[:, 1])
        
        # Total vertical and horizontal displacement
        total_disp_x = np.abs(positions[-1, 0] - positions[0, 0])
        total_disp_y = np.abs(positions[-1, 1] - positions[0, 1])
        
        # Ratio of vertical to horizontal movement
        ratio_yx = np.sum(np.abs(displacements[:, 1])) / (np.sum(np.abs(displacements[:, 0])) + 0.0001)
        
        # Add features to the list
        features.extend([
            direction_changes_x, direction_changes_y,
            mean_velocity_x, mean_velocity_y,
            std_velocity_x, std_velocity_y,
            total_disp_x, total_disp_y,
            ratio_yx
        ])
        
        return features
    
    def analyze_gesture(self, object_index):
        """Analyze the current gesture and send appropriate commands"""
        try:
            # Extract features
            features = self.extract_gesture_features(object_index)
            
            # Skip if movement is too small
            if features is None:
                return
            
            # Scale features
            features_scaled = self.object_scaler.transform([features])
            
            # Predict gesture
            gesture = self.object_model.predict(features_scaled)[0]
            confidence = np.max(self.object_model.predict_proba(features_scaled))
            
            if confidence > self.confidence_threshold:  # Only accept confident predictions
                print(f"Detected gesture: {gesture} from object {object_index} (confidence: {confidence:.2f})")
                
                # Send gesture information back to Processing
                self.client_processing.send_message("/gesture/detected", [gesture, confidence, object_index])
                
                # Calculate movement magnitude for scaling adjustments
                positions = np.array(list(self.position_histories[object_index]))
                movement_magnitude = self.calculate_total_movement(positions)
                
                # Handle specific gestures
                if gesture == "volume_up":
                    self.adjust_volume(object_index, 0.1, movement_magnitude)
                elif gesture == "volume_down":
                    self.adjust_volume(object_index, -0.1, movement_magnitude)
                elif gesture == "next_track":
                    self.change_track(object_index, 1)
                elif gesture == "prev_track":
                    self.change_track(object_index, -1)
        except Exception as e:
            print(f"Error analyzing gesture for object {object_index}: {e}")
    
    def adjust_volume(self, object_index, amount, movement_magnitude=1.0):
        """Send volume adjustment to Ableton for a specific object/track"""
        # Scale the amount based on movement magnitude
        # Limit the scaling to reasonable bounds
        scale_factor = min(2.0, max(0.5, movement_magnitude * 2))
        scaled_amount = amount * scale_factor
        
        print(f"Adjusting volume for object {object_index} by {scaled_amount:.2f} (base: {amount}, magnitude: {movement_magnitude:.2f})")
        
        # Send to Processing for actual volume adjustment
        self.client_processing.send_message("/control/volume_adjust", [object_index, scaled_amount])
    
    def change_track(self, object_index, direction):
        """Change track in Ableton for a specific object"""
        if direction > 0:
            self.current_tracks[object_index] += 1
        else:
            self.current_tracks[object_index] = max(0, self.current_tracks[object_index] - 1)
        
        self.client_processing.send_message("/control/track_change", [object_index, self.current_tracks[object_index]])
        
    def sensor_handler(self, address, x, y, z):
        """Handle incoming sensor data from phone (from run_conductor.py)"""
        current_time = time.time()
        self.live_data_buffer.append((current_time, address, x, y, z))
        
        # Prune old data from the buffer
        while self.live_data_buffer and current_time - self.live_data_buffer[0][0] > self.analysis_window_seconds:
            self.live_data_buffer.popleft()

    def predict_and_perform_sensor(self):
        """
        Analyze sensor data for gestures (from run_conductor.py)
        """
        current_time = time.time()
        
        # Check if cooldown has passed
        if current_time - self.last_prediction_time < self.prediction_cooldown_seconds:
            return
            
        if len(self.live_data_buffer) < 20: # Make sure we have enough data to analyze
            return
        
        if not self.sensor_model_exists:
            return

        # Feature Engineering (same as in training)
        df = pd.DataFrame(list(self.live_data_buffer), columns=['timestamp', 'sensor', 'x', 'y', 'z'])
        
        features = []
        for sensor_type in ['/accelerometer', '/gyroscope']:
            sensor_data = df[df['sensor'] == sensor_type]
            if not sensor_data.empty:
                for axis in ['x', 'y', 'z']:
                    axis_data = sensor_data[axis]
                    # Calculate features and handle NaN values
                    mean_val = axis_data.mean()
                    std_val = axis_data.std()
                    min_val = axis_data.min()
                    max_val = axis_data.max()
                    abs_mean_val = np.abs(axis_data).mean()
                    
                    # Replace NaN values with 0
                    features.extend([
                        0.0 if np.isnan(mean_val) else mean_val,
                        0.0 if np.isnan(std_val) else std_val,
                        0.0 if np.isnan(min_val) else min_val,
                        0.0 if np.isnan(max_val) else max_val,
                        0.0 if np.isnan(abs_mean_val) else abs_mean_val
                    ])
            else:
                features.extend([0] * 5 * 3)

        # Scale and Predict
        # Additional safety check: replace any remaining NaN values
        features = [0.0 if np.isnan(f) or np.isinf(f) else f for f in features]
        
        if self.sensor_scaler is not None and self.sensor_model is not None:
            features_scaled = self.sensor_scaler.transform([features])
            
            # Final safety check before prediction
            if np.isnan(features_scaled).any() or np.isinf(features_scaled).any():
                print("ERROR: NaN or Inf values detected in scaled features, skipping prediction")
                return
            
            prediction_proba = self.sensor_model.predict_proba(features_scaled)[0]
            confidence = max(prediction_proba)
            gesture = self.sensor_model.classes_[np.argmax(prediction_proba)]

            # Only act on confident predictions
            if confidence > 0.5:
                print(f"SENSOR GESTURE DETECTED! '{gesture}' with {confidence:.2f} confidence")
                
                # Apply the sensor-based gesture actions
                self.handle_sensor_gesture(gesture, confidence)
                
                # Reset the buffer and start cooldown
                self.live_data_buffer.clear()
                self.last_prediction_time = current_time

    def handle_sensor_gesture(self, gesture, confidence):
        """Handle sensor-based gesture actions (from run_conductor.py)"""
        # Use track 0 as default for sensor gestures (can be made configurable)
        default_track = 0
        
        if gesture == 'stir':
            print(f"Firing clip for track {default_track}")
            # Send both the track index and clip slot index (0) to trigger the clip
            self.client_ableton.send_message("/live/clip/fire", [default_track, 0])
                
        elif gesture == 'wave':
            # Turn off soloing on all tracks
            print("Wave gesture detected - turning off all solos")
            for track_idx in range(self.max_objects):
                self.client_ableton.send_message("/live/track/set/solo", [track_idx, 0])
                print("All track solos disabled")
        
        elif gesture == 'circle':
            # Control Space and Reverb parameters specifically
            print("Circle gesture detected - modulating Space and Reverb!")
            
            # First, query current values to see what they are before we change them
            print("Querying current parameter values...")
            self.client_ableton.send_message("/live/device/get/parameter/value", [0, 0, 4])  # Space
            self.client_ableton.send_message("/live/device/get/parameter/value", [1, 0, 3])  # Reverb
            
            # Track 0 (Outland Bells): Control "Space" parameter [4] (range 0-127)
            space_value = np.random.randint(25, 115)  # Random value between 25-115 for dramatic but musical effect
            print(f"Setting Space parameter [4] to: {space_value}")
            self.client_ableton.send_message("/live/device/set/parameter/value", [0, 0, 4, space_value])
            
            # Track 1 (Dry Session Kit): Control "Reverb" parameter [3] (range 0-127)
            reverb_value = np.random.randint(10, 100)  # Random value between 10-100 for controlled reverb
            print(f"Setting Reverb parameter [3] to: {reverb_value}")
            self.client_ableton.send_message("/live/device/set/parameter/value", [1, 0, 3, reverb_value])
            
            # Query values again after setting to verify they changed
            time.sleep(0.1)  # Small delay to allow Ableton to process
            print("Querying parameter values after setting...")
            self.client_ableton.send_message("/live/device/get/parameter/value", [0, 0, 4])  # Space
            self.client_ableton.send_message("/live/device/get/parameter/value", [1, 0, 3])  # Reverb
            
            print(f"Applied Space modulation: {space_value:.2f}, Reverb modulation: {reverb_value:.2f}")

        elif gesture == 'crescendo':
            # Solo each track alternatively
            print("Crescendo gesture detected - soloing tracks alternatively")
            
            # Get current time to determine which track to solo
            current_time = time.time()
            track_to_solo = int(current_time) % self.max_objects  # Cycle through available tracks
            
            # Unsolo all tracks first
            for track_idx in range(self.max_objects):
                self.client_ableton.send_message("/live/track/set/solo", [track_idx, 0])
                
            # Solo the selected track
            self.client_ableton.send_message("/live/track/set/solo", [track_to_solo, 1])
            print(f"Soloing track {track_to_solo}")

    def query_device_parameters(self, max_tracks=5, max_devices=3):
        """Query and print all available device parameters for debugging/setup"""
        print("\n" + "="*60)
        print("QUERYING ABLETON LIVE DEVICE PARAMETERS")
        print("="*60)
        
        for track_idx in range(max_tracks):
            print(f"\n--- TRACK {track_idx} ---")
            
            # Get number of devices on this track
            try:
                # Query track devices - note: this is a simplified approach
                # In practice, you'd need to implement OSC response listening
                print(f"Checking devices on track {track_idx}...")
                
                for device_idx in range(max_devices):
                    print(f"\n  Device {device_idx}:")
                    
                    # Get device name
                    print("    Querying device name...")
                    self.client_ableton.send_message("/live/device/get/name", [track_idx, device_idx])
                    
                    # Get number of parameters
                    print("    Querying parameter count...")
                    self.client_ableton.send_message("/live/device/get/num_parameters", [track_idx, device_idx])
                    
                    # Get all parameter names
                    print("    Querying parameter names...")
                    self.client_ableton.send_message("/live/device/get/parameters/name", [track_idx, device_idx])
                    
                    # Small delay to avoid overwhelming
                    time.sleep(0.1)
                    
            except Exception as e:
                print(f"    Error querying track {track_idx}, device {device_idx}: {e}")
        
        print("\n" + "="*60)
        print("PARAMETER QUERY COMPLETE")
        print("Note: Responses will be sent back via OSC.")
        print("You may need to set up response listeners to see the actual data.")
        print("="*60)

    def setup_parameter_response_listeners(self):
        """Set up OSC listeners to capture parameter query responses"""
        print("Setting up parameter response listeners...")
        
        # Add handlers for device parameter responses
        self.dispatcher.map("/live/device/get/name", self.handle_device_name_response)
        self.dispatcher.map("/live/device/get/num_parameters", self.handle_num_parameters_response)
        self.dispatcher.map("/live/device/get/parameters/name", self.handle_parameter_names_response)
        # Add handler for parameter value responses
        self.dispatcher.map("/live/device/get/parameter/value", self.handle_parameter_value_response)
    
    def handle_device_name_response(self, unused_addr, track_idx, device_idx, name):
        """Handle device name response"""
        print(f"DEVICE: Track {track_idx}, Device {device_idx} = '{name}'")
    
    def handle_num_parameters_response(self, unused_addr, track_idx, device_idx, num_params):
        """Handle number of parameters response"""
        print(f"PARAMS COUNT: Track {track_idx}, Device {device_idx} has {num_params} parameters")
    
    def handle_parameter_names_response(self, unused_addr, track_idx, device_idx, *param_names):
        """Handle parameter names response"""
        print(f"PARAM NAMES: Track {track_idx}, Device {device_idx}:")
        for i, name in enumerate(param_names):
            print(f"  [{i}] {name}")
    
    def handle_parameter_value_response(self, unused_addr, track_idx, device_idx, param_idx, value):
        """Handle parameter value response"""
        print(f"PARAM VALUE: Track {track_idx}, Device {device_idx}, Parameter {param_idx} = {value}")

    def run(self):
        """Run the server with both object tracking and sensor prediction"""
        print("Combined gesture recognizer is running. Press Ctrl+C to stop.")
        print("Listening for:")
        print("  - Object tracking data on /tracking/position")
        print("  - Phone sensor data on /accelerometer and /gyroscope")
        
        try:
            # Instead of serve_forever, we run a loop to periodically call sensor predictor
            while True:
                self.server.handle_request()  # Handle one incoming request
                self.predict_and_perform_sensor()  # Check if we should predict from sensor data
                time.sleep(0.01)  # Small sleep to prevent 100% CPU usage
        except KeyboardInterrupt:
            self.server.shutdown()
            print("Combined gesture recognizer stopped.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-port", default=12000, type=int, help="Port for object tracking data")
    parser.add_argument("--phone-port", default=6448, type=int, help="Port for phone sensor data")
    parser.add_argument("--processing-port", default=12001, type=int)
    parser.add_argument("--ableton-port", default=11000, type=int)
    parser.add_argument("--confidence-threshold", default=0.75, type=float,
                       help="Minimum confidence level for gesture recognition (0.0-1.0)")
    parser.add_argument("--gesture-cooldown", default=1.0, type=float,
                       help="Seconds between gesture recognitions")
    parser.add_argument("--min-movement", default=0.2, type=float,
                       help="Minimum movement required for gesture recognition")
    parser.add_argument("--query-parameters", action="store_true",
                       help="Query and display all available device parameters")
    args = parser.parse_args()
    
    recognizer = GestureRecognizer(
        input_port=args.input_port,
        phone_port=args.phone_port,
        processing_port=args.processing_port,
        ableton_port=args.ableton_port,
        confidence_threshold=args.confidence_threshold,
        gesture_cooldown=args.gesture_cooldown,
        min_movement=args.min_movement
    )
    
    if args.query_parameters:
        print("Parameter query mode - will query parameters and then run normally")
        recognizer.query_device_parameters()
        time.sleep(2)  # Give time for responses
    
    recognizer.run()