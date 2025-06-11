import argparse
import numpy as np
import time
import pickle
from pythonosc import dispatcher
from pythonosc import osc_server
from pythonosc import udp_client
from collections import deque
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

class GestureRecognizer:
    def __init__(self, input_port=12000, processing_port=12001, ableton_port=11000,
                 confidence_threshold=0.75, gesture_cooldown=1.0, min_movement=0.2):
        # OSC communication
        self.client_processing = udp_client.SimpleUDPClient("127.0.0.1", processing_port)
        self.client_ableton = udp_client.SimpleUDPClient("127.0.0.1", ableton_port)
        
        # Tracking data storage
        self.position_history = deque(maxlen=30)  # Store last 30 positions for gesture analysis
        self.last_gesture_time = time.time()
        self.gesture_cooldown = gesture_cooldown  # Seconds between gesture recognition attempts
        self.confidence_threshold = confidence_threshold  # Minimum confidence for gesture detection
        self.min_movement = min_movement  # Minimum movement for gesture detection
        self.current_track = 0
        
        # Load ML model (if it exists) or create an empty one
        try:
            with open('gesture_model.pkl', 'rb') as f:
                self.model = pickle.load(f)
                self.scaler = pickle.load(f)
                print("Loaded existing gesture model")
                self.model_exists = True
        except FileNotFoundError:
            self.model = RandomForestClassifier(n_estimators=50)
            self.scaler = StandardScaler()
            self.model_exists = False
            print("No gesture model found. Use training mode to create one.")
        
        # Training data storage
        self.training_data = []
        self.training_labels = []
        self.recording_gesture = False
        self.current_gesture_label = None
        self.training_feedback = {}  # Store feedback about recent training samples
        
        # Set up OSC server
        self.dispatcher = dispatcher.Dispatcher()
        self.dispatcher.map("/tracking/position", self.position_handler)
        self.dispatcher.map("/tracking/track", self.track_handler)
        self.dispatcher.map("/training/start", self.start_recording_handler)
        self.dispatcher.map("/training/stop", self.stop_recording_handler)
        self.dispatcher.map("/training/save", self.save_model_handler)
        
        # Start OSC server
        self.server = osc_server.ThreadingOSCUDPServer(
            ("127.0.0.1", input_port), self.dispatcher)
        print(f"Starting OSC server on 127.0.0.1:{input_port}")
        print(f"Sending to Processing on port {processing_port}")
        print(f"Sending to Ableton on port {ableton_port}")
        print(f"Recognition settings: confidence={confidence_threshold}, cooldown={gesture_cooldown}s, min_movement={min_movement}")
    
    def position_handler(self, unused_addr, x, y, detected):
        """Handle incoming position updates from Processing"""
        if not detected:
            self.position_history.clear()
            return
            
        # Add position to history
        self.position_history.append((x, y))
        
        # If we're recording a gesture for training
        if self.recording_gesture and self.current_gesture_label:
            if len(self.position_history) >= 15:  # Wait until we have enough positions
                # Check for significant movement
                positions = np.array(list(self.position_history))
                total_movement = self.calculate_total_movement(positions)
                
                # Only record samples with significant movement
                if total_movement > self.min_movement:
                    # Extract features and record
                    features = self.extract_gesture_features()
                    if features is not None:
                        self.training_data.append(features)
                        self.training_labels.append(self.current_gesture_label)
                        print(f"Recorded sample for gesture '{self.current_gesture_label}' (movement: {total_movement:.3f})")
                        
                        # Send feedback to Processing
                        self.client_processing.send_message("/training/sample", 
                                                           [self.current_gesture_label, total_movement])
        
        # If we have enough positions and not in training mode, analyze for gesture
        elif len(self.position_history) >= 15 and self.model_exists:
            current_time = time.time()
            if current_time - self.last_gesture_time > self.gesture_cooldown:
                self.analyze_gesture()
                self.last_gesture_time = current_time
    
    def calculate_total_movement(self, positions):
        """Calculate total movement in a sequence of positions"""
        if len(positions) < 2:
            return 0.0
            
        displacements = np.diff(positions, axis=0)
        total_movement = np.sum(np.sqrt(np.sum(displacements**2, axis=1)))
        return total_movement
    
    def track_handler(self, unused_addr, track_number):
        """Handle track change messages"""
        self.current_track = track_number
        print(f"Current track set to {track_number}")
        
    def start_recording_handler(self, unused_addr, gesture_label):
        """Start recording samples for a specific gesture"""
        self.recording_gesture = True
        self.current_gesture_label = gesture_label
        self.position_history.clear()
        print(f"Started recording samples for gesture: {gesture_label}")
        
    def stop_recording_handler(self, unused_addr):
        """Stop recording samples"""
        self.recording_gesture = False
        self.current_gesture_label = None
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
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)
        
        # Train the model
        self.model.fit(X_scaled, y)
        self.model_exists = True
        
        # Save the model
        with open('gesture_model.pkl', 'wb') as f:
            pickle.dump(self.model, f)
            pickle.dump(self.scaler, f)
        
        print(f"Model trained and saved with {len(self.training_data)} samples.")
        self.client_processing.send_message("/training/status", ["success", f"Model saved with {len(self.training_data)} samples"])
    
    def extract_gesture_features(self):
        """Extract features from position history for gesture recognition"""
        # Convert deque to numpy array for easier processing
        positions = np.array(list(self.position_history))
        
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
    
    def analyze_gesture(self):
        """Analyze the current gesture and send appropriate commands"""
        try:
            # Extract features
            features = self.extract_gesture_features()
            
            # Skip if movement is too small
            if features is None:
                return
            
            # Scale features
            features_scaled = self.scaler.transform([features])
            
            # Predict gesture
            gesture = self.model.predict(features_scaled)[0]
            confidence = np.max(self.model.predict_proba(features_scaled))
            
            if confidence > self.confidence_threshold:  # Only accept confident predictions
                print(f"Detected gesture: {gesture} (confidence: {confidence:.2f})")
                
                # Send gesture information back to Processing
                self.client_processing.send_message("/gesture/detected", [gesture, confidence])
                
                # Calculate movement magnitude for scaling adjustments
                positions = np.array(list(self.position_history))
                movement_magnitude = self.calculate_total_movement(positions)
                
                # Handle specific gestures
                if gesture == "volume_up":
                    self.adjust_volume(0.1, movement_magnitude)
                elif gesture == "volume_down":
                    self.adjust_volume(-0.1, movement_magnitude)
                elif gesture == "next_track":
                    self.change_track(1)
                elif gesture == "prev_track":
                    self.change_track(-1)
        except Exception as e:
            print(f"Error analyzing gesture: {e}")
    
    def adjust_volume(self, amount, movement_magnitude=1.0):
        """Send volume adjustment to Ableton"""
        # Scale the amount based on movement magnitude
        # Limit the scaling to reasonable bounds
        scale_factor = min(2.0, max(0.5, movement_magnitude * 2))
        scaled_amount = amount * scale_factor
        
        print(f"Adjusting volume by {scaled_amount:.2f} (base: {amount}, magnitude: {movement_magnitude:.2f})")
        
        # Send to Processing for actual volume adjustment
        self.client_processing.send_message("/control/volume_adjust", scaled_amount)
    
    def change_track(self, direction):
        """Change track in Ableton"""
        if direction > 0:
            self.current_track += 1
        else:
            self.current_track = max(0, self.current_track - 1)
        
        self.client_processing.send_message("/control/track_change", self.current_track)
        
    def run(self):
        """Run the server forever"""
        print("Gesture recognizer is running. Press Ctrl+C to stop.")
        self.server.serve_forever()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-port", default=12000, type=int)
    parser.add_argument("--processing-port", default=12001, type=int)
    parser.add_argument("--ableton-port", default=11000, type=int)
    parser.add_argument("--confidence-threshold", default=0.75, type=float,
                       help="Minimum confidence level for gesture recognition (0.0-1.0)")
    parser.add_argument("--gesture-cooldown", default=1.0, type=float,
                       help="Seconds between gesture recognitions")
    parser.add_argument("--min-movement", default=0.2, type=float,
                       help="Minimum movement required for gesture recognition")
    args = parser.parse_args()
    
    recognizer = GestureRecognizer(
        input_port=args.input_port,
        processing_port=args.processing_port,
        ableton_port=args.ableton_port,
        confidence_threshold=args.confidence_threshold,
        gesture_cooldown=args.gesture_cooldown,
        min_movement=args.min_movement
    )
    recognizer.run()