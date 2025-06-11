# run_conductor.py
import time
import numpy as np
import joblib
from collections import deque
from pythonosc import dispatcher, osc_server, udp_client
import pandas as pd

# --- Configuration ---
# For listening to the phone
PHONE_IP = "0.0.0.0"
PHONE_PORT = 6448

# For sending to Ableton
# ABLETON_IP = "172.16.0.3"
ABLETON_IP = "192.168.25.240"
# ABLETON_IP = "127.0.0.1"
ABLETON_PORT = 11000


# Model and Analysis settings
MODEL_FILE = 'conductor_model.joblib'
SCALER_FILE = 'scaler.joblib'
ANALYSIS_WINDOW_SECONDS = 1.5 # Must be same as recording duration
PREDICTION_COOLDOWN_SECONDS = 2.0 # Wait 2s between predictions

# --- Global variables ---
live_data_buffer = deque()
last_prediction_time = 0
model = joblib.load(MODEL_FILE)
scaler = joblib.load(SCALER_FILE)
client = udp_client.SimpleUDPClient(ABLETON_IP, ABLETON_PORT)

# Tap tempo variables
tap_times = deque(maxlen=4)  # Store last 4 tap times for averaging
last_tap_time = 0
tap_timeout = 3.0  # Reset tap sequence if no tap for 3 seconds

# --- OSC Handler for incoming phone data ---
def sensor_handler(address, x, y, z):
    global live_data_buffer
    current_time = time.time()
    live_data_buffer.append((current_time, address, x, y, z))
    
    # Prune old data from the buffer
    while live_data_buffer and current_time - live_data_buffer[0][0] > ANALYSIS_WINDOW_SECONDS:
        live_data_buffer.popleft()

def tap_tempo_handler(address, *args):
    """
    Handle tap tempo input from /touch1 endpoint
    Calculate BPM based on time between taps and send to Ableton
    """
    global tap_times, last_tap_time
    current_time = time.time()
    
    # Check if this tap is too far from the last one (reset sequence)
    if current_time - last_tap_time > tap_timeout:
        tap_times.clear()
        print("Tap tempo sequence reset")
    
    # Add this tap time
    tap_times.append(current_time)
    last_tap_time = current_time
    
    print(f"Tap! ({len(tap_times)} taps recorded)")
    
    # Need at least 2 taps to calculate tempo
    if len(tap_times) >= 2:
        # Calculate intervals between taps
        intervals = []
        for i in range(1, len(tap_times)):
            intervals.append(tap_times[i] - tap_times[i-1])
        
        # Average interval in seconds
        avg_interval = sum(intervals) / len(intervals)
        
        # Convert to BPM (60 seconds / interval = beats per minute)
        bpm = 60.0 / avg_interval
        
        # Constrain to reasonable BPM range
        bpm = max(60, min(200, bpm))
        
        print(f"Calculated BPM: {bpm:.1f} (from {len(tap_times)} taps)")
        
        # Send to Ableton
        client.send_message("/live/song/set/tempo", bpm)
        print(f"Set Ableton tempo to {bpm:.1f} BPM")

def predict_and_perform(address, *args):
    """
    This function is called periodically, not by an OSC message.
    It checks if enough time has passed to make a new prediction.
    """
    global last_prediction_time
    current_time = time.time()
    
    # Check if cooldown has passed
    if current_time - last_prediction_time < PREDICTION_COOLDOWN_SECONDS:
        return
        
    if len(live_data_buffer) < 20: # Make sure we have enough data to analyze
        return
    
    # print(f"Analyzing buffer of size: {len(live_data_buffer)}")

    # Feature Engineering (same as in training)
    df = pd.DataFrame(list(live_data_buffer), columns=['timestamp', 'sensor', 'x', 'y', 'z'])
    
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
    
    features_scaled = scaler.transform([features])
    
    # Final safety check before prediction
    if np.isnan(features_scaled).any() or np.isinf(features_scaled).any():
        print("ERROR: NaN or Inf values detected in scaled features, skipping prediction")
        return
    
    prediction_proba = model.predict_proba(features_scaled)[0]
    confidence = max(prediction_proba)
    gesture = model.classes_[np.argmax(prediction_proba)]

    # print(f"Model Prediction: '{gesture}', Confidence: {confidence:.2f}")

    # Only act on confident predictions
    if confidence > 0.5:
        print(f"CONFIDENCE THRESHOLD MET! Detected gesture: '{gesture}'")
        print(f"Detected gesture: '{gesture}' with {confidence:.2f} confidence")
        
        # --- ACTION MAPPING ---
        if gesture == 'stir':
            # Toggle play/stop for the first clip slot (track 0, clip 0)
            print("Sending OSC: /live/clip_slot/stop [0, 0]")
            client.send_message("/live/clip_slot/stop", [0, 0])
            print("Sending OSC: /live/song/start_playing")
            client.send_message("/live/song/start_playing", [])

        elif gesture == 'wave':
            # Turn off soloing on all tracks
            print("Wave gesture detected - turning off all solos")
            for track_idx in range(3):
                client.send_message("/live/track/set/solo", [track_idx, 0])
                print("All track solos disabled")
        
        elif gesture == 'circle':
            # Do something funky and aggressive: dramatic parameter changes (NO TEMPO CHANGE)
            print("Circle gesture detected - going absolutely WILD!")
            
            # Super aggressive parameter modulation on multiple tracks
            for track_idx in range(3):  # First 3 tracks
                # Extreme filter parameters - full chaos mode
                filter_cutoff = np.random.uniform(0.05, 1.0)  # Even more extreme filter sweeps
                filter_resonance = np.random.uniform(0.0, 0.95)  # Maximum resonance for screaming filters
                client.send_message("/live/device/set/parameter/value", [track_idx, 0, 4, filter_cutoff])
                client.send_message("/live/device/set/parameter/value", [track_idx, 0, 5, filter_resonance])
                
                # Crazy LFO and modulation parameters
                lfo_rate = np.random.uniform(0.05, 1.0)  # From super slow to super fast
                lfo_amount = np.random.uniform(0.2, 0.9)  # Higher minimum for more dramatic effect
                client.send_message("/live/device/set/parameter/value", [track_idx, 0, 6, lfo_rate])
                client.send_message("/live/device/set/parameter/value", [track_idx, 0, 7, lfo_amount])
                
                # Extreme synthesis parameters
                envelope_decay = np.random.uniform(0.1, 1.0)  # Full envelope range
                distortion_amount = np.random.uniform(0.0, 0.85)  # More aggressive distortion
                client.send_message("/live/device/set/parameter/value", [track_idx, 0, 8, envelope_decay])
                client.send_message("/live/device/set/parameter/value", [track_idx, 0, 9, distortion_amount])
                
                # Additional wild parameters - hit even more controls
                extra_param1 = np.random.uniform(0.0, 1.0)  # Random parameter 10
                extra_param2 = np.random.uniform(0.0, 1.0)  # Random parameter 11
                client.send_message("/live/device/set/parameter/value", [track_idx, 0, 10, extra_param1])
                client.send_message("/live/device/set/parameter/value", [track_idx, 0, 11, extra_param2])
            
            # Dramatic continuous reverb control - no more on/off, full range chaos
            reverb_amount = np.random.uniform(0.0, 1.0)  # Full reverb range from dry to soaking wet
            reverb_decay = np.random.uniform(0.2, 0.95)  # Reverb decay time
            reverb_predelay = np.random.uniform(0.0, 0.8)  # Reverb pre-delay
            
            client.send_message("/live/device/set/enabled", [1, 1, 1])  # Always enable reverb device
            client.send_message("/live/device/set/parameter/value", [1, 1, 0, reverb_amount])  # reverb mix
            client.send_message("/live/device/set/parameter/value", [1, 1, 1, reverb_decay])  # reverb decay
            client.send_message("/live/device/set/parameter/value", [1, 1, 2, reverb_predelay])  # reverb predelay
            
            # Extreme track volume and pan modulation
            for track_idx in range(3):
                volume_change = np.random.uniform(0.4, 1.0)  # More dramatic volume range
                pan_position = np.random.uniform(0.0, 1.0)  # Random stereo positioning
                client.send_message("/live/track/set/volume", [track_idx, volume_change])
                client.send_message("/live/track/set/panning", [track_idx, pan_position])
            
            # Random track muting/unmuting for rhythmic chaos
            for track_idx in range(3):
                mute_chance = int(np.random.choice([0, 1], p=[0.7, 0.3]))  # 30% chance to mute
                client.send_message("/live/track/set/mute", [track_idx, mute_chance])
            
            print(f"Applied EXTREME modulation - Filters: {filter_cutoff:.2f}/{filter_resonance:.2f}, Reverb: {reverb_amount:.2f}, LFO madness engaged!")

        elif gesture == 'crescendo':
            # Solo each track alternatively
            print("Crescendo gesture detected - soloing tracks alternatively")
            
            # Get current time to determine which track to solo
            track_to_solo = int(current_time) % 3  # Cycle through tracks 0, 1, 2
            
            # Unsolo all tracks first
            for track_idx in range(3):
                client.send_message("/live/track/set/solo", [track_idx, 0])
                
                # Solo the selected track
                client.send_message("/live/track/set/solo", [track_to_solo, 1])
                print(f"Soloing track {track_to_solo}")

        # Reset the buffer and start cooldown
        live_data_buffer.clear()
        last_prediction_time = current_time

# --- Main Setup ---
if __name__ == "__main__":
    disp = dispatcher.Dispatcher()
    disp.map("/*", sensor_handler) # Catch all sensor data
    disp.map("/touch1", tap_tempo_handler) # Tap tempo handler for touch input
    # We add a default handler that we will call in our main loop
    disp.set_default_handler(predict_and_perform)

    server = osc_server.ThreadingOSCUDPServer((PHONE_IP, PHONE_PORT), disp)
    print(f"Conductor is running. Listening on {server.server_address}, sending to Ableton on {ABLETON_IP}:{ABLETON_PORT}")
    
    try:
        # Instead of serve_forever, we run a loop to periodically call our predictor
        while True:
            server.handle_request() # Handle one incoming request
            predict_and_perform(None) # Check if we should predict
            time.sleep(0.01) # Small sleep to prevent 100% CPU usage
    except KeyboardInterrupt:
        server.shutdown()
        print("Conductor stopped.")