#!/usr/bin/env python3
"""
Standalone script to query Ableton Live device parameters via AbletonOSC.
This will help you discover what parameters are available for control.
"""

import time
import argparse
from pythonosc import dispatcher, osc_server, udp_client

class AbletonParameterQuerier:
    def __init__(self, ableton_port=11000, listen_port=11001):
        self.client_ableton = udp_client.SimpleUDPClient("127.0.0.1", ableton_port)
        
        # Set up OSC server to receive responses
        self.dispatcher = dispatcher.Dispatcher()
        self.setup_response_handlers()
        
        # Server to listen for responses from Ableton
        self.server = osc_server.ThreadingOSCUDPServer(
            ("127.0.0.1", listen_port), self.dispatcher)
        
        print(f"Listening for responses on port {listen_port}")
        print(f"Sending queries to Ableton on port {ableton_port}")
        
        # Start the server in a separate thread
        import threading
        self.server_thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.server_thread.start()
    
    def setup_response_handlers(self):
        """Set up handlers for Ableton's responses"""
        self.dispatcher.map("/live/device/get/name", self.handle_device_name)
        self.dispatcher.map("/live/device/get/num_parameters", self.handle_num_parameters)
        self.dispatcher.map("/live/device/get/parameters/name", self.handle_parameter_names)
        self.dispatcher.map("/live/track/get/num_devices", self.handle_num_devices)
        self.dispatcher.map("/live/device/get/class_name", self.handle_device_class)
        self.dispatcher.map("/live/device/get/type", self.handle_device_type)
    
    def handle_device_name(self, unused_addr, track_idx, device_idx, name):
        print(f"  Device [{device_idx}]: {name}")
    
    def handle_device_class(self, unused_addr, track_idx, device_idx, class_name):
        print(f"    Class: {class_name}")
    
    def handle_device_type(self, unused_addr, track_idx, device_idx, device_type):
        print(f"    Type: {device_type}")
    
    def handle_num_devices(self, unused_addr, track_idx, num_devices):
        print(f"Track {track_idx}: {num_devices} devices")
    
    def handle_num_parameters(self, unused_addr, track_idx, device_idx, num_params):
        print(f"    Parameters: {num_params} total")
    
    def handle_parameter_names(self, unused_addr, track_idx, device_idx, *param_names):
        print(f"    Parameter List:")
        for i, name in enumerate(param_names):
            print(f"      [{i:2}] {name}")
        print()
    
    def query_all_parameters(self, max_tracks=8, max_devices=8):
        """Query all device parameters in your Live set"""
        print("="*70)
        print("QUERYING ABLETON LIVE DEVICE PARAMETERS")
        print("="*70)
        print("Make sure AbletonOSC is loaded in Ableton Live!")
        print()
        
        for track_idx in range(max_tracks):
            print(f"\n--- TRACK {track_idx} ---")
            
            # First get number of devices on track
            self.client_ableton.send_message("/live/track/get/num_devices", [track_idx])
            time.sleep(0.1)
            
            # Query each device
            for device_idx in range(max_devices):
                try:
                    # Get device info
                    self.client_ableton.send_message("/live/device/get/name", [track_idx, device_idx])
                    time.sleep(0.05)
                    
                    self.client_ableton.send_message("/live/device/get/class_name", [track_idx, device_idx])
                    time.sleep(0.05)
                    
                    self.client_ableton.send_message("/live/device/get/type", [track_idx, device_idx])
                    time.sleep(0.05)
                    
                    # Get parameter count
                    self.client_ableton.send_message("/live/device/get/num_parameters", [track_idx, device_idx])
                    time.sleep(0.05)
                    
                    # Get all parameter names
                    self.client_ableton.send_message("/live/device/get/parameters/name", [track_idx, device_idx])
                    time.sleep(0.2)  # Give more time for parameter names response
                    
                except Exception as e:
                    if device_idx == 0:  # Only show error for first device
                        print(f"  No devices found on track {track_idx}")
                    break
        
        print("="*70)
        print("QUERY COMPLETE!")
        print("Use the parameter indices [0], [1], etc. with:")
        print("/live/device/set/parameter/value [track] [device] [param_index] [value]")
        print("="*70)
    
    def query_specific_device(self, track_idx, device_idx):
        """Query a specific device's parameters"""
        print(f"\n--- TRACK {track_idx}, DEVICE {device_idx} ---")
        
        # Get device info
        self.client_ableton.send_message("/live/device/get/name", [track_idx, device_idx])
        time.sleep(0.1)
        
        self.client_ableton.send_message("/live/device/get/class_name", [track_idx, device_idx])
        time.sleep(0.1)
        
        # Get parameters
        self.client_ableton.send_message("/live/device/get/parameters/name", [track_idx, device_idx])
        time.sleep(0.3)
    
    def shutdown(self):
        """Clean shutdown"""
        self.server.shutdown()

def main():
    parser = argparse.ArgumentParser(description="Query Ableton Live device parameters")
    parser.add_argument("--ableton-port", default=11000, type=int, 
                       help="Port to send to AbletonOSC (default: 11000)")
    parser.add_argument("--listen-port", default=11001, type=int,
                       help="Port to listen for responses (default: 11001)")
    parser.add_argument("--max-tracks", default=8, type=int,
                       help="Maximum number of tracks to query")
    parser.add_argument("--max-devices", default=8, type=int,
                       help="Maximum number of devices per track to query")
    parser.add_argument("--track", type=int, help="Query specific track only")
    parser.add_argument("--device", type=int, help="Query specific device only (requires --track)")
    
    args = parser.parse_args()
    
    querier = AbletonParameterQuerier(args.ableton_port, args.listen_port)
    
    try:
        if args.track is not None:
            if args.device is not None:
                querier.query_specific_device(args.track, args.device)
            else:
                print(f"Querying all devices on track {args.track}")
                for device_idx in range(args.max_devices):
                    querier.query_specific_device(args.track, device_idx)
                    time.sleep(0.2)
        else:
            querier.query_all_parameters(args.max_tracks, args.max_devices)
        
        # Give time for all responses
        print("\nWaiting for responses...")
        time.sleep(3)
        
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        querier.shutdown()

if __name__ == "__main__":
    main()
