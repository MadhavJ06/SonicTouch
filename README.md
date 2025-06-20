# SonicTouch: Gesture-Based Music interactive System

## Overview

SonicTouch enables the control musical parameters through intuitive gestures using colored objects with camera tracking and gesture recognition using sensors on a mobile device (e.g. phone). The system provides an engaging platform for physical rehabilitation by mapping movement patterns to real-time audio feedback in Ableton Live.

## Key Features

- **Multi-Object Tracking**: Simultaneously track up to 4 differently colored objects
- **Machine Learning Gesture Recognition**: Train custom gestures using Sensors2OSC
- **Real-Time Audio Control**: Direct integration with Ableton Live via AbletonOSC
- **Dual Control Modes**:
  - Direct control (position mapping to volume/pan)
  - ML gesture recognition for complex movements for device parameter control in Ableton Live
- **Phone Sensor Integration**: Additional control via smartphone accelerometer,gyroscope, light sensors etc.
- **Customizable Sensitivity and gestures**: Adaptable to different mobility levels

## System Architecture

The project consists of two main components:

1. **Processing Visual Interface** (`object.pde`)
   - Handles camera input and color-based object tracking
   - Provides visual feedback and user interface
   - Sends tracking data to Python backend via OSC

2. **Python Backend** (`object_ml.py`)
   - Processes tracking data using machine learning
   - Contains gesture recognition models
   - Communicates with Ableton Live for audio control
   - Supports phone sensor data integration

## Requirements

### Software Dependencies

```bash
# Python packages
pip install -r requirements.txt
# Processing IDE with libraries:
# - Video library (for camera input)
# - oscP5 library (for OSC communication)
# - netP5 library (for network communication)
```

### Hardware Requirements

- **Camera**: Webcam or built-in camera for object tracking
- **Colored Objects**: distinctly colored objects for tracking (e.g., colored balls, markers)
- **DAW**: Ableton Live (To use AbletonOSC)
- **Smartphone** : For sensor-based control using gestures

### Software Setup

1. **Ableton Live** with **AbletonOSC** plugin
   - Configure AbletonOSC to listen on port 11000
   - Enable OSC communication in Ableton preferences

2. **Processing IDE** (version 3.x or 4.x)
   - Install required libraries via Tools > Manage Tools

## Installation & Setup

1. **Clone or download the project files**

   ```bash
   git clone https://github.com/MadhavJ06/SonicTouch.git
   cd SonicTouch
   ```

2. **Install Python dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Set up Processing**
   - Install Processing IDE from https://processing.org/
   - Install required libraries: Video, oscP5, netP5

4. **Configure Ableton Live**
   - Install AbletonOSC plugin
   - Set OSC input port to 11000
   - Set OSC output port to 11001

## Usage Instructions

### Basic Setup

1. **Start Ableton Live** with [AbletonOSC](https://github.com/ideoforms/AbletonOSC)
2. **Run the Python backend**:

   ```bash
   python object_ml.py
   ```

3. **Open and run** `object.pde` in Processing IDE

### Object Tracking Setup

1. **Select objects to track**: Click on colored objects in the camera view
2. **Choose control mode**: Press 'm' to toggle between direct control and ML gestures
3. **Adjust sensitivity**: Use UP/DOWN arrows to adjust color tracking sensitivity
4. **Select active object**: Press 1-4 to switch between tracked objects

### Control Modes

#### Direct Control Mode

- **Y-axis position**: Controls track volume (up = louder, down = quieter)
- **X-axis position**: Controls track panning (left = left pan, right = right pan)

## References

[AbletonOSC](https://github.com/ideoforms/AbletonOSC)
