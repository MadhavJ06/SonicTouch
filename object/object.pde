import processing.video.*;
import oscP5.*;
import netP5.*;

// --- Camera settings ---
Capture cam;
int camWidth = 640;
int camHeight = 480;

// --- OSC communication ---
OscP5 oscP5;
NetAddress abletonAddress;
NetAddress pythonAddress;
boolean useMLGestures = true;

// --- Object tracking settings ---
color targetColor;
boolean colorSelected = false;
float colorSensitivity = 35;
PVector trackedPoint = new PVector(0, 0);
boolean objectDetected = false;
ArrayList<PVector> gestureTrail = new ArrayList<PVector>();
int maxTrailPoints = 30;

// --- Track settings ---
int currentTrack = 0;
float currentVolume = 0.7;
float previousVolume = -1;

// --- ML Gesture recognition variables ---
String lastDetectedGesture = "";
float gestureConfidence = 0.0;
int gestureDisplayTime = 0;

// --- Training mode variables ---
boolean trainingMode = false;
String[] availableGestures = {"volume_up", "volume_down", "next_track", "prev_track"};
int currentGestureIndex = 0;
boolean recordingGesture = false;
String trainingStatus = "";
int trainingStatusDisplayTime = 0;
float lastSampleQuality = 0.0;
int samplesRecorded = 0;

void settings() {
  size(camWidth, camHeight);
}

void setup() {
  // Initialize camera
  String[] cameras = Capture.list();
  if (cameras.length == 0) {
    println("No cameras available");
    exit();
  } else {
    println("Available cameras:");
    for (int i = 0; i < cameras.length; i++) {
      println(i + ": " + cameras[i]);
    }
    
    // Initialize with first camera
    cam = new Capture(this, cameras[0]);
    cam.start();
  }
  
  // Initialize OSC communication
  oscP5 = new OscP5(this, 12001);  // Listen on port 12001 for messages from Python
  abletonAddress = new NetAddress("127.0.0.1", 11000);  // Send to Ableton on 11000
  pythonAddress = new NetAddress("127.0.0.1", 12000);   // Send to Python ML on 12000
  
  // Request initial volume
  requestTrackVolume();
  
  println("------------------------------------");
  println("ML GESTURE CONTROL FOR ABLETON LIVE");
  println("------------------------------------");
  println("Click on an object to track its color");
  println("Press 'm' to toggle between direct control and ML gesture recognition");
  println("Press 't' to enter training mode");
  println("LEFT/RIGHT arrows change Ableton track");
  println("UP/DOWN arrows adjust color sensitivity");
  println("Press 'r' to reset color selection");
}

void draw() {
  // Update camera feed
  if (cam.available()) {
    cam.read();
  }
  
  // Display camera image
  image(cam, 0, 0);
  
  // Only track if a color has been selected
  if (colorSelected) {
    trackObject();
    
    // Send tracking data to Python ML if enabled
    if (useMLGestures && objectDetected) {
      sendTrackingData();
    }
  }
  
  // Draw UI elements
  drawUI();
  
  // Draw training mode UI if active
  if (trainingMode) {
    drawTrainingUI();
  }
}

void trackObject() {
  cam.loadPixels();
  
  float sumX = 0;
  float sumY = 0;
  int count = 0;
  
  // Search for pixels matching the target color
  for (int x = 0; x < cam.width; x++) {
    for (int y = 0; y < cam.height; y++) {
      int loc = x + y * cam.width;
      color currentColor = cam.pixels[loc];
      
      // Calculate color distance
      float r1 = red(currentColor);
      float g1 = green(currentColor);
      float b1 = blue(currentColor);
      
      float r2 = red(targetColor);
      float g2 = green(targetColor);
      float b2 = blue(targetColor);
      
      float d = dist(r1, g1, b1, r2, g2, b2);
      
      // If color is close enough to target, count it
      if (d < colorSensitivity) {
        sumX += x;
        sumY += y;
        count++;
      }
    }
  }
  
  // If enough matching pixels found, update tracked position
  if (count > 50) {
    objectDetected = true;
    
    // Calculate average position (centroid)
    trackedPoint.x = sumX / count;
    trackedPoint.y = sumY / count;
    
    // Add point to the gesture trail
    if (gestureTrail.size() >= maxTrailPoints) {
      gestureTrail.remove(0);
    }
    gestureTrail.add(new PVector(trackedPoint.x, trackedPoint.y));
    
    // If not using ML, directly map Y position to volume
    if (!useMLGestures) {
      // Map Y position to volume (top = max volume, bottom = min volume)
      float newVolume = map(trackedPoint.y, 0, camHeight, 1.0, 0.0);
      currentVolume = newVolume;
      
      // Send volume to Ableton if it changed enough
      if (abs(currentVolume - previousVolume) > 0.01) {
        previousVolume = currentVolume;
        setTrackVolume();
      }
    }
    
    // Draw the gesture trail if in training mode or recently detected gesture
    if (trainingMode || millis() < gestureDisplayTime + 2000) {
      // Draw the gesture path
      stroke(trainingMode ? color(255, 0, 0, 200) : color(255, 255, 0, 150));
      strokeWeight(trainingMode ? 3 : 2);
      noFill();
      beginShape();
      for (PVector p : gestureTrail) {
        vertex(p.x, p.y);
      }
      endShape();
    }
    
    // Draw tracking indicator
    noFill();
    stroke(0, 255, 0);
    strokeWeight(2);
    ellipse(trackedPoint.x, trackedPoint.y, 40, 40);
    
    // Draw cross at center
    line(trackedPoint.x - 10, trackedPoint.y, trackedPoint.x + 10, trackedPoint.y);
    line(trackedPoint.x, trackedPoint.y - 10, trackedPoint.x, trackedPoint.y + 10);
  } else {
    objectDetected = false;
  }
}

void drawUI() {
  // Background for UI elements
  fill(0, 150);
  noStroke();
  rect(0, 0, width, 60);
  rect(0, height-40, width, 40);
  
  // Show selected color
  fill(255);
  text("Selected Color:", 10, 20);
  
  if (colorSelected) {
    fill(targetColor);
    rect(120, 8, 30, 20);
  } else {
    fill(150);
    text("Click to select", 120, 20);
  }
  
  // Show current track and volume
  fill(255);
  text("Track: " + currentTrack, 200, 20);
  text("Volume: " + nf(currentVolume, 0, 2), 300, 20);
  text("Sensitivity: " + int(colorSensitivity), 430, 20);
  
  // Show control mode
  fill(255);
  text("Mode: " + (useMLGestures ? "ML GESTURES" : "DIRECT CONTROL"), 10, 40);
  
  // Show detected gesture if available
  if (useMLGestures && !lastDetectedGesture.equals("") && millis() < gestureDisplayTime + 2000) {
    fill(255, 255, 0);
    text("Gesture: " + lastDetectedGesture + " (" + nf(gestureConfidence, 0, 2) + ")", 200, 40);
  }
  
  // Show tracking status
  if (colorSelected) {
    fill(objectDetected ? color(0, 255, 0) : color(255, 0, 0));
    text(objectDetected ? "TRACKING" : "NOT DETECTED", 430, 40);
  }
  
  // Show controls reminder
  fill(255);
  String controls = "LEFT/RIGHT: Track | UP/DOWN: Sensitivity | M: Toggle Mode | T: Train | R: Reset";
  text(controls, 10, height-15);
  
  // Draw volume bar
  noStroke();
  fill(50);
  rect(width-30, 70, 20, height-140);
  
  if (objectDetected) {
    fill(0, 255, 0);
  } else if (colorSelected) {
    fill(255, 165, 0); // Orange
  } else {
    fill(150);
  }
  
  float barHeight = currentVolume * (height-140);
  rect(width-30, 70 + (height-140) - barHeight, 20, barHeight);
}

void drawTrainingUI() {
  // Semi-transparent overlay
  fill(0, 200);
  rect(0, 0, width, height);
  
  // Title
  fill(255);
  textSize(24);
  textAlign(CENTER, CENTER);
  text("GESTURE TRAINING MODE", width/2, 50);
  
  // Current gesture
  fill(255, 255, 0);
  text("Current gesture: " + availableGestures[currentGestureIndex], width/2, 100);
  text("Samples recorded: " + samplesRecorded, width/2, 130);
  
  // Recording status
  if (recordingGesture) {
    fill(255, 0, 0);
    ellipse(width/2, 160, 20, 20);
    fill(255);
    text("RECORDING SAMPLES", width/2, 190);
    
    // Show quality of last sample
    if (lastSampleQuality > 0) {
      fill(lastSampleQuality > 0.3 ? color(0, 255, 0) : color(255, 165, 0));
      text("Movement quality: " + nf(lastSampleQuality, 0, 2), width/2, 220);
    }
    
    fill(255);
    text("Move the object to demonstrate the gesture", width/2, 250);
  } else {
    fill(255);
    text("Press SPACE to start recording samples", width/2, 160);
  }
  
  // Display training status message if any
  if (!trainingStatus.equals("") && millis() < trainingStatusDisplayTime + 3000) {
    fill(255, 255, 0);
    text(trainingStatus, width/2, 280);
  }
  
  // Instructions
  fill(200);
  text("← → : Switch gesture type", width/2, height-120);
  text("SPACE : Start/stop recording samples", width/2, height-90);
  text("S : Save model and exit training", width/2, height-60);
  text("ESC : Exit training mode", width/2, height-30);
  
  textAlign(LEFT, BASELINE); // Reset text alignment
  textSize(16); // Reset text size
}

void sendTrackingData() {
  // Send tracked object position to Python ML
  OscMessage msg = new OscMessage("/tracking/position");
  msg.add((float) trackedPoint.x / camWidth); // Normalize x position
  msg.add((float) trackedPoint.y / camHeight); // Normalize y position
  msg.add(1); // Object is detected
  oscP5.send(msg, pythonAddress);
  
  // Also send current track
  OscMessage trackMsg = new OscMessage("/tracking/track");
  trackMsg.add(currentTrack);
  oscP5.send(trackMsg, pythonAddress);
}

void mousePressed() {
  // Ignore clicks when in training mode
  if (trainingMode) return;
  
  // Check if we're clicking in the camera image area (avoid UI elements)
  if (mouseY > 60 && mouseY < height - 40) {
    int loc = mouseX + mouseY * cam.width;
    
    if (loc >= 0 && loc < cam.pixels.length) {
      targetColor = cam.pixels[loc];
      colorSelected = true;
      gestureTrail.clear(); // Clear the gesture trail
      println("Selected color: R=" + red(targetColor) + ", G=" + green(targetColor) + ", B=" + blue(targetColor));
    }
  }
}

void keyPressed() {
  // Handle training mode specially
  if (trainingMode) {
    handleTrainingModeKeys();
    return;
  }
  
  // Toggle between direct control and ML gestures
  if (key == 'm' || key == 'M') {
    useMLGestures = !useMLGestures;
    println("Switched to " + (useMLGestures ? "ML GESTURE mode" : "DIRECT CONTROL mode"));
  }
  
  // Enter training mode
  if (key == 't' || key == 'T') {
    trainingMode = true;
    samplesRecorded = 0;
    gestureTrail.clear();
    println("Entered training mode");
  }
  
  // Track controls
  if (keyCode == LEFT) {
    if (currentTrack > 0) {
      currentTrack--;
      requestTrackVolume();
      sendTrackingData();  // Update Python with new track
      println("Switched to track " + currentTrack);
    }
  } else if (keyCode == RIGHT) {
    currentTrack++;
    requestTrackVolume();
    sendTrackingData();  // Update Python with new track
    println("Switched to track " + currentTrack);
  }
  
  // Sensitivity controls
  if (keyCode == UP) {
    colorSensitivity += 5;
    if (colorSensitivity > 150) colorSensitivity = 150;
    println("Color sensitivity: " + colorSensitivity);
  } else if (keyCode == DOWN) {
    colorSensitivity -= 5;
    if (colorSensitivity < 5) colorSensitivity = 5;
    println("Color sensitivity: " + colorSensitivity);
  }
  
  // Reset color selection
  if (key == 'r' || key == 'R') {
    colorSelected = false;
    gestureTrail.clear();
    println("Color selection reset");
  }
}

void handleTrainingModeKeys() {
  // Change current gesture
  if (keyCode == LEFT) {
    currentGestureIndex = (currentGestureIndex + availableGestures.length - 1) % availableGestures.length;
    samplesRecorded = 0;  // Reset sample count when changing gesture
  } else if (keyCode == RIGHT) {
    currentGestureIndex = (currentGestureIndex + 1) % availableGestures.length;
    samplesRecorded = 0;  // Reset sample count when changing gesture
  }
  
  // Start/stop recording samples for current gesture
  if (key == ' ') {
    recordingGesture = !recordingGesture;
    gestureTrail.clear(); // Clear the gesture trail
    
    if (recordingGesture) {
      // Start recording
      OscMessage msg = new OscMessage("/training/start");
      msg.add(availableGestures[currentGestureIndex]);
      oscP5.send(msg, pythonAddress);
      println("Started recording samples for " + availableGestures[currentGestureIndex]);
    } else {
      // Stop recording
      OscMessage msg = new OscMessage("/training/stop");
      oscP5.send(msg, pythonAddress);
      println("Stopped recording samples");
    }
  }
  
  // Save model
  if (key == 's' || key == 'S') {
    // Tell Python to save the model
    OscMessage msg = new OscMessage("/training/save");
    oscP5.send(msg, pythonAddress);
    trainingStatus = "Saving model...";
    trainingStatusDisplayTime = millis();
    println("Saving model and exiting training mode");
  }
  
  // Exit training mode without saving
  if (keyCode == ESC) {
    if (recordingGesture) {
      // Stop recording first
      OscMessage msg = new OscMessage("/training/stop");
      oscP5.send(msg, pythonAddress);
    }
    trainingMode = false;
    recordingGesture = false;
    key = 0; // Prevent ESC from closing the app
    println("Exited training mode without saving");
  }
}

// --- OSC Event Handlers ---
void oscEvent(OscMessage msg) {
  if (msg.addrPattern().equals("/gesture/detected")) {
    String gesture = msg.get(0).stringValue();
    float confidence = msg.get(1).floatValue();
    
    lastDetectedGesture = gesture;
    gestureConfidence = confidence;
    gestureDisplayTime = millis();
    println("Received gesture: " + gesture + " with confidence " + confidence);
  }
  else if (msg.addrPattern().equals("/control/volume_adjust")) {
    float amount = msg.get(0).floatValue();
    float newVolume = constrain(currentVolume + amount, 0, 1);
    
    if (newVolume != currentVolume) {
      currentVolume = newVolume;
      setTrackVolume();
      println("Volume adjusted by " + amount + " to " + currentVolume);
    }
  }
  else if (msg.addrPattern().equals("/control/track_change")) {
    int track = msg.get(0).intValue();
    
    if (track != currentTrack && track >= 0) {
      currentTrack = track;
      requestTrackVolume();
      println("Track changed to: " + currentTrack);
    }
  }
  else if (msg.addrPattern().equals("/training/sample")) {
    String gesture = msg.get(0).stringValue();
    float quality = msg.get(1).floatValue();
    
    lastSampleQuality = quality;
    samplesRecorded++;
    println("Recorded sample for " + gesture + " with quality " + quality);
  }
  else if (msg.addrPattern().equals("/training/status")) {
    String status = msg.get(0).stringValue();
    String message = msg.get(1).stringValue();
    
    trainingStatus = message;
    trainingStatusDisplayTime = millis();
    println("Training status: " + status + " - " + message);
    
    if (status.equals("success")) {
      // Exit training mode after successful save
      trainingMode = false;
      recordingGesture = false;
    }
  }
  else if (msg.addrPattern().equals("/live/track/get/volume")) {
    int track = msg.get(0).intValue();
    float volume = msg.get(1).floatValue();
    
    if (track == currentTrack) {
      currentVolume = volume;
      previousVolume = volume;
      println("← Received volume for track " + track + ": " + nf(volume, 0, 2));
    }
  }
  else if (msg.addrPattern().equals("/live/error")) {
    println("← Error: " + msg.get(0).stringValue());
  }
}

// --- Ableton OSC Communication ---
void setTrackVolume() {
  println("→ Setting track " + currentTrack + " volume to " + nf(currentVolume, 0, 2));
  
  OscMessage msg = new OscMessage("/live/track/set/volume");
  msg.add(currentTrack);
  msg.add(currentVolume);
  oscP5.send(msg, abletonAddress);
}

void requestTrackVolume() {
  println("→ Requesting track " + currentTrack + " volume");
  
  OscMessage msg = new OscMessage("/live/track/get/volume");
  msg.add(currentTrack);
  oscP5.send(msg, abletonAddress);
}