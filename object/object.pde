import processing.video.*;
import oscP5.*;
import netP5.*;

// --- Camera settings ---
Capture cam;
int camWidth = 800;
int camHeight = 540;

// --- OSC communication ---
OscP5 oscP5;
NetAddress abletonAddress;
NetAddress pythonAddress;
boolean useMLGestures = true;

// --- Multi-object tracking settings ---
int maxObjects = 4;  // Support up to 4 different colored objects
color[] targetColors = new color[maxObjects];
boolean[] colorsSelected = new boolean[maxObjects];
float[] colorSensitivities = new float[maxObjects];
PVector[] trackedPoints = new PVector[maxObjects];
boolean[] objectsDetected = new boolean[maxObjects];
ArrayList<PVector>[] gestureTrails = new ArrayList[maxObjects];
int maxTrailPoints = 30;
int currentSelectedObject = 0;  // Which object we're currently selecting/modifying

// --- Track settings ---
int currentTrack = 0;
float[] trackVolumes = new float[maxObjects];
float[] previousVolumes = new float[maxObjects];
// New pan control variables
float[] trackPans = new float[maxObjects];
float[] previousPans = new float[maxObjects];

// --- ML Gesture recognition variables ---
String lastDetectedGesture = "";
float gestureConfidence = 0.0;
int gestureDisplayTime = 0;

// --- Training mode variables ---
boolean trainingMode = false;
String[] availableGestures = {"volume_up", "volume_down", "next_track", "prev_track", "pan_left", "pan_right"};
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
  
  // Initialize multi-object tracking arrays
  for (int i = 0; i < maxObjects; i++) {
    targetColors[i] = color(0);
    colorsSelected[i] = false;
    colorSensitivities[i] = 35.0;
    trackedPoints[i] = new PVector(0, 0);
    objectsDetected[i] = false;
    gestureTrails[i] = new ArrayList<PVector>();
    trackVolumes[i] = 0.7;
    previousVolumes[i] = -1;
    // Initialize pan values
    trackPans[i] = 0.5; // Center pan (0.0 = left, 0.5 = center, 1.0 = right)
    previousPans[i] = -1;
  }
  
  // Initialize OSC communication
  oscP5 = new OscP5(this, 12001);  // Listen on port 12001 for messages from Python
  abletonAddress = new NetAddress("127.0.0.1", 11000);  // Send to Ableton on 11000
  pythonAddress = new NetAddress("127.0.0.1", 12000);   // Send to Python ML on 12000
  
  // Request initial volumes and pans for all tracks
  for (int i = 0; i < maxObjects; i++) {
    requestTrackVolume(i);
    requestTrackPan(i);
  }
  
  println("------------------------------------");
  println("MULTI-OBJECT ML GESTURE CONTROL FOR ABLETON LIVE");
  println("------------------------------------");
  println("Click on objects to track their colors (up to " + maxObjects + " objects)");
  println("Press 1-" + maxObjects + " to select which object to modify");
  println("Press 'm' to toggle between direct control and ML gesture recognition");
  println("Press 't' to enter training mode");
  println("LEFT/RIGHT arrows change Ableton track for current object");
  println("UP/DOWN arrows adjust color sensitivity for current object");
  println("Press 'r' to reset color selection for current object");
  println("Press 'c' to clear all color selections");
  println("In direct control mode: Y position = volume, X position = pan");
}

void draw() {
  // Update camera feed
  if (cam.available()) {
    cam.read();
  }
  
  // Display camera image
  image(cam, 0, 0);
  
  // Track all selected objects
  for (int i = 0; i < maxObjects; i++) {
    if (colorsSelected[i]) {
      trackObject(i);
      
      // Send tracking data to Python ML if enabled
      if (useMLGestures && objectsDetected[i]) {
        sendTrackingData(i);
      }
    }
  }
  
  // Draw UI elements
  drawUI();
  
  // Draw training mode UI if active
  if (trainingMode) {
    drawTrainingUI();
  }
}

void trackObject(int objectIndex) {
  cam.loadPixels();
  
  float sumX = 0;
  float sumY = 0;
  int count = 0;
  
  color targetColor = targetColors[objectIndex];
  float sensitivity = colorSensitivities[objectIndex];
  
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
      if (d < sensitivity) {
        sumX += x;
        sumY += y;
        count++;
      }
    }
  }
  
  // If enough matching pixels found, update tracked position
  if (count > 50) {
    objectsDetected[objectIndex] = true;
    
    // Calculate average position (centroid)
    trackedPoints[objectIndex].x = sumX / count;
    trackedPoints[objectIndex].y = sumY / count;
    
    // Add point to the gesture trail
    ArrayList<PVector> trail = gestureTrails[objectIndex];
    if (trail.size() >= maxTrailPoints) {
      trail.remove(0);
    }
    trail.add(new PVector(trackedPoints[objectIndex].x, trackedPoints[objectIndex].y));
    
    // If not using ML, directly map positions to parameters
    if (!useMLGestures) {
      // Map Y position to volume (top = max volume, bottom = min volume)
      float newVolume = map(trackedPoints[objectIndex].y, 0, camHeight, 1.0, 0.0);
      trackVolumes[objectIndex] = newVolume;
      
      // Map X position to pan (left = pan left, right = pan right)
      float newPan = map(trackedPoints[objectIndex].x, 0, camWidth, 0.0, 1.0);
      trackPans[objectIndex] = newPan;
      
      // Send volume to Ableton if it changed enough
      if (abs(trackVolumes[objectIndex] - previousVolumes[objectIndex]) > 0.01) {
        previousVolumes[objectIndex] = trackVolumes[objectIndex];
        setTrackVolume(objectIndex);
      }
      
      // Send pan to Ableton if it changed enough
      if (abs(trackPans[objectIndex] - previousPans[objectIndex]) > 0.01) {
        previousPans[objectIndex] = trackPans[objectIndex];
        setTrackPan(objectIndex);
      }
    }
    
    // Draw the gesture trail if in training mode or recently detected gesture
    if (trainingMode || millis() < gestureDisplayTime + 2000) {
      // Use different colors for different objects
      color trailColor = getObjectColor(objectIndex);
      stroke(red(trailColor), green(trailColor), blue(trailColor), trainingMode ? 200 : 150);
      strokeWeight(trainingMode ? 3 : 2);
      noFill();
      beginShape();
      for (PVector p : trail) {
        vertex(p.x, p.y);
      }
      endShape();
    }
    
    // Draw tracking indicator with object-specific color
    noFill();
    color indicatorColor = getObjectColor(objectIndex);
    stroke(red(indicatorColor), green(indicatorColor), blue(indicatorColor));
    strokeWeight(2);
    ellipse(trackedPoints[objectIndex].x, trackedPoints[objectIndex].y, 40, 40);
    
    // Draw cross at center
    line(trackedPoints[objectIndex].x - 10, trackedPoints[objectIndex].y, 
         trackedPoints[objectIndex].x + 10, trackedPoints[objectIndex].y);
    line(trackedPoints[objectIndex].x, trackedPoints[objectIndex].y - 10, 
         trackedPoints[objectIndex].x, trackedPoints[objectIndex].y + 10);
    
    // Draw object number
    fill(255);
    textAlign(CENTER, CENTER);
    text(str(objectIndex), trackedPoints[objectIndex].x, trackedPoints[objectIndex].y - 25);
    
    // Draw pan position indicator
    if (!useMLGestures) {
      // Add pan indicator line
      stroke(255, 255, 0); // Yellow line for pan
      float panLength = 30; // Length of pan indicator line
      float panAngle = map(trackPans[objectIndex], 0, 1, PI, 0); // Map pan value to angle
      float panX = trackedPoints[objectIndex].x + cos(panAngle) * panLength;
      float panY = trackedPoints[objectIndex].y;
      line(trackedPoints[objectIndex].x, trackedPoints[objectIndex].y, panX, panY);
    }
    
    textAlign(LEFT, BASELINE); // Reset text alignment
  } else {
    objectsDetected[objectIndex] = false;
  }
}

color getObjectColor(int objectIndex) {
  // Return distinct colors for each object
  switch(objectIndex) {
    case 0: return color(0, 255, 0);    // Green
    case 1: return color(255, 0, 0);    // Red
    case 2: return color(0, 0, 255);    // Blue
    case 3: return color(255, 255, 0);  // Yellow
    default: return color(255, 255, 255); // White
  }
}

void drawUI() {
  // Background for UI elements
  fill(0, 150);
  noStroke();
  rect(0, 0, width, 80);
  rect(0, height-60, width, 60);
  
  // Show selected colors for all objects
  fill(255);
  text("Objects:", 10, 20);
  
  for (int i = 0; i < maxObjects; i++) {
    int xPos = 70 + i * 80;
    
    // Highlight current selected object
    if (i == currentSelectedObject) {
      fill(255, 255, 0, 100);
      rect(xPos - 5, 5, 75, 35);
    }
    
    fill(255);
    text(str(i), xPos, 20);
    
    if (colorsSelected[i]) {
      fill(targetColors[i]);
      rect(xPos + 15, 8, 25, 15);
      
      // Show tracking status
      fill(objectsDetected[i] ? color(0, 255, 0) : color(255, 0, 0));
      ellipse(xPos + 50, 15, 8, 8);
    } else {
      fill(150);
      text("--", xPos + 15, 20);
    }
  }
  
  // Show current object info
  fill(255);
  text("Current Object: " + currentSelectedObject, 10, 40);
  if (colorsSelected[currentSelectedObject]) {
    text("Track: " + currentSelectedObject, 150, 40);
    text("Volume: " + nf(trackVolumes[currentSelectedObject], 0, 2), 220, 40);
    text("Pan: " + nf(trackPans[currentSelectedObject], 0, 2), 320, 40);
    text("Sensitivity: " + int(colorSensitivities[currentSelectedObject]), 420, 40);
  } else {
    text("No color selected", 150, 40);
  }
  
  // Show control mode
  fill(255);
  text("Mode: " + (useMLGestures ? "ML GESTURES" : "DIRECT CONTROL"), 10, 60);
  
  // Show detected gesture if available
  if (useMLGestures && !lastDetectedGesture.equals("") && millis() < gestureDisplayTime + 2000) {
    fill(255, 255, 0);
    text("Gesture: " + lastDetectedGesture + " (" + nf(gestureConfidence, 0, 2) + ")", 200, 60);
  }
  
  // Show controls reminder
  fill(255);
  String controls1 = "1-" + maxObjects + ": Select Object | LEFT/RIGHT: Track | UP/DOWN: Sensitivity | M: Toggle Mode";
  String controls2 = "T: Train | R: Reset Current | C: Clear All | Click: Select Color";
  text(controls1, 10, height-35);
  text(controls2, 10, height-15);
  
  // Draw volume bars for all objects
  for (int i = 0; i < maxObjects; i++) {
    int xPos = width - 150 + i * 35;
    
    noStroke();
    fill(50);
    rect(xPos, 70, 25, height-160);
    
    if (objectsDetected[i]) {
      color objColor = getObjectColor(i);
      fill(red(objColor), green(objColor), blue(objColor));
    } else if (colorsSelected[i]) {
      fill(255, 165, 0); // Orange
    } else {
      fill(150);
    }
    
    float barHeight = trackVolumes[i] * (height-160);
    rect(xPos, 70 + (height-160) - barHeight, 25, barHeight);
    
    // Pan indicator
    stroke(255, 255, 0);
    strokeWeight(2);
    float panPos = map(trackPans[i], 0, 1, xPos-5, xPos+30);
    line(panPos, 70 + (height-160) - barHeight - 10, panPos, 70 + (height-160) - barHeight - 5);
    noStroke();
    
    // Object number
    fill(255);
    textAlign(CENTER, CENTER);
    text(str(i), xPos + 12, height-140);
    textAlign(LEFT, BASELINE); // Reset
  }
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
  
  // Current gesture and object
  fill(255, 255, 0);
  text("Current gesture: " + availableGestures[currentGestureIndex], width/2, 100);
  text("Training with object: " + currentSelectedObject, width/2, 130);
  
  // Show which objects are available for training
  String availableObjects = "Available objects: ";
  for (int i = 0; i < maxObjects; i++) {
    if (colorsSelected[i]) {
      availableObjects += i + " ";
    }
  }
  fill(200);
  text(availableObjects, width/2, 160);
  
  fill(255, 255, 0);
  text("Samples recorded: " + samplesRecorded, width/2, 180);
  
  // Recording status
  if (recordingGesture) {
    fill(255, 0, 0);
    ellipse(width/2, 210, 20, 20);
    fill(255);
    text("RECORDING SAMPLES", width/2, 240);
    
    // Show quality of last sample
    if (lastSampleQuality > 0) {
      fill(lastSampleQuality > 0.3 ? color(0, 255, 0) : color(255, 165, 0));
      text("Movement quality: " + nf(lastSampleQuality, 0, 2), width/2, 270);
    }
    
    fill(255);
    text("Move object " + currentSelectedObject + " to demonstrate the gesture", width/2, 300);
  } else {
    fill(255);
    text("Press SPACE to start recording samples", width/2, 210);
  }
  
  // Display training status message if any
  if (!trainingStatus.equals("") && millis() < trainingStatusDisplayTime + 3000) {
    fill(255, 255, 0);
    text(trainingStatus, width/2, 330);
  }
  
  // Instructions
  fill(200);
  text("← → : Switch gesture type", width/2, height-150);
  text("1-" + maxObjects + " : Select training object", width/2, height-120);
  text("SPACE : Start/stop recording samples", width/2, height-90);
  text("S : Save model and exit training", width/2, height-60);
  text("ESC : Exit training mode", width/2, height-30);
  
  textAlign(LEFT, BASELINE); // Reset text alignment
  textSize(16); // Reset text size
}

void sendTrackingData(int objectIndex) {
  // Send tracked object position to Python ML
  OscMessage msg = new OscMessage("/tracking/position");
  msg.add(objectIndex); // Add object index
  msg.add((float) trackedPoints[objectIndex].x / camWidth); // Normalize x position
  msg.add((float) trackedPoints[objectIndex].y / camHeight); // Normalize y position
  msg.add(1); // Object is detected
  oscP5.send(msg, pythonAddress);
  
  // Also send current track for this object
  OscMessage trackMsg = new OscMessage("/tracking/track");
  trackMsg.add(objectIndex); // Object index
  trackMsg.add(objectIndex); // Track index (object index maps to track index)
  oscP5.send(trackMsg, pythonAddress);
}

void mousePressed() {
  // Ignore clicks when in training mode
  if (trainingMode) return;
  
  // Check if we're clicking in the camera image area (avoid UI elements)
  if (mouseY > 80 && mouseY < height - 60) {
    int loc = mouseX + mouseY * cam.width;
    
    if (loc >= 0 && loc < cam.pixels.length) {
      targetColors[currentSelectedObject] = cam.pixels[loc];
      colorsSelected[currentSelectedObject] = true;
      gestureTrails[currentSelectedObject].clear(); // Clear the gesture trail
      
      color selectedColor = targetColors[currentSelectedObject];
      println("Selected color for object " + currentSelectedObject + ": R=" + red(selectedColor) + 
              ", G=" + green(selectedColor) + ", B=" + blue(selectedColor));
    }
  }
}

void keyPressed() {
  // Handle training mode specially
  if (trainingMode) {
    handleTrainingModeKeys();
    return;
  }
  
  // Object selection (1-4 keys)
  if (key >= '1' && key <= '4') {
    int objectIndex = key - '1';
    if (objectIndex < maxObjects) {
      currentSelectedObject = objectIndex;
      println("Selected object " + currentSelectedObject);
    }
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
    for (int i = 0; i < maxObjects; i++) {
      gestureTrails[i].clear();
    }
    println("Entered training mode");
  }
  
  // Track controls for current selected object
  if (keyCode == LEFT) {
    // Previous track - in this case, just change the current selected object
    currentSelectedObject = (currentSelectedObject + maxObjects - 1) % maxObjects;
    println("Selected object " + currentSelectedObject);
  } else if (keyCode == RIGHT) {
    // Next track - in this case, just change the current selected object
    currentSelectedObject = (currentSelectedObject + 1) % maxObjects;
    println("Selected object " + currentSelectedObject);
  }
  
  // Sensitivity controls for current selected object
  if (keyCode == UP) {
    colorSensitivities[currentSelectedObject] += 5;
    if (colorSensitivities[currentSelectedObject] > 150) {
      colorSensitivities[currentSelectedObject] = 150;
    }
    println("Object " + currentSelectedObject + " sensitivity: " + colorSensitivities[currentSelectedObject]);
  } else if (keyCode == DOWN) {
    colorSensitivities[currentSelectedObject] -= 5;
    if (colorSensitivities[currentSelectedObject] < 5) {
      colorSensitivities[currentSelectedObject] = 5;
    }
    println("Object " + currentSelectedObject + " sensitivity: " + colorSensitivities[currentSelectedObject]);
  }
  
  // Reset color selection for current object
  if (key == 'r' || key == 'R') {
    colorsSelected[currentSelectedObject] = false;
    gestureTrails[currentSelectedObject].clear();
    println("Color selection reset for object " + currentSelectedObject);
  }
  
  // Clear all color selections
  if (key == 'c' || key == 'C') {
    for (int i = 0; i < maxObjects; i++) {
      colorsSelected[i] = false;
      gestureTrails[i].clear();
    }
    println("All color selections cleared");
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
  
  // Change which object to train with
  if (key >= '1' && key <= '4') {
    int objectIndex = key - '1';
    if (objectIndex < maxObjects && colorsSelected[objectIndex]) {
      currentSelectedObject = objectIndex;
      println("Training mode: selected object " + currentSelectedObject);
    }
  }
  
  // Start/stop recording samples for current gesture
  if (key == ' ') {
    recordingGesture = !recordingGesture;
    for (int i = 0; i < maxObjects; i++) {
      gestureTrails[i].clear(); // Clear all gesture trails
    }
    
    if (recordingGesture) {
      // Start recording
      OscMessage msg = new OscMessage("/training/start");
      msg.add(availableGestures[currentGestureIndex]);
      msg.add(currentSelectedObject); // Add which object we're training
      oscP5.send(msg, pythonAddress);
      println("Started recording samples for " + availableGestures[currentGestureIndex] + " with object " + currentSelectedObject);
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
    int objectIndex = msg.get(2).intValue(); // Get which object detected the gesture
    
    lastDetectedGesture = gesture + " (obj " + objectIndex + ")";
    gestureConfidence = confidence;
    gestureDisplayTime = millis();
    println("Received gesture: " + gesture + " from object " + objectIndex + " with confidence " + confidence);
  }
  else if (msg.addrPattern().equals("/control/volume_adjust")) {
    int objectIndex = msg.get(0).intValue();
    float amount = msg.get(1).floatValue();
    float newVolume = constrain(trackVolumes[objectIndex] + amount, 0, 1);
    
    if (newVolume != trackVolumes[objectIndex]) {
      trackVolumes[objectIndex] = newVolume;
      setTrackVolume(objectIndex);
      println("Volume adjusted for object " + objectIndex + " by " + amount + " to " + trackVolumes[objectIndex]);
    }
  }
  else if (msg.addrPattern().equals("/control/pan_adjust")) {
    int objectIndex = msg.get(0).intValue();
    float amount = msg.get(1).floatValue();
    float newPan = constrain(trackPans[objectIndex] + amount, 0, 1);
    
    if (newPan != trackPans[objectIndex]) {
      trackPans[objectIndex] = newPan;
      setTrackPan(objectIndex);
      println("Pan adjusted for object " + objectIndex + " by " + amount + " to " + trackPans[objectIndex]);
    }
  }
  else if (msg.addrPattern().equals("/control/track_change")) {
    int objectIndex = msg.get(0).intValue();
    int track = msg.get(1).intValue();
    
    if (track >= 0) {
      // In multi-object mode, each object controls its own track (track = objectIndex)
      requestTrackVolume(objectIndex);
      requestTrackPan(objectIndex);
      println("Track changed for object " + objectIndex + " to: " + track);
    }
  }
  else if (msg.addrPattern().equals("/training/sample")) {
    String gesture = msg.get(0).stringValue();
    float quality = msg.get(1).floatValue();
    int objectIndex = msg.get(2).intValue();
    
    lastSampleQuality = quality;
    samplesRecorded++;
    println("Recorded sample for " + gesture + " with object " + objectIndex + " with quality " + quality);
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
    
    if (track < maxObjects) {
      trackVolumes[track] = volume;
      previousVolumes[track] = volume;
      println("← Received volume for track " + track + ": " + nf(volume, 0, 2));
    }
  }
  else if (msg.addrPattern().equals("/live/track/get/pan")) {
    int track = msg.get(0).intValue();
    float pan = msg.get(1).floatValue();
    
    if (track < maxObjects) {
      trackPans[track] = pan;
      previousPans[track] = pan;
      println("← Received pan for track " + track + ": " + nf(pan, 0, 2));
    }
  }
  else if (msg.addrPattern().equals("/live/error")) {
    println("← Error: " + msg.get(0).stringValue());
  }
}

// --- Ableton OSC Communication ---
void setTrackVolume(int trackIndex) {
  println("→ Setting track " + trackIndex + " volume to " + nf(trackVolumes[trackIndex], 0, 2));
  
  OscMessage msg = new OscMessage("/live/track/set/volume");
  msg.add(trackIndex);
  msg.add(trackVolumes[trackIndex]);
  oscP5.send(msg, abletonAddress);
}

void requestTrackVolume(int trackIndex) {
  println("→ Requesting track " + trackIndex + " volume");
  
  OscMessage msg = new OscMessage("/live/track/get/volume");
  msg.add(trackIndex);
  oscP5.send(msg, abletonAddress);
}

void setTrackPan(int trackIndex) {
  println("→ Setting track " + trackIndex + " pan to " + nf(trackPans[trackIndex], 0, 2));
  
  // Map from 0-1 range to -1 to 1 range for Ableton
  float mappedPan = map(trackPans[trackIndex], 0.0, 1.0, -1.0, 1.0);
  
  // Send using standard track pan control with the correct address
  OscMessage msg = new OscMessage("/live/track/set/panning");
  msg.add(trackIndex);    // Track number
  msg.add(mappedPan);     // Pan value (-1.0 to 1.0)
  oscP5.send(msg, abletonAddress);
}

void requestTrackPan(int trackIndex) {
  println("→ Requesting track " + trackIndex + " pan");
  
  OscMessage msg = new OscMessage("/live/track/get/panning");
  msg.add(trackIndex);
  oscP5.send(msg, abletonAddress);
}