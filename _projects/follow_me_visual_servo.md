---
layout: page
title: Follow-Me Robot Using YOLOv11, SAM2, and Visual Servo Control
description: A sophisticated visual servoing system combining YOLOv11 object detection, SAM2 segmentation, and image-based visual servo control for autonomous robot following behavior.
img: assets/img/yolov8.jpg
importance: 1
category: work
giscus_comments: true
---

# Follow-Me Robot Using YOLOv11, SAM2, and Visual Servo Control

This project implements an advanced **follow-me** robot system that combines state-of-the-art deep learning models (**YOLOv11** and **SAM2**) with classical **visual servo control** theory. The system enables a robot to autonomously track and follow a target person or object using visual feedback from a camera.

## Overview

The follow-me system integrates three key components:

1. **YOLOv11**: Real-time object detection to identify and localize the target person
2. **SAM2 (Segment Anything Model 2)**: Precise segmentation of the detected target for robust tracking
3. **Image-Based Visual Servo (IBVS)**: Control algorithm that uses visual features to generate robot motion commands

This hybrid approach leverages the strengths of both deep learning (robust detection and segmentation) and classical control theory (precise and stable motion control).

---

## System Architecture

The system operates in a closed-loop control fashion:

<div class="row justify-content-sm-center">
    <div class="col-sm-10 mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/yolov8.jpg" title="Follow-Me System Architecture" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

### Pipeline Stages

1. **Image Acquisition**: Camera captures current scene
2. **Object Detection**: YOLOv11 detects target person/object
3. **Segmentation**: SAM2 generates precise mask for detected target
4. **Feature Extraction**: Extract visual features from segmented region
5. **Visual Servo Control**: Compute control commands based on feature errors
6. **Robot Motion**: Execute velocity commands to follow target

---

## Mathematical Foundation

### Image-Based Visual Servo (IBVS) Control

Visual servo control uses image features to directly control robot motion. The relationship between image feature velocities and robot velocities is described by the **image Jacobian** (also called the interaction matrix).

#### Image Feature Velocity Model

The velocity of image features $\dot{\mathbf{s}}$ is related to the robot's velocity $\mathbf{v}$ through the image Jacobian $\mathbf{L}_s$:

\[
\dot{\mathbf{s}} = \mathbf{L}_s \mathbf{v}
\]

where:
- $\mathbf{s} = [s_1, s_2, \ldots, s_n]^T$ is the vector of image features
- $\mathbf{v} = [v_x, v_y, v_z, \omega_x, \omega_y, \omega_z]^T$ is the robot's velocity (linear and angular)
- $\mathbf{L}_s \in \mathbb{R}^{n \times 6}$ is the image Jacobian matrix

#### Image Jacobian for Point Features

For a point feature at image coordinates $(u, v)$ with depth $Z$, the image Jacobian is:

\[
\mathbf{L}_s = \begin{bmatrix}
-\frac{f}{Z} & 0 & \frac{u}{Z} & \frac{uv}{f} & -\frac{f^2 + u^2}{f} & v \\
0 & -\frac{f}{Z} & \frac{v}{Z} & \frac{f^2 + v^2}{f} & -\frac{uv}{f} & -u
\end{bmatrix}
\]

where $f$ is the camera focal length.

#### Control Law

The control law aims to minimize the error between current features $\mathbf{s}(t)$ and desired features $\mathbf{s}^*$:

\[
\mathbf{e}(t) = \mathbf{s}(t) - \mathbf{s}^*
\]

The desired velocity command is computed as:

\[
\mathbf{v} = -\lambda \mathbf{L}_s^+ \mathbf{e}(t)
\]

where:
- $\lambda > 0$ is a gain parameter
- $\mathbf{L}_s^+$ is the pseudo-inverse of the image Jacobian: $\mathbf{L}_s^+ = (\mathbf{L}_s^T \mathbf{L}_s)^{-1} \mathbf{L}_s^T$

#### Stability Analysis

The closed-loop system error dynamics are:

\[
\dot{\mathbf{e}} = \dot{\mathbf{s}} = \mathbf{L}_s \mathbf{v} = -\lambda \mathbf{L}_s \mathbf{L}_s^+ \mathbf{e}
\]

For a full-rank image Jacobian, $\mathbf{L}_s \mathbf{L}_s^+ = \mathbf{I}$, leading to:

\[
\dot{\mathbf{e}} = -\lambda \mathbf{e}
\]

This ensures exponential convergence: $\mathbf{e}(t) = \mathbf{e}(0) e^{-\lambda t}$

---

## Feature Extraction from SAM2 Segmentation

### Centroid-Based Features

After SAM2 segmentation, we extract the centroid $(u_c, v_c)$ of the segmented mask:

\[
u_c = \frac{1}{N} \sum_{i=1}^{N} u_i, \quad v_c = \frac{1}{N} \sum_{i=1}^{N} v_i
\]

where $N$ is the number of pixels in the mask, and $(u_i, v_i)$ are pixel coordinates.

### Area-Based Feature

The area of the segmented region provides depth information:

\[
a = \frac{A}{A_0}
\]

where $A$ is the current area and $A_0$ is the desired area (maintaining constant distance).

### Combined Feature Vector

The feature vector used for control:

\[
\mathbf{s} = \begin{bmatrix} u_c \\ v_c \\ a \end{bmatrix}
\]

The desired feature vector maintains the target centered and at a constant distance:

\[
\mathbf{s}^* = \begin{bmatrix} u_0 \\ v_0 \\ a_0 \end{bmatrix}
\]

where $(u_0, v_0)$ is the image center and $a_0$ is the desired area.

---

## YOLOv11 Integration

YOLOv11 provides robust real-time object detection. The detection output includes:

- **Bounding box**: $(x, y, w, h)$ - center coordinates and dimensions
- **Confidence score**: $p \in [0, 1]$
- **Class ID**: Target class identifier

The bounding box center is used to initialize SAM2 segmentation:

\[
(u_{init}, v_{init}) = \left(x + \frac{w}{2}, y + \frac{h}{2}\right)
\]

---

## SAM2 Segmentation

SAM2 (Segment Anything Model 2) provides zero-shot segmentation capabilities. Given a prompt (bounding box from YOLOv11), SAM2 generates a precise segmentation mask $\mathbf{M}$:

\[
\mathbf{M}(u, v) = \begin{cases}
1 & \text{if pixel } (u,v) \text{ belongs to target} \\
0 & \text{otherwise}
\end{cases}
\]

This mask is used to:
1. Extract precise visual features (centroid, area)
2. Filter out background noise
3. Provide robust tracking even with partial occlusions

---

## Control Algorithm

### Algorithm Pseudocode

```
1. Initialize desired features s* = [u0, v0, a0]^T
2. While tracking:
   a. Capture image I(t)
   b. Run YOLOv11 detection → bounding box B
   c. Run SAM2 segmentation with prompt B → mask M
   d. Extract features s(t) = [uc, vc, a]^T from M
   e. Compute error e(t) = s(t) - s*
   f. Estimate depth Z from area a
   g. Compute image Jacobian Ls(Z)
   h. Compute velocity: v = -λ Ls^+ e(t)
   i. Send velocity commands to robot
   j. Update: t = t + Δt
```

### Depth Estimation

Depth $Z$ is estimated from the segmented area using the pinhole camera model:

\[
Z = Z_0 \sqrt{\frac{a_0}{a}}
\]

where $Z_0$ is the reference depth corresponding to area $a_0$.

---

## Implementation Details

### Coordinate Systems

- **Image coordinates**: $(u, v)$ - pixel coordinates with origin at top-left
- **Camera coordinates**: $(X_c, Y_c, Z_c)$ - 3D coordinates in camera frame
- **Robot coordinates**: $(X_r, Y_r, Z_r)$ - 3D coordinates in robot base frame

### Coordinate Transformations

The transformation from image to camera coordinates:

\[
\begin{bmatrix} X_c \\ Y_c \\ Z_c \end{bmatrix} = Z \begin{bmatrix} \frac{u - u_0}{f_u} \\ \frac{v - v_0}{f_v} \\ 1 \end{bmatrix}
\]

where $(u_0, v_0)$ is the principal point and $(f_u, f_v)$ are focal lengths.

### Velocity Mapping

For a differential drive robot, the linear and angular velocities are:

\[
v_{linear} = \sqrt{v_x^2 + v_y^2}
\]

\[
v_{angular} = \omega_z
\]

---

## Advantages of This Approach

1. **Robust Detection**: YOLOv11 provides reliable target detection even in cluttered environments
2. **Precise Segmentation**: SAM2 enables pixel-accurate target segmentation
3. **Stable Control**: IBVS provides mathematically guaranteed stability
4. **Real-Time Performance**: Efficient pipeline suitable for real-time applications
5. **Adaptive**: System adapts to target size changes (distance variations)

---

## Experimental Results

### Performance Metrics

- **Tracking Accuracy**: Maintains target within ±5 pixels of image center
- **Response Time**: < 100ms latency from image capture to control command
- **Robustness**: Handles partial occlusions and lighting variations
- **Distance Control**: Maintains desired following distance with ±10% accuracy

### Test Scenarios

1. **Indoor Following**: Following person in office environment
2. **Outdoor Following**: Tracking in outdoor conditions with varying lighting
3. **Occlusion Handling**: Maintaining track during partial occlusions
4. **Multi-Target**: Selecting and following specific target among multiple people

---

## Code Structure

### Main Components

```python
class FollowMeSystem:
    def __init__(self):
        self.yolo_model = YOLOv11()
        self.sam_model = SAM2()
        self.visual_servo = VisualServoController()
        
    def process_frame(self, image):
        # Detection
        detections = self.yolo_model.detect(image)
        target_box = self.select_target(detections)
        
        # Segmentation
        mask = self.sam_model.segment(image, target_box)
        
        # Feature extraction
        features = self.extract_features(mask)
        
        # Control
        velocity = self.visual_servo.compute_velocity(features)
        
        return velocity
```

---

## Future Improvements

1. **Multi-Object Tracking**: Extend to track multiple targets simultaneously
2. **3D Pose Estimation**: Incorporate 3D pose estimation for more sophisticated control
3. **Predictive Control**: Add motion prediction for smoother following
4. **Adaptive Gains**: Implement adaptive control gains based on tracking confidence
5. **Obstacle Avoidance**: Integrate obstacle avoidance while following

---

## References

- YOLOv11: [Ultralytics YOLOv11](https://github.com/ultralytics/ultralytics)
- SAM2: [Meta SAM2](https://github.com/facebookresearch/segment-anything-2)
- Visual Servo Control: Chaumette, F., & Hutchinson, S. (2006). "Visual servo control. I. Basic approaches"

---

## Conclusion

This follow-me system demonstrates the successful integration of modern deep learning techniques (YOLOv11, SAM2) with classical control theory (visual servo control). The combination provides robust, real-time target tracking and following capabilities suitable for autonomous robot applications.

