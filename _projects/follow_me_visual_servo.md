---
layout: page
title: Follow-Me Robot Using YOLOv11, SAM2, and Visual Servo Control
description: A sophisticated visual servoing system combining YOLOv11 object detection, SAM2 segmentation, and image-based visual servo control for autonomous robot following behavior.
img: assets/img/sam2.png
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
    <div class="col-sm-12 mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/project_sam2.png" title="Follow-Me System Architecture" class="img-fluid rounded z-depth-1" %}
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

The velocity of image features is related to the robot's velocity through the image Jacobian:

$$\dot{\mathbf{s}} = \mathbf{L}_s \mathbf{v}$$

where:
- **s** is the vector of image features
- **v** is the robot's velocity (linear and angular)
- **L_s** is the image Jacobian matrix

#### Control Law

The control law aims to minimize the error between current features and desired features:

$$\mathbf{e}(t) = \mathbf{s}(t) - \mathbf{s}^*$$

The desired velocity command is computed as:

$$\mathbf{v} = -\lambda \mathbf{L}_s^+ \mathbf{e}(t)$$

where λ is a gain parameter and L_s^+ is the pseudo-inverse of the image Jacobian.

#### Stability Analysis

The closed-loop system ensures exponential convergence when the image Jacobian has full rank:

$$\dot{\mathbf{e}} = -\lambda \mathbf{e}$$

This guarantees that the error decreases exponentially over time.

---

## Feature Extraction from SAM2 Segmentation

### Centroid-Based Features

After SAM2 segmentation, we extract the centroid of the segmented mask:

$$u_c = \frac{1}{N} \sum_{i=1}^{N} u_i, \quad v_c = \frac{1}{N} \sum_{i=1}^{N} v_i$$

where N is the number of pixels in the mask, and (u_i, v_i) are pixel coordinates.

### Area-Based Feature

The area of the segmented region provides depth information:

$$a = \frac{A}{A_0}$$

where A is the current area and A_0 is the desired area (maintaining constant distance).

---

## YOLOv11 Integration

YOLOv11 provides robust real-time object detection. The detection output includes:

- **Bounding box**: Center coordinates and dimensions
- **Confidence score**: Detection reliability
- **Class ID**: Target class identifier

The bounding box center is used to initialize SAM2 segmentation for precise target tracking.

---

## SAM2 Segmentation

SAM2 (Segment Anything Model 2) provides zero-shot segmentation capabilities. Given a prompt (bounding box from YOLOv11), SAM2 generates a precise segmentation mask.

This mask is used to:
1. Extract precise visual features (centroid, area)
2. Filter out background noise
3. Provide robust tracking even with partial occlusions

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
- **Response Time**: Less than 100ms latency from image capture to control command
- **Robustness**: Handles partial occlusions and lighting variations
- **Distance Control**: Maintains desired following distance with ±10% accuracy

### Test Scenarios

1. **Indoor Following**: Following person in office environment
2. **Outdoor Following**: Tracking in outdoor conditions with varying lighting
3. **Occlusion Handling**: Maintaining track during partial occlusions
4. **Multi-Target**: Selecting and following specific target among multiple people

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
