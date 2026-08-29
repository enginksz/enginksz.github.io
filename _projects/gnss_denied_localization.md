---
layout: page
title: GNSS-Denied Visual Localization for UAVs using Satellite Imagery
description: Deep learning-based visual localization system for autonomous UAV navigation in GPS-denied environments
img: assets/img/navwogps/eb94e55c-d80c-4118-bb8f-171e4417d1ef.png
importance: 1
category: work
giscus_comments: true
links:
  - title: Paper
    url: /assets/pdf/gnss_denied_localization.pdf
tags:
  - Computer Vision
  - Deep Learning
  - Robotics
  - SLAM
  - UAV Navigation
---
## Overview

**Paper accepted at [SIU 2026](https://siu2026.pirireis.edu.tr/)** (34th Signal Processing and Communications Applications Conference, 7–10 July 2026, Piri Reis University).

Visual Localization project is a real-time visual localization system designed for Unmanned Aerial Vehicles (UAVs) operating in GNSS-denied environments. The system leverages deep learning-based feature matching between onboard camera images and geo-referenced satellite imagery to provide accurate position estimates without relying on GPS signals.

The core innovation lies in dynamically generating perspective-transformed map regions from satellite tiles based on the UAV's estimated attitude and position, enabling robust visual feature matching even under challenging conditions such as illumination changes, viewpoint variations, and texture-poor environments.

## Problem Statement

Traditional GNSS-based navigation systems face critical limitations in:

- **Urban canyons**: High-rise buildings block satellite signals
- **Indoor environments**: No GPS coverage
- **Jamming scenarios**: Intentional interference disrupts navigation
- **Accuracy limitations**: Consumer-grade GPS provides 5-10 meter accuracy

Visual odometry and SLAM approaches offer alternatives but suffer from drift accumulation over time. Our system addresses these challenges by providing absolute position estimates through visual matching with geo-referenced satellite imagery.

## Technical Approach

### System Architecture

The system operates through five integrated modules:

1. **Dynamic Map Generation**: Satellite tiles are dynamically retrieved and assembled based on the UAV's estimated position and attitude
2. **Perspective Transformation**: Camera images are transformed to match the satellite map's coordinate system using homography
3. **Deep Feature Extraction**: State-of-the-art deep learning models extract robust visual features
4. **Feature Matching**: Learned matchers establish correspondences between camera and map features
5. **State Estimation**: Multiple position estimation methods with Kalman filtering provide robust pose estimates

### Deep Learning Pipeline

#### Feature Extraction

The system supports multiple deep learning-based feature extractors:

- **SuperPoint**: Fast, general-purpose feature detector with 256-dimensional descriptors
- **ALIKED**: Adaptive Local Invariant Keypoint Detection, optimized for low-contrast images
- **DISK**: Disk descriptor-based features
- **SIFT**: Classical SIFT algorithm (baseline comparison)
- **ORB:** Oriented FAST and Rotated BRIEF

Each extractor is optimized for real-time inference on embedded platforms, with TensorRT support for NVIDIA Jetson devices.

#### Feature Matching

**LightGlue** serves as the core matching engine, utilizing graph neural networks for robust feature correspondence:

- Bidirectional matching with outlier rejection
- Confidence-aware matching scores
- Real-time performance on embedded hardware
- Support for up to 1024 (default) and 2048 keypoints per image

The matching pipeline includes:

- Image pre-processing and contrast enhancement
- Homography-based geometric verification
- RANSAC-based outlier rejection

### Position Estimation Methods

The system implements three complementary position estimation approaches:

#### 1. ....

#### 2. ....

#### 3. Extended Kalman Filter (EKF)

Sophisticated filtering approach with camera parameters in state:

- **State vector**: `[x, vx, y, vy, z, vz, qw, qx, qy, qz, fx, fy]` (12-dimensional)
  - Position (x, y, z) and velocity (vx, vy, vz) in NED coordinates
  - Quaternion orientation (qw, qx, qy, qz)
  - Camera focal length parameters (fx, fy)
- **Minimum requirement**: ...
- **Output**: Filtered position estimate with uncertainty quantification

### Post-Processing Pipeline

**....**

### Coordinate Transformations

The system handles multiple coordinate systems:

- **Camera Frame**: Image pixel coordinates
- **NED Frame**: North-East-Down local navigation frame
- **WGS84**: Geodetic coordinates (latitude, longitude, altitude)
- **UTM**: Universal Transverse Mercator projection coordinates

Transformations include:

- Homography-based perspective correction
- Camera-to-IMU frame alignment
- Geodetic coordinate conversions
- Elevation integration from DEM (Digital Elevation Model) data

## Implementation Details

### Real-Time Processing

The system is optimized for real-time operation on embedded platforms:

- **Frame rate**: 10-30 Hz depending on hardware
- **Latency**: <100ms end-to-end processing time
- **Memory**: Efficient caching of satellite tiles
- **GPU acceleration**: TensorRT support for Jetson platforms

## Experimental Results

### Real-World Flight Tests

The system has been extensively tested in various conditions:

- **Daytime flights**: Clear visibility, good contrast
- **Night flights**: Low-light conditions with artificial illumination
- **Different altitudes**: 50m to 500m above ground level
- **Various terrains**: Urban, rural, and mixed environments

### Performance Metrics

**Position Accuracy**:

- Mean position error: **4.8 meters** (comparable to consumer GPS)
- RMS error: 6.2 meters
- 95th percentile error: 12.5 meters

**Robustness**:

- Matching success rate: >85% in favorable conditions
- Optic flow fallback activation: <5% of frames
- System uptime: >95% during test flights

**Computational Performance**:

- Processing time: 30-80ms per frame (NVIDIA Jetson AGX)
- Memory usage: <2GB RAM
- Power consumption: <15W on Jetson platforms

### Comparison with Baselines

| Method             | Mean Error (m) | RMS Error (m) | Drift          |
| ------------------ | -------------- | ------------- | -------------- |
| **NAVWOGPS** | **4.8**  | **6.2** | **None** |
| Visual Odometry    | 12.3           | 18.5          | High           |
| GPS (Consumer)     | 5.2            | 8.1           | Low            |
| GPS (RTK)          | 0.1            | 0.2           | None           |

## Key Features

### Advantages

1. **Absolute Positioning**: No drift accumulation unlike visual odometry
2. **GNSS Independence**: Operates without GPS signals
3. **Real-Time Performance**: Optimized for embedded platforms
4. **Robust Matching**: Deep learning features handle challenging conditions
5. **Modular Design**: Easy to integrate with existing navigation stacks

### Limitations

1. **Satellite Imagery Dependency**: Requires internet connectivity for tile retrieval
2. **Altitude Constraints**: Optimal performance between 50-1000m AGL (The zoom level was tested up to 11.~1000 meter)
3. **Texture Requirements**: Performance degrades in texture-poor environments
4. **Computational Resources**: Requires GPU for real-time operation

## Applications

The system is suitable for:

- **Search and Rescue**: Navigation in GPS-denied areas
- **Infrastructure Inspection**: Precise positioning for autonomous inspection
- **Military Operations**: Operations in GPS-jammed environments
- **Indoor-Outdoor Transitions**: Seamless navigation across environments
- **Urban Air Mobility**: Navigation in dense urban environments

## Future Work

Potential improvements and extensions:

1. **Offline Map Storage**: Pre-downloaded satellite tiles for offline operation
2. **Multi-Modal Fusion**: Integration with IMU, barometer, and other sensors
3. **Seasonal Adaptation**: Handling seasonal changes in satellite imagery
4. **Semantic Understanding**: Integration with semantic segmentation for improved matching
5. **Distributed Processing**: Multi-UAV collaborative localization

## Technologies Used

- **Deep Learning**: PyTorch, SuperPoint, ALIKED, LightGlue
- **Computer Vision**: OpenCV,  RANSAC
- **State Estimation**: Extended Kalman Filter, Kalman Filter variants
- **Geospatial**: PyMap3D, Bing Maps API, SRTM elevation data
- **Embedded Computing**: TensorRT, NVIDIA Jetson platforms
- **Robotics**: ROS/ROS2 integration, MAVLink protocol

## Visualizations

### Feature Matching Results

<div class="row justify-content-sm-center">
    <div class="col-sm-8 mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/navwogps/Screenshot from 2024-10-10 17-34-40.png" title="Feature Matching Results" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

*Deep learning-based feature matching between camera image and satellite map. Green lines indicate matched features with high confidence scores.*

### Position Estimation Results

<div class="row justify-content-sm-center">
    <div class="col-sm-8 mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/navwogps/Screenshot from 2024-10-10 17-45-19.png" title="Position Estimation Accuracy" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

*Comparison of estimated trajectory (blue) with ground truth GPS trajectory (red) during a test flight. Mean position error: 4.8 meters.*

### Estimation of UAV Position in NED frame

<div class="row justify-content-sm-center">
    <div class="col-sm-8 mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/navwogps/09_ekim_figure_1.png" title="Real-Time Visualization" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

*Real-time visualization showing UAV position on satellite map with matched features and confidence indicators.*

## Conclusion

Visual Localization demonstrates that deep learning-based visual localization can provide reliable, absolute position estimates for UAVs in GNSS-denied environments. By combining state-of-the-art feature matching with robust state estimation, the system achieves GPS-comparable accuracy while operating independently of satellite navigation systems.

The modular architecture and extensive configuration options make the system adaptable to various mission requirements and hardware platforms, from small consumer drones to larger autonomous systems.

## References

1. DeTone, D., Malisiewicz, T., & Rabinovich, A. (2018). SuperPoint: Self-Supervised Interest Point Detection and Description. CVPR.
2. Sarlin, P. E., et al. (2020). SuperGlue: Learning Feature Matching with Graph Neural Networks. CVPR.
3. Zhao, X., et al. (2023). ALIKED: A Lighter Keypoint and Descriptor Extraction Network via Deformable Transformation. ICCV.
4. Lindenberger, P., et al. (2023). LightGlue: Local Feature Matching at Light Speed. ICCV.

---

*This project was developed as part of research on GNSS-denied navigation for autonomous systems. For more details, please contact with me.*
