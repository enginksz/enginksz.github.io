---
layout: page
title: Crazyflie Flight Tests and Experiments
description: Autonomous flight tests and experiments with Crazyflie 2.1 nano quadcopter, including control system development and real-world flight demonstrations
img: assets/img/video/IMG_0425.MOV
importance: 2
category: work
giscus_comments: true
tags:
  - Robotics
  - UAV
  - Autonomous Systems
  - Control Systems
  - Embedded Systems
---

## Overview

This project documents my work with the **Crazyflie 2.1** nano quadcopter, focusing on autonomous flight control, system integration, and real-world flight testing. The Crazyflie is a small, open-source quadcopter platform designed for research and development in autonomous systems, making it an ideal platform for testing control algorithms and navigation systems.

The work was conducted in early 2022, during which I performed extensive flight tests, control system tuning, and autonomous navigation experiments. A highlight of this project was the implementation and successful execution of a **complex rotating star pattern maneuver**, which demonstrates advanced trajectory following capabilities and precise control system coordination. This maneuver required sophisticated trajectory generation algorithms, real-time coordinate transformations, and careful tuning of cascaded PID controllers.

This project served as a foundation for understanding UAV dynamics, sensor fusion, real-time control systems, and complex trajectory planning algorithms.

## Crazyflie 2.1 Platform

The Crazyflie 2.1 is a palm-sized quadcopter featuring:

- **Dimensions**: 92mm × 92mm × 29mm
- **Weight**: 27g (without battery)
- **Flight time**: 7 minutes (with standard battery)
- **Onboard sensors**: 
  - MPU-9250 IMU (gyroscope, accelerometer, magnetometer)
  - BMP280 barometer
  - VL53L1x time-of-flight distance sensor
- **Communication**: 
  - Radio (2.4GHz) for control and telemetry
  - USB for direct connection
  - Bluetooth Low Energy (BLE)
- **Processing**: STM32F405 microcontroller (168 MHz ARM Cortex-M4)

### Key Features

1. **Open-Source**: Full hardware and software stack is open-source
2. **Modular Design**: Supports expansion decks for additional sensors
3. **ROS Integration**: Native ROS/ROS2 support via `crazyflie_ros` package
4. **Python API**: High-level Python API for rapid development
5. **Real-Time Control**: Low-level control loop running at 1kHz

## Project Objectives

The main objectives of this project were:

1. **System Integration**: Set up and configure the Crazyflie platform with necessary software stack
2. **Control System Development**: Implement and tune PID controllers for stable flight
3. **Autonomous Navigation**: Develop basic autonomous flight capabilities
4. **Sensor Fusion**: Integrate IMU, barometer, and other sensors for state estimation
5. **Real-World Testing**: Conduct flight tests in various conditions to validate system performance

## Technical Approach

### Control System Architecture

The Crazyflie uses a cascaded control architecture with three hierarchical levels:

1. **Rate Control** (Inner Loop): Direct motor speed control at 1kHz
2. **Attitude Control** (Middle Loop): Roll, pitch, and yaw angle control at 250Hz
3. **Position Control** (Outer Loop): X, Y, Z position control at 100Hz

This cascaded structure provides stability and allows for independent tuning of each control layer.

### PID Control Implementation

#### Position Control Loop

The position controller uses PID control for each axis (X, Y, Z):

$$u_p(t) = K_{p,p} e_p(t) + K_{i,p} \int_0^t e_p(\tau) d\tau + K_{d,p} \frac{de_p(t)}{dt}$$

where:
- $e_p(t) = p_{desired}(t) - p_{current}(t)$ is the position error
- $K_{p,p}$, $K_{i,p}$, $K_{d,p}$ are proportional, integral, and derivative gains for position
- $u_p(t)$ is the desired velocity command

#### Attitude Control Loop

The attitude controller regulates roll ($\phi$), pitch ($\theta$), and yaw ($\psi$) angles:

$$u_a(t) = K_{p,a} e_a(t) + K_{i,a} \int_0^t e_a(\tau) d\tau + K_{d,a} \frac{de_a(t)}{dt}$$

where:
- $e_a(t) = a_{desired}(t) - a_{current}(t)$ is the attitude error
- $K_{p,a}$, $K_{i,a}$, $K_{d,a}$ are attitude control gains
- $u_a(t)$ is the angular rate command

#### Rate Control Loop

The rate controller directly controls angular velocities:

$$u_r(t) = K_{p,r} e_r(t) + K_{i,r} \int_0^t e_r(\tau) d\tau + K_{d,r} \frac{de_r(t)}{dt}$$

where:
- $e_r(t) = \omega_{desired}(t) - \omega_{current}(t)$ is the angular rate error
- $K_{p,r}$, $K_{i,r}$, $K_{d,r}$ are rate control gains
- $u_r(t)$ is the motor command

### Quadrotor Dynamics Model

The quadrotor dynamics can be described using Newton-Euler equations:

#### Translational Dynamics

$$\begin{bmatrix} \ddot{x} \\ \ddot{y} \\ \ddot{z} \end{bmatrix} = \begin{bmatrix} 0 \\ 0 \\ -g \end{bmatrix} + \frac{T}{m} R(\phi, \theta, \psi) \begin{bmatrix} 0 \\ 0 \\ 1 \end{bmatrix}$$

where:
- $(x, y, z)$ is the position in inertial frame
- $g$ is gravitational acceleration
- $T$ is total thrust
- $m$ is quadrotor mass
- $R(\phi, \theta, \psi)$ is the rotation matrix from body to inertial frame

#### Rotational Dynamics

$$\mathbf{I} \dot{\boldsymbol{\omega}} = \boldsymbol{\tau} - \boldsymbol{\omega} \times \mathbf{I} \boldsymbol{\omega}$$

where:
- $\mathbf{I}$ is the inertia tensor
- $\boldsymbol{\omega} = [\omega_x, \omega_y, \omega_z]^T$ is angular velocity in body frame
- $\boldsymbol{\tau} = [\tau_x, \tau_y, \tau_z]^T$ is control torque

### Star Pattern Trajectory Generation

One of the most challenging maneuvers implemented was a **star pattern** trajectory with rotation**. This requires precise coordination of position and yaw control.

#### Star Pattern Geometry

A 5-pointed star pattern is generated using parametric equations:

$$x(t) = R \cos\left(\frac{2\pi k}{5}\right) \cos(\omega_r t) - R \sin\left(\frac{2\pi k}{5}\right) \sin(\omega_r t)$$

$$y(t) = R \cos\left(\frac{2\pi k}{5}\right) \sin(\omega_r t) + R \sin\left(\frac{2\pi k}{5}\right) \cos(\omega_r t)$$

where:
- $R$ is the star radius
- $k \in \{0, 1, 2, 3, 4\}$ selects the star point
- $\omega_r$ is the rotation rate around the center
- $t$ is time

#### Simultaneous Rotation Maneuver

The critical challenge is maintaining the star pattern while simultaneously rotating the entire pattern around its center. This requires:

1. **Coordinate Transformation**: Converting star pattern coordinates to rotating frame
2. **Yaw Synchronization**: Coordinating yaw angle with pattern rotation
3. **Velocity Feedforward**: Pre-computing required velocities for smooth trajectory following

The rotating star pattern coordinates are:

$$\begin{bmatrix} x_r(t) \\ y_r(t) \end{bmatrix} = \begin{bmatrix} \cos(\omega_r t) & -\sin(\omega_r t) \\ \sin(\omega_r t) & \cos(\omega_r t) \end{bmatrix} \begin{bmatrix} x_s(t) \\ y_s(t) \end{bmatrix}$$

where $(x_s, y_s)$ are the static star coordinates and $(x_r, y_r)$ are the rotated coordinates.

#### Trajectory Following Algorithm

The trajectory following algorithm uses a combination of feedforward and feedback control:

**Feedforward Component**:
$$\mathbf{v}_{ff}(t) = \frac{d\mathbf{p}_{desired}(t)}{dt}$$

**Feedback Component**:
$$\mathbf{v}_{fb}(t) = K_p (\mathbf{p}_{desired}(t) - \mathbf{p}_{current}(t)) + K_d (\mathbf{v}_{desired}(t) - \mathbf{v}_{current}(t))$$

**Total Velocity Command**:
$$\mathbf{v}_{cmd}(t) = \mathbf{v}_{ff}(t) + \mathbf{v}_{fb}(t)$$

This approach ensures both accurate trajectory tracking and disturbance rejection.

### Waypoint Navigation Algorithm

For autonomous navigation, a waypoint tracking algorithm was implemented:

1. **Path Planning**: Generate smooth trajectory between waypoints using Bézier curves or splines
2. **Lookahead Distance**: Calculate desired position based on current speed and lookahead distance
3. **Velocity Profiling**: Generate velocity profile considering acceleration limits
4. **Yaw Control**: Maintain desired heading or align with velocity vector

The lookahead-based waypoint tracking:

$$d_{lookahead} = k_v v_{current} + d_{min}$$

where $k_v$ is velocity gain and $d_{min}$ is minimum lookahead distance.

### State Estimation

The onboard state estimation combines multiple sensor sources using complementary filtering:

#### Attitude Estimation

The attitude is estimated using a complementary filter combining gyroscope and accelerometer:

$$\hat{\boldsymbol{\theta}}(t) = \alpha (\hat{\boldsymbol{\theta}}(t-1) + \boldsymbol{\omega}_{gyro} \Delta t) + (1-\alpha) \boldsymbol{\theta}_{accel}$$

where:
- $\alpha$ is the filter coefficient (typically 0.98-0.99)
- $\boldsymbol{\omega}_{gyro}$ is gyroscope angular rate
- $\boldsymbol{\theta}_{accel}$ is attitude from accelerometer
- $\Delta t$ is sampling period

#### Position Estimation

Position estimation combines:
- **IMU**: High-frequency attitude estimation (gyroscope + accelerometer) at 1kHz
- **Barometer**: Altitude estimation with low-pass filtering
- **Optical Flow** (optional): Horizontal velocity estimation
- **External Positioning** (optional): Motion capture system for ground truth

The altitude estimation:

$$z_{filtered}(t) = \beta z_{baro}(t) + (1-\beta) z_{filtered}(t-1)$$

where $\beta$ is the barometer filter coefficient.

### Visual Localization with LightGlue

For advanced positioning capabilities, the project integrated **LightGlue**, a state-of-the-art deep learning-based feature matching system for visual localization. LightGlue enables precise position estimation by matching visual features between camera images and reference maps.

#### LightGlue Architecture

LightGlue is a graph neural network-based feature matcher that provides robust correspondence between image features. The system architecture consists of:

1. **Feature Extraction**: Deep learning models (SuperPoint, ALIKED, DISK) extract keypoints and descriptors
2. **Graph Construction**: Features are represented as nodes in a graph
3. **Attention-Based Matching**: Graph neural networks establish correspondences
4. **Confidence Scoring**: Each match is assigned a confidence score
5. **Outlier Rejection**: Geometric verification filters incorrect matches

#### Mathematical Foundation

LightGlue uses attention mechanisms to establish feature correspondences. Given two sets of features $\mathbf{F}_A$ and $\mathbf{F}_B$ from images $A$ and $B$:

**Self-Attention**:
$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}}\right)\mathbf{V}$$

where $\mathbf{Q}$, $\mathbf{K}$, $\mathbf{V}$ are query, key, and value matrices derived from feature descriptors.

**Cross-Attention**:
The cross-attention mechanism establishes correspondences between features from different images:

$$\mathbf{M}_{ij} = \text{softmax}\left(\frac{\mathbf{f}_i^A \cdot \mathbf{f}_j^B}{\sqrt{d}}\right)$$

where $\mathbf{f}_i^A$ and $\mathbf{f}_j^B$ are feature descriptors, and $\mathbf{M}_{ij}$ represents the matching confidence.

**Bidirectional Matching**:
LightGlue performs bidirectional matching to ensure consistency:

$$m_{ij} = \arg\max_k \mathbf{M}_{ik} \quad \text{and} \quad m_{ji} = \arg\max_k \mathbf{M}_{kj}$$

A match is accepted only if $m_{ij} = j$ and $m_{ji} = i$ (mutual nearest neighbors).

#### Position Estimation from Matches

Given matched feature correspondences, the position is estimated using:

1. **Homography Estimation**: Compute homography matrix $\mathbf{H}$ using RANSAC:

$$\mathbf{H} = \arg\min_{\mathbf{H}} \sum_i \rho(\|\mathbf{x}_i' - \mathbf{H}\mathbf{x}_i\|^2)$$

where $\mathbf{x}_i$ and $\mathbf{x}_i'$ are matched feature points, and $\rho$ is a robust loss function.

2. **Perspective-n-Point (PnP)**: Solve for camera pose using matched 3D-2D correspondences:

$$\min_{\mathbf{R}, \mathbf{t}} \sum_i \|\mathbf{x}_i - \pi(\mathbf{R}\mathbf{X}_i + \mathbf{t})\|^2$$

where $\mathbf{R}$ and $\mathbf{t}$ are rotation and translation, $\mathbf{X}_i$ are 3D points, and $\pi$ is the projection function.

3. **Bundle Adjustment**: Refine position estimate by minimizing reprojection error:

$$E = \sum_{i,j} \|\mathbf{x}_{ij} - \pi(\mathbf{R}_j\mathbf{X}_i + \mathbf{t}_j)\|^2$$

#### LightGlue Performance Characteristics

- **Matching Speed**: 10-30ms per image pair on embedded hardware (NVIDIA Jetson)
- **Accuracy**: >95% matching precision on challenging image pairs
- **Robustness**: Handles illumination changes, viewpoint variations, and partial occlusions
- **Scalability**: Supports up to 2048 keypoints per image
- **Real-Time Capability**: Optimized for real-time operation at 10-30 Hz

#### Integration with Crazyflie

The LightGlue system was integrated with Crazyflie for visual localization:

1. **Onboard Camera**: Downward-facing camera captures ground images
2. **Feature Extraction**: Real-time keypoint detection and description
3. **Map Matching**: Match features against pre-loaded reference maps
4. **Position Update**: Fuse visual position estimates with IMU/barometer data
5. **Control Integration**: Use position estimates for precise trajectory following

This integration enables the Crazyflie to maintain accurate position estimates even in GPS-denied environments, significantly improving the accuracy of complex maneuvers like the star pattern trajectory.

### Coordinate Transformations

The system requires transformations between multiple coordinate frames:

#### Body to Inertial Frame Rotation

The rotation matrix from body frame to inertial frame (NED):

$$R_{BI} = \begin{bmatrix}
c\theta c\psi & s\phi s\theta c\psi - c\phi s\psi & c\phi s\theta c\psi + s\phi s\psi \\
c\theta s\psi & s\phi s\theta s\psi + c\phi c\psi & c\phi s\theta s\psi - s\phi c\psi \\
-s\theta & s\phi c\theta & c\phi c\theta
\end{bmatrix}$$

where $c$ and $s$ denote cosine and sine functions.

#### Velocity Transformation

Body frame velocities are transformed to inertial frame:

$$\begin{bmatrix} \dot{x} \\ \dot{y} \\ \dot{z} \end{bmatrix} = R_{BI} \begin{bmatrix} u \\ v \\ w \end{bmatrix}$$

where $(u, v, w)$ are body frame velocities.

### Development Tools

- **Crazyflie Python API**: For high-level control and telemetry
- **ROS/ROS2**: For integration with robotics software stack
- **CFclient**: Official Crazyflie client for configuration and monitoring
- **Python**: Custom scripts for flight automation, trajectory generation, and data logging
- **NumPy/SciPy**: For trajectory planning and control algorithm implementation

## Flight Test Results

### Test Scenarios

During the project, I conducted various flight tests:

1. **Manual Control Tests**: Basic flight stability and responsiveness
2. **Hover Tests**: Maintaining stable position in hover mode
3. **Waypoint Navigation**: Autonomous flight to predefined waypoints
4. **Star Pattern Maneuver**: Complex 5-pointed star pattern with simultaneous rotation (demonstrated in video)
5. **Indoor vs Outdoor**: Performance comparison in different environments

### Star Pattern Maneuver

The most challenging test was the **rotating star pattern** maneuver, which demonstrates advanced trajectory following capabilities. This maneuver requires:

- **Precise Position Control**: Following a complex 5-pointed star trajectory
- **Simultaneous Rotation**: Rotating the entire pattern around its center while maintaining the star shape
- **Yaw Synchronization**: Coordinating yaw angle with the rotation
- **Real-Time Computation**: Generating trajectory points in real-time at 100Hz

**Technical Challenges Overcome**:

1. **Coordinate Frame Management**: Handling multiple coordinate transformations simultaneously
2. **Velocity Feedforward**: Pre-computing required velocities for smooth trajectory following
3. **Control Loop Coordination**: Synchronizing position and attitude control loops
4. **Computational Constraints**: Running trajectory generation on limited embedded hardware

The maneuver successfully demonstrates the robustness of the control system and trajectory generation algorithms under demanding flight conditions.

### Performance Metrics

- **Stability**: Maintained stable hover with ±5cm position accuracy
- **Trajectory Tracking**: Star pattern tracking error <10cm RMS
- **Response Time**: Control system response time <50ms
- **Battery Life**: Achieved 6-7 minutes of flight time per battery
- **Range**: Radio communication range up to 100m (line of sight)
- **Rotation Accuracy**: Maintained star pattern shape during rotation with <5° angular deviation

## Challenges and Solutions

### Challenge 1: Sensor Noise and Drift

**Problem**: IMU sensors exhibit noise and drift, causing position estimation errors over time.

**Solution**: 
- Implemented complementary filter for attitude estimation
- Used barometer for altitude correction
- Integrated external positioning system (motion capture) for ground truth

### Challenge 2: Control System Tuning

**Problem**: PID controller parameters need careful tuning for stable flight.

**Solution**:
- Systematic parameter tuning using Ziegler-Nichols method
- Flight test iterations with incremental parameter adjustments
- Logged telemetry data for offline analysis

### Challenge 3: Limited Payload Capacity

**Problem**: Crazyflie's small size limits additional sensor payload.

**Solution**:
- Optimized sensor selection for weight
- Used lightweight expansion decks
- Prioritized essential sensors for mission requirements

### Challenge 4: Complex Trajectory Following (Star Pattern with Rotation)

**Problem**: The rotating star pattern maneuver requires precise coordination of multiple control loops while maintaining stability. The simultaneous rotation adds significant complexity to trajectory generation and control.

**Technical Challenges**:
- Real-time coordinate transformations at 100Hz
- Synchronization of position and yaw control loops
- Velocity feedforward computation for smooth trajectory following
- Handling of non-linear dynamics during rapid direction changes

**Solution**:
- Implemented efficient trajectory generation algorithm using pre-computed waypoints
- Used feedforward control to anticipate required velocities
- Tuned PID gains specifically for aggressive maneuvers
- Implemented adaptive lookahead distance based on trajectory curvature
- Added trajectory smoothing using Bézier curve interpolation between waypoints

**Algorithm Details**:

The trajectory generation uses a two-stage approach:

1. **Offline Planning**: Pre-compute star pattern waypoints with rotation
2. **Online Execution**: Real-time interpolation and feedforward velocity computation

The feedforward velocity is computed as:

$$\mathbf{v}_{ff} = \frac{\mathbf{p}_{next} - \mathbf{p}_{current}}{\Delta t} + \mathbf{a}_{desired} \cdot \Delta t$$

where $\mathbf{a}_{desired}$ accounts for centripetal acceleration during rotation.

## Applications and Use Cases

The Crazyflie platform is suitable for:

- **Research**: Testing control algorithms and navigation systems
- **Education**: Teaching robotics and autonomous systems concepts
- **Indoor Navigation**: GPS-denied navigation in indoor environments
- **Swarm Robotics**: Multi-robot coordination and formation flying
- **Sensor Development**: Testing new sensor technologies in flight

## LightGlue Visual Localization System

### Feature Matching Visualization

The following visualization demonstrates LightGlue's feature matching capabilities used for visual localization:

<div class="row justify-content-sm-center">
    <div class="col-sm-8 mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/navwogps/Screenshot from 2024-10-10 17-34-40.png" title="LightGlue Feature Matching" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

*LightGlue feature matching results showing correspondences between camera image and reference map. Green lines indicate matched features with high confidence scores. The bidirectional matching ensures robust correspondences even under challenging conditions.*

### Position Estimation Accuracy

The following visualization shows the position estimation accuracy achieved using LightGlue-based visual localization:

<div class="row justify-content-sm-center">
    <div class="col-sm-8 mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/navwogps/Screenshot from 2024-10-10 17-45-19.png" title="LightGlue Position Estimation" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

*Comparison of estimated trajectory (blue) with ground truth GPS trajectory (red) during a test flight. The LightGlue-based visual localization system achieves mean position error of 4.8 meters, comparable to consumer-grade GPS.*

### Real-Time Localization Visualization

The following figure demonstrates real-time position estimation using LightGlue:

<div class="row justify-content-sm-center">
    <div class="col-sm-8 mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/navwogps/09_ekim_figure_1.png" title="Real-Time LightGlue Localization" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

*Real-time visualization showing UAV position on reference map with matched features and confidence indicators. The LightGlue system provides robust position estimates for precise trajectory following.*

## Flight Test Video

The following video demonstrates the **rotating star pattern maneuver**, one of the most challenging flight tests conducted during this project:

<div class="row justify-content-sm-center">
    <div class="col-sm-8 mt-3 mt-md-0">
        {% include video.liquid path="assets/img/video/IMG_0425.MOV" title="Crazyflie Star Pattern Maneuver" class="img-fluid rounded z-depth-1" controls=true %}
    </div>
</div>

*This video shows the Crazyflie executing a complex 5-pointed star pattern trajectory while simultaneously rotating around its center. This maneuver demonstrates advanced autonomous flight capabilities, precise trajectory following, and robust control system performance. The drone maintains the star shape throughout the rotation, requiring precise coordination of position and yaw control loops, real-time trajectory generation, and sophisticated feedforward control algorithms.*

## Key Learnings

This project provided valuable insights into:

1. **UAV Dynamics**: Understanding quadcopter flight dynamics, non-linear control, and multi-rotor aerodynamics
2. **Real-Time Systems**: Working with embedded systems, real-time constraints, and control loop timing
3. **Sensor Fusion**: Combining multiple sensors (IMU, barometer) for robust state estimation using complementary filters
4. **Control Theory**: Practical application of PID control, cascaded control loops, and feedforward control
5. **Trajectory Planning**: Generating complex trajectories (star patterns) with rotation and real-time execution
6. **Coordinate Transformations**: Managing multiple coordinate frames (body, inertial, NED) for accurate control
7. **Visual Localization**: Deep learning-based feature matching using LightGlue for precise position estimation
8. **Graph Neural Networks**: Understanding attention mechanisms and graph-based feature matching
9. **System Integration**: Integrating hardware, software, control algorithms, trajectory generation, and visual localization
10. **Control Tuning**: Systematic PID parameter tuning for different flight regimes (hover, aggressive maneuvers)

## Future Work

Potential extensions and improvements:

1. **Advanced Control**: Implement model predictive control (MPC) for improved performance
2. **Enhanced Visual Localization**: Further optimize LightGlue integration for real-time performance on embedded platforms
3. **SLAM Integration**: Integrate simultaneous localization and mapping with LightGlue-based loop closure
4. **Multi-Modal Fusion**: Combine LightGlue visual localization with IMU/barometer for improved accuracy
5. **Swarm Coordination**: Develop multi-robot coordination algorithms with shared visual maps
6. **Machine Learning**: Apply reinforcement learning for adaptive control and trajectory optimization
7. **Onboard Processing**: Optimize LightGlue for real-time execution directly on Crazyflie's limited computational resources

## Technologies Used

- **Hardware**: Crazyflie 2.1 nano quadcopter
- **Software**: 
  - Crazyflie Python API
  - ROS/ROS2
  - Python (NumPy, SciPy for trajectory planning)
  - C (firmware development)
- **Control Systems**: 
  - PID controllers (position, attitude, rate)
  - Cascaded control architecture
  - Feedforward control
  - Trajectory following algorithms
- **Algorithms**:
  - Complementary filter for attitude estimation
  - Bézier curve interpolation for smooth trajectories
  - Lookahead-based waypoint tracking
  - Real-time coordinate transformations
  - LightGlue graph neural network for feature matching
  - RANSAC for robust homography estimation
  - PnP (Perspective-n-Point) for pose estimation
- **Visual Localization**:
  - LightGlue: Deep learning-based feature matching
  - SuperPoint/ALIKED: Feature extraction
  - Homography estimation and geometric verification
- **Sensors**: IMU (MPU-9250), barometer (BMP280), time-of-flight sensors (VL53L1x), camera (for visual localization)
- **Mathematical Tools**: Linear algebra, quaternion rotations, coordinate transformations, graph neural networks, attention mechanisms

## References

1. Lindenberger, P., Sarlin, P. E., & Pollefeys, M. (2023). LightGlue: Local Feature Matching at Light Speed. *International Conference on Computer Vision (ICCV)*.

2. DeTone, D., Malisiewicz, T., & Rabinovich, A. (2018). SuperPoint: Self-Supervised Interest Point Detection and Description. *Conference on Computer Vision and Pattern Recognition (CVPR)*.

3. Zhao, X., et al. (2023). ALIKED: A Lighter Keypoint and Descriptor Extraction Network via Deformable Transformation. *International Conference on Computer Vision (ICCV)*.

4. Sarlin, P. E., et al. (2020). SuperGlue: Learning Feature Matching with Graph Neural Networks. *Conference on Computer Vision and Pattern Recognition (CVPR)*.

5. Bitcraze AB. (2022). Crazyflie 2.1 Documentation. Retrieved from https://www.bitcraze.io/documentation/

## Conclusion

The Crazyflie flight test project provided hands-on experience with autonomous UAV systems, from hardware setup to control system development and real-world flight testing. The platform's open-source nature and comprehensive documentation made it an excellent choice for learning and experimentation.

The skills and knowledge gained from this project have been valuable for subsequent work in robotics and autonomous systems, particularly in areas such as sensor fusion, control systems, and real-time embedded programming.

---

*This project was conducted in early 2022 as part of my work in autonomous systems and robotics. The Crazyflie platform continues to be a valuable tool for research and development in UAV applications.*
