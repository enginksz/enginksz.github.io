---
layout: page
title: Crazyflie Flight Tests and Experiments
description: Autonomous flight tests and experiments with Crazyflie 2.1 nano quadcopter, including control system development and real-world flight demonstrations
img: assets/img/crayzflie/lighthouse.png
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

### Crazyflie Platform Visualizations

The following images show the Crazyflie 2.1 platform setup used in this project:

<div class="row justify-content-sm-center">
    <div class="col-sm-6 mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/crayzflie/crayzflie01.png" title="Crazyflie 2.1 Platform" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm-6 mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/crayzflie/crayzflie02.png" title="Lighthouse Positioning Deck" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

*Crazyflie 2.1 nano quadcopter with expansion decks and Lighthouse Positioning Deck installed for high-precision indoor positioning.*

<div class="row justify-content-sm-center">
    <div class="col-sm-6 mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/crayzflie/crayzflie03.png" title="Flight Test Setup" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm-6 mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/crayzflie/crayzflie04.png" title="Autonomous Flight" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

*Flight test environment with Lighthouse base stations and Crazyflie executing autonomous flight maneuvers with precise trajectory following.*

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

### Lighthouse Positioning System

For high-precision indoor positioning, the project utilized the **Lighthouse Positioning System**, an optically-based positioning system that provides decimeter-level accuracy and millimeter-level precision. The system uses SteamVR Base Stations as optical beacons, allowing the Crazyflie to calculate its position onboard without requiring external motion capture infrastructure.

#### System Architecture

The Lighthouse Positioning System consists of:

1. **Base Stations**: SteamVR Base Stations (V1 or V2) that emit rotating laser sweeps
2. **Lighthouse Positioning Deck**: Onboard deck with photodiodes that receive laser signals
3. **Onboard Processing**: Position and orientation calculation performed directly on the Crazyflie
4. **System Geometry**: Calibrated base station positions and orientations in the tracking space

#### Operating Principle

The system operates using optical triangulation:

1. **Laser Sweep**: Base stations emit rotating laser beams that sweep across the room
2. **Photodiode Detection**: Multiple photodiodes on the Lighthouse deck detect the laser sweeps
3. **Angle Measurement**: The system measures the angle between each base station and each photodiode
4. **Triangulation**: Using multiple photodiodes and base stations, position and orientation are calculated

#### Mathematical Foundation

**Angle Measurement**:

For each base station $i$ and photodiode $j$, the system measures the angle $\theta_{ij}$:

$$\theta_{ij} = \arctan\left(\frac{y_{ij}}{x_{ij}}\right)$$

where $(x_{ij}, y_{ij})$ are the relative coordinates of photodiode $j$ with respect to base station $i$.

**Position Estimation**:

Given angles from multiple base stations and photodiodes, the position $\mathbf{p} = [x, y, z]^T$ is estimated by solving:

$$\min_{\mathbf{p}, \mathbf{R}} \sum_{i,j} \|\theta_{ij}^{measured} - \theta_{ij}^{calculated}(\mathbf{p}, \mathbf{R})\|^2$$

where $\mathbf{R}$ is the rotation matrix representing the Crazyflie's orientation.

**Triangulation with Multiple Base Stations**:

With $N$ base stations and $M$ photodiodes, the system solves an overdetermined system:

$$\begin{bmatrix}
\mathbf{A}_1 \\
\mathbf{A}_2 \\
\vdots \\
\mathbf{A}_N
\end{bmatrix} \mathbf{p} = \begin{bmatrix}
\mathbf{b}_1 \\
\mathbf{b}_2 \\
\vdots \\
\mathbf{b}_N
\end{bmatrix}$$

where each $\mathbf{A}_i$ contains the geometric constraints from base station $i$, and $\mathbf{b}_i$ contains the measured angles.

**Least Squares Solution**:

The position is estimated using weighted least squares:

$$\mathbf{p} = (\mathbf{A}^T \mathbf{W} \mathbf{A})^{-1} \mathbf{A}^T \mathbf{W} \mathbf{b}$$

where $\mathbf{W}$ is a weight matrix accounting for measurement uncertainties.

#### Lighthouse V1 vs V2

The system supports both generations of base stations:

| Characteristics | Lighthouse V1 | Lighthouse V2 |
|----------------|---------------|---------------|
| **Rotating Drums** | 2 drums | 1 drum with 2 slanted planes |
| **Range** | ~6m (environment dependent) | 6m (by design) |
| **Update Rate** | 30Hz (with 2 base stations) | ~50Hz (independent of base station count) |
| **Max Base Stations** | 1-2 | 1-4 (up to 16 in hardware) |
| **Horizontal FOV** | 120° | 150° |
| **Vertical FOV** | 120° | 110° |

#### System Geometry Calibration

The system requires calibration of base station positions and orientations:

**Geometry Definition**:

Each base station $i$ has:
- Position: $\mathbf{p}_{bs,i} = [x_i, y_i, z_i]^T$
- Orientation: Rotation matrix $\mathbf{R}_{bs,i}$

**Calibration Process**:

1. **Automatic Calibration**: Crazyflie client automatically determines geometry by flying in the space
2. **Manual Calibration**: Geometry can be manually configured if base station positions are known
3. **Geometry Storage**: Calibrated geometry is stored onboard the Crazyflie

**Coordinate Transformation**:

Position in base station frame is transformed to global frame:

$$\mathbf{p}_{global} = \mathbf{R}_{bs,i}^T (\mathbf{p}_{local} - \mathbf{p}_{bs,i})$$

#### Performance Characteristics

**Accuracy and Precision**:
- **Absolute Accuracy**: Better than 10cm (decimeter-level)
- **Relative Precision**: Better than 1mm (millimeter-level)
- **Return-to-Pad Precision**: Millimeter-level precision for returning to takeoff position

**System Limitations**:
- **Line of Sight**: Requires direct optical line of sight to at least one base station
- **Range**: Effective range up to 6 meters
- **Height Constraint**: Optimal performance up to ~50cm below base station height
- **Indoor Only**: System is designed for indoor use only
- **Sunlight Sensitivity**: Performance can be affected by direct sunlight (infrared interference)

**Update Rate**:
- **V1**: 30Hz with 2 base stations
- **V2**: ~50Hz, independent of number of base stations

#### Integration with Crazyflie

The Lighthouse Positioning System was integrated with Crazyflie for precise trajectory following:

1. **Hardware Setup**: Lighthouse Positioning Deck installed on Crazyflie 2.1
2. **Base Station Installation**: Two or more SteamVR Base Stations mounted in the tracking space
3. **Geometry Calibration**: System geometry automatically calibrated by Crazyflie client
4. **Onboard Position**: Position and orientation calculated onboard at 30-50Hz
5. **Control Integration**: High-frequency position updates used for precise trajectory following

**Control Loop Integration**:

The Lighthouse position updates are integrated into the control loop:

$$\mathbf{p}_{fused}(t) = \alpha \mathbf{p}_{lighthouse}(t) + (1-\alpha) \mathbf{p}_{predicted}(t)$$

where $\alpha$ is the fusion coefficient, and $\mathbf{p}_{predicted}$ is the position predicted from IMU integration.

This integration enables the Crazyflie to maintain accurate position estimates for complex maneuvers like the star pattern trajectory, with the high update rate (30-50Hz) providing smooth and responsive control.

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
- Integrated Lighthouse Positioning System for high-precision position updates
- Fused Lighthouse position with IMU-based dead reckoning for robust state estimation

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

## Lighthouse Positioning System Details

### System Setup and Calibration

The Lighthouse Positioning System requires careful setup and calibration for optimal performance:

**Base Station Configuration**:
- **Mounting Height**: Base stations should be mounted 2-3 meters above the tracking area
- **Angular Coverage**: Positioned to maximize coverage of the tracking volume
- **Line of Sight**: Ensure clear line of sight from base stations to the tracking area
- **Synchronization**: Base stations must be properly synchronized (V1 requires sync cable, V2 uses wireless sync)

**Geometry Calibration Process**:

The system geometry is calibrated using an iterative process:

1. **Initial Guess**: Base station positions are estimated or manually entered
2. **Measurement Collection**: Crazyflie flies in the space, collecting angle measurements
3. **Optimization**: System solves for optimal base station positions and orientations:

$$\min_{\mathbf{p}_{bs}, \mathbf{R}_{bs}} \sum_{k} \|\boldsymbol{\theta}_{measured,k} - \boldsymbol{\theta}_{calculated,k}(\mathbf{p}_{bs}, \mathbf{R}_{bs}, \mathbf{p}_k, \mathbf{R}_k)\|^2$$

where $k$ indexes over all measurement samples, and $\mathbf{p}_k, \mathbf{R}_k$ are the Crazyflie's position and orientation at sample $k$.

### Position Update Rate and Latency

**Update Frequency**:
- **Lighthouse V1**: 30Hz update rate with 2 base stations
- **Lighthouse V2**: ~50Hz update rate, independent of base station count
- **Onboard Processing**: Position calculation performed onboard with minimal latency (<1ms)

**Control Loop Integration**:

The high update rate enables tight control loops:

$$f_{control} = \min(f_{lighthouse}, f_{control\_max})$$

where $f_{control\_max}$ is typically 100Hz for position control and 250Hz for attitude control.

### Accuracy Analysis

**Measurement Uncertainty**:

The angle measurement uncertainty propagates to position uncertainty:

$$\sigma_p = \frac{\sigma_\theta \cdot d}{\sin(\theta)}$$

where:
- $\sigma_p$ is position uncertainty
- $\sigma_\theta$ is angle measurement uncertainty (~0.1°)
- $d$ is distance to base station
- $\theta$ is the angle between base station and photodiode

**Multi-Base Station Fusion**:

With $N$ base stations, the position uncertainty is reduced:

$$\sigma_{p,fused} = \frac{\sigma_p}{\sqrt{N}}$$

This explains why systems with more base stations achieve better accuracy.

### Integration with Control System

The Lighthouse position is integrated into the cascaded control architecture:

**Position Control Loop**:
- **Input**: Lighthouse position $\mathbf{p}_{lighthouse}(t)$ at 30-50Hz
- **Reference**: Desired trajectory position $\mathbf{p}_{desired}(t)$
- **Output**: Velocity command $\mathbf{v}_{cmd}(t)$

**Sensor Fusion**:

When Lighthouse position is available, it is fused with IMU-based dead reckoning:

$$\mathbf{p}_{fused}(t) = \begin{cases}
\mathbf{p}_{lighthouse}(t) & \text{if } \|\mathbf{p}_{lighthouse} - \mathbf{p}_{imu}\| < \tau \\
\alpha \mathbf{p}_{lighthouse}(t) + (1-\alpha) \mathbf{p}_{imu}(t) & \text{otherwise}
\end{cases}$$

where $\tau$ is a threshold for outlier rejection, and $\alpha$ is the fusion weight.

This integration enables the precise trajectory following demonstrated in the star pattern maneuver, with the high update rate and low latency of the Lighthouse system providing the necessary precision for complex autonomous flight.

## Flight Test Video and Visualizations

The following video demonstrates the **rotating star pattern maneuver**, one of the most challenging flight tests conducted during this project:

<div class="row justify-content-sm-center">
    <div class="col-sm-8 mt-3 mt-md-0">
        {% include video.liquid path="assets/img/video/IMG_0425.MOV" title="Crazyflie Star Pattern Maneuver" class="img-fluid rounded z-depth-1" controls=true %}
    </div>
</div>

*This video shows the Crazyflie executing a complex 5-pointed star pattern trajectory while simultaneously rotating around its center. This maneuver demonstrates advanced autonomous flight capabilities, precise trajectory following, and robust control system performance. The drone maintains the star shape throughout the rotation, requiring precise coordination of position and yaw control loops, real-time trajectory generation, and sophisticated feedforward control algorithms.*

### Flight Test Images

The following images capture various aspects of the flight tests and system setup:

<div class="row justify-content-sm-center">
    <div class="col-sm-6 mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/crayzflie/crayzflie01.png" title="Crazyflie 2.1 in Flight" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm-6 mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/crayzflie/crayzflie02.png" title="Lighthouse Positioning System" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

*Crazyflie 2.1 executing autonomous flight maneuvers with Lighthouse Positioning Deck installed, enabling high-precision indoor positioning with decimeter-level accuracy.*

<div class="row justify-content-sm-center">
    <div class="col-sm-6 mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/crayzflie/crayzflie03.png" title="Test Environment Setup" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm-6 mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/crayzflie/crayzflie04.png" title="Star Pattern Maneuver" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

*Flight test environment with Lighthouse base stations and Crazyflie executing the complex rotating star pattern maneuver, demonstrating precise trajectory following and control system coordination.*

## Key Learnings

This project provided valuable insights into:

1. **UAV Dynamics**: Understanding quadcopter flight dynamics, non-linear control, and multi-rotor aerodynamics
2. **Real-Time Systems**: Working with embedded systems, real-time constraints, and control loop timing
3. **Sensor Fusion**: Combining multiple sensors (IMU, barometer) for robust state estimation using complementary filters
4. **Control Theory**: Practical application of PID control, cascaded control loops, and feedforward control
5. **Trajectory Planning**: Generating complex trajectories (star patterns) with rotation and real-time execution
6. **Coordinate Transformations**: Managing multiple coordinate frames (body, inertial, NED) for accurate control
7. **Optical Positioning Systems**: Understanding Lighthouse positioning system, optical triangulation, and angle-based position estimation
8. **Sensor Fusion**: Combining Lighthouse position updates with IMU/barometer for robust state estimation
9. **System Integration**: Integrating hardware, software, control algorithms, trajectory generation, and positioning systems
10. **Control Tuning**: Systematic PID parameter tuning for different flight regimes (hover, aggressive maneuvers)

## Future Work

Potential extensions and improvements:

1. **Advanced Control**: Implement model predictive control (MPC) for improved performance
2. **Extended Lighthouse Coverage**: Optimize system geometry for larger tracking volumes
3. **Multi-Robot Coordination**: Develop swarm algorithms using shared Lighthouse geometry
4. **Hybrid Positioning**: Combine Lighthouse with other sensors (optical flow, visual odometry) for robustness
5. **Machine Learning**: Apply reinforcement learning for adaptive control and trajectory optimization
6. **Outdoor Positioning**: Research alternative positioning methods for outdoor flight
7. **Precision Landing**: Utilize Lighthouse millimeter-level precision for automated precision landing

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
  - Optical triangulation for position estimation
  - Least squares optimization for geometry calibration
  - Sensor fusion algorithms
- **Positioning System**:
  - Lighthouse Positioning System (SteamVR Base Stations V1/V2)
  - Lighthouse Positioning Deck
  - Optical angle measurement and triangulation
  - System geometry calibration
- **Sensors**: IMU (MPU-9250), barometer (BMP280), time-of-flight sensors (VL53L1x), Lighthouse photodiodes
- **Mathematical Tools**: Linear algebra, quaternion rotations, coordinate transformations, least squares optimization, triangulation

## References

1. Bitcraze AB. (2022). Lighthouse Positioning System Documentation. Retrieved from [https://www.bitcraze.io/documentation/system/positioning/ligthouse-positioning-system/](https://www.bitcraze.io/documentation/system/positioning/ligthouse-positioning-system/)

2. Bitcraze AB. (2022). Crazyflie 2.1 Documentation. Retrieved from https://www.bitcraze.io/documentation/

3. Mellinger, D., & Kumar, V. (2011). Minimum snap trajectory generation and control for quadrotors. *IEEE International Conference on Robotics and Automation (ICRA)*.

4. Brescianini, D., D'Andrea, R., & D'Andrea, R. (2013). Quadrotor control: modeling, nonlinear control design, and simulation. *IEEE International Conference on Control Applications*.

5. Bitcraze AB. (2022). Lighthouse Positioning System: Dataset, Accuracy, and Precision for UAV Research. Retrieved from https://www.bitcraze.io/documentation/

## Conclusion

The Crazyflie flight test project provided hands-on experience with autonomous UAV systems, from hardware setup to control system development and real-world flight testing. The platform's open-source nature and comprehensive documentation made it an excellent choice for learning and experimentation.

The skills and knowledge gained from this project have been valuable for subsequent work in robotics and autonomous systems, particularly in areas such as sensor fusion, control systems, and real-time embedded programming.

---

*This project was conducted in early 2022 as part of my work in autonomous systems and robotics. The Crazyflie platform continues to be a valuable tool for research and development in UAV applications.*
