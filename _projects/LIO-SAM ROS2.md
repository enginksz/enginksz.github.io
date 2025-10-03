---
layout: page
title: LIO-SAM Gazebo ROS2
description: This repository contains the implementation of LIO-SAM (Lidar Inertial Odometry via Smoothing and Mapping) integrated with ROS 2 (Robot Operating System 2) for robust and accurate SLAM (Simultaneous Localization and Mapping) applications.
img: assets/img/liosam.jpg
importance: 2
category: work
giscus_comments: true
---

This repository contains the implementation of LIO-SAM (Lidar Inertial Odometry via Smoothing and Mapping) integrated with ROS 2 (Robot Operating System 2) for robust and accurate SLAM (Simultaneous Localization and Mapping) applications. LIO-SAM leverages both Lidar and IMU (Inertial Measurement Unit) data to achieve high accuracy in mapping and localization in real-time.

## Features

- **Tightly coupled integration**: Combines Lidar and IMU data for improved accuracy in challenging environments.
- **Real-time performance**: Designed for efficient computation, making it suitable for embedded systems.
- **Factor graph optimization**: Utilizes advanced optimization techniques to refine localization estimates.
- **ROS 2 support**: Facilitates easy communication and control in robotics applications.

## Requirements

To run this project, you need to have the following installed:

- ROS 2 (Foxy, Galactic, or later)
- Gazebo (for simulation)
- Colcon (build tool for ROS 2 packages)
- A compatible Lidar and IMU setup

## Installation

1. Clone this repository to your local machine:

   ```bash
   git clone https://github.com/TixiaoShan/LIO-SAM.git
   cd LIO-SAM
   ```

2. Checkout the ROS 2 branch:

   ```bash
   git checkout ros2
   ```

3. Build the package using Colcon:

   ```bash
   colcon build
   ```

4. Source the setup file:

   ```bash
   source install/setup.bash
   ```

## Running LIO-SAM with ROS 2

1. Launch LIO-SAM nodes in the first terminal:

   ```bash
   ros2 launch lio_sam run.launch.py
   ```

2. Launch the Gazebo simulation in a second terminal:

   ```bash
   ros2 launch robot_gazebo robot_sim.launch.py
   ```

## Usage

### ROS 2 Topics

- `/lidar_points`: Raw point cloud data from the Lidar sensor.
- `/imu/data`: IMU data providing orientation and acceleration.
- `/tf`: Transformation frames for the robot's pose.
- `/map`: The generated map from the LIO-SAM algorithm.
- `/odometry`: Estimated odometry data from LIO-SAM.

You can visualize the results in Rviz or any other compatible visualization tool by running:

```bash
ros2 run rviz2 rviz2

## Contributing

Feel free to submit issues, create pull requests, or fork this repository to improve and extend the project.
```
