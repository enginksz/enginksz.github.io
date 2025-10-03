---
layout: page
title: Use of Kalman Filter in Autonomous Vehicles
description: This project explores the implementation of obstacle avoidance in autonomous systems using imitation learning. The aim is to train agents to navigate environments while effectively avoiding obstacles.
img: assets/img/kalmanfilter.png
importance: 2
category: work
giscus_comments: true
---

# Use of Kalman Filter in Autonomous Vehicles

In autonomous driving systems, the detection of static and dynamic objects in the environment is a critical requirement for the safe and efficient control of vehicle motion dynamics. These systems utilize various sensors such as **LIDAR, radar, cameras, and GPS** to collect environmental data, which is then used to plan the vehicle's movements. The data obtained from these sensors allows for the detection and tracking of other vehicles, pedestrians, cyclists, and other objects in the vicinity.

However, sensor data alone is insufficient. Autonomous systems must be capable of **accurately predicting the future positions** of both the vehicle itself and surrounding dynamic objects. These predictions are crucial for **motion planning, route optimization**, and **collision avoidance systems**. Specifically, parameters such as speed, direction, and acceleration must be dynamically analyzed to identify potential collision scenarios in advance.

This prediction process is challenging due to noise in sensor data and uncertainties in system modeling. Sensor measurements can be corrupted by environmental conditions or inherent limitations of the sensors themselves. For instance, radar signals may weaken in rainy conditions, LIDAR devices might misinterpret objects in foggy environments, and camera sensors can be sensitive to changes in lighting. These uncertainties are persistent challenges in real-world scenarios and necessitate strategies to minimize such errors.

At this juncture, sensor fusion and prediction algorithms become essential in autonomous driving systems. Combining data from different sensors compensates for the weaknesses of individual sensors, resulting in more reliable environmental perception. The Kalman Filter is a widely used method in such sensor fusion systems. The Kalman Filter provides optimal state estimations for linear dynamic systems. By utilizing current sensor measurements, the vehicle estimates its own state (position, velocity, acceleration) while simultaneously predicting the future states of surrounding objects.

The Kalman Filter minimizes measurement noise and model uncertainties from each sensor, delivering the most probable state and prediction. Operating with mathematical tools such as state-space models, noise covariance matrices, and dynamic system matrices, this filter continuously updates and refines the system's state changes over time. Essentially, the Kalman Filter integrates the current state with the predicted state, minimizing noise levels and errors to provide a final, accurate prediction.

Successful implementation of this approach requires modeling linear dynamic systems, matrix algebra, Markov processes, and covariance analyses—advanced mathematical techniques. Additionally, each sensor's data accuracy and error models must be meticulously analyzed and appropriately integrated into the filter.

Through these techniques, autonomous driving systems not only comprehend the current state but also predict future states with high accuracy by reducing uncertainties, thereby ensuring safe and efficient driving.

# Kalman Filter

##  Sensor Data and Noise

An autonomous vehicle receives data from a GPS sensor with an accuracy precision of approximately 5-10 meters. This level of accuracy is inadequate for autonomous driving, as higher precision is required to anticipate and prevent accidents. For example, in a forested or rural area where the robot navigates through terrains with pits and cliffs, inaccurate measurements exceeding a few meters could result in the robot falling. Therefore, relying solely on GPS data is insufficient. Instead, it is necessary to integrate the vehicle's motion dynamics with data from additional sensors.

We also possess knowledge about the robot's movement: commands are sent to the wheel or track motors, and if the robot is moving in a particular direction without any obstacles, it is likely to continue moving in that direction in the next instant. However, we do not have complete information about the movement: the robot can be pushed by wind, wheels might slip slightly, or it may roll unpredictably on uneven terrain. Consequently, the amount of wheel rotation may not accurately represent the actual distance traveled by the robot, leading to imperfect predictions. Additionally, sensors used to monitor the robot's movement (e.g., accelerometers and gyroscopes) can be subject to noise, affecting the accuracy of predictions. These uncertainties and errors are critical factors that must be considered in the robot's position and motion estimations.

## Mathematical Model Construction

Assume that our autopilot system has a state vector (𝑥) representing the vehicle's position (𝑝) and velocity (𝑣):

                           𝑥𝑘 = [ 𝑝 𝑣]

The state vector encompasses the fundamental parameters of our system, which can be expanded as needed. For instance, the state vector could include fuel level, engine temperature, or data from other sensors.

##  State Transition Model
To model the movement of our autopilot system, we use the state transition equation. These equations predict the vehicle's next state based on the current state and control inputs. A simple linear model is expressed as follows:

### Kalman Filter Steps

The Kalman Filter algorithm consists of two main stages: Prediction and Update. These steps reduce noise and uncertainties in sensor data, enabling more accurate state estimations.

Prediction: The current state is predicted.
Update: The predicted state is updated using measurement data.

<div class="row justify-content-sm-center">
    <div class="col-sm-8 mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/kalmansteps.png" title="Kalman Filter Steps Image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

Here:

𝑥^𝑘 ∣𝑘−1 : Predicted state at time 𝑘

𝑥^𝑘∣𝑘 : Updated state at time 𝑘

𝑃𝑘∣𝑘−1 : Predicted error covariance at time 𝑘

𝑃𝑘∣𝑘 : Updated error covariance at time 𝑘

𝐴: State transition matrix

𝐵: Control input matrix

𝑢𝑘 : Control input at time 𝑘

𝑄: Process noise covariance matrix

𝐻: Measurement matrix

𝑅: Measurement noise covariance matrix

𝑧𝑘 : Measurement at time 𝑘

𝐼: Identity matrix

As previously mentioned, understanding the Kalman Filter from definitions and complex equations alone is nearly impossible. Therefore, in most cases, state matrices are simplified, resulting in the more manageable equations shown below:

<div class="row justify-content-sm-center">
    <div class="col-sm-8 mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/kalmanstep1.png" title="Kalman Filter Measurement Image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

Here, the subscript 𝑘 indices represent different states. These indices can be considered as separate time intervals; for example, 𝑘=1 corresponds to 1 ms, and 
𝑘=2 corresponds to 2 ms.

#### Step 1: Model Construction

In the first step, we perform model construction, thereby determining the 
𝐴, 𝐵, and 𝐻 matrices. These matrices are typically constant values, often equal to 1.

#### Step 2: Determining Noise Matrices

The most challenging step is determining the 𝑅(measurement noise covariance matrix) and 
𝑄 (process noise covariance matrix). Finding 𝑅 is generally more straightforward because we have clear information about the sources of environmental noise. However, determining 𝑄 is more complex, and providing a specific method at this stage is difficult.


#### Step 3: Initializing the Process

To initialize the Kalman Filter, we need initial estimates for the starting state (𝑥0) and the initial error covariance (𝑃0).

Measurement uncertainty (i.e., variance) determines how much the Kalman Gain (𝐾) should be weighted. Specifically, the higher the measurement uncertainty, the smaller the Kalman Gain needs to be to prevent the measurement from overly influencing the state vector estimation.

<div class="row justify-content-sm-center">
    <div class="col-sm-8 mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/kalmanstep2.png" title="Kalman Filter Measurement Image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

Don't be intimidated by this expression. We are performing matrix multiplications, but fundamentally, what happens is that we multiply matrices containing the measurement variance and then invert them, subsequently multiplying by the predicted covariance. Because we take the inverse, as the measurement variance increases, the value of this multiplication decreases. Consequently, the higher the measurement variance, the less informative the measurement is. The Kalman Gain ensures that this information is appropriately factored into the final state estimation. I recommend deriving this expression manually to gain a deeper understanding.

Thus, in the next step of the Kalman Filter—the update phase—we use the update equations to obtain the final state estimation (𝑥^𝑘) and its covariance as described above.

#### Iteration Process

The obtained state vector prediction and covariance prediction are used as prior information in the next step. Here, we employ a Bayesian approach. The posterior distribution obtained is used as the prior distribution in the next iteration. The filter continues to operate iteratively, and as new information arrives, the variance of the estimation error decreases.


## Kalman Filter Implementation: C++ Example

Let's implement a simple Kalman Filter in C++:

```cpp
#include <iostream>
#include <Eigen/Dense> 

using namespace Eigen;

class KalmanFilter {
public:
    KalmanFilter(double dt, const MatrixXd& A, const MatrixXd& B, const MatrixXd& H, const MatrixXd& Q, const MatrixXd& R, const MatrixXd& P)
        : A(A), B(B), H(H), Q(Q), R(R), P(P), I(A.rows(), A.cols()), x(A.rows())
    {
        I.setIdentity();
        x.setZero();
        this->dt = dt;
    }

    void init(const VectorXd& x0) {
        x = x0;
    }

    void predict(const VectorXd& u) {
        x = A * x + B * u;
        P = A * P * A.transpose() + Q;
    }

    void update(const VectorXd& z) {
        VectorXd y = z - H * x;
        MatrixXd S = H * P * H.transpose() + R;
        MatrixXd K = P * H.transpose() * S.inverse();
        x = x + K * y;
        P = (I - K * H) * P;
    }

    VectorXd state() const {
        return x;
    }

private:
    MatrixXd A, B, H, Q, R, P, I;
    VectorXd x;
    double dt;
};

int main() {
    // Time step
    double dt = 1.0;

    // State transition matrix (A)
    MatrixXd A(2, 2);
    A << 1, dt,
         0, 1;

    // Control input matrix (B)
    MatrixXd B(2, 1);
    B << 0.5 * dt * dt,
         dt;

    // Measurement matrix (H)
    MatrixXd H(1, 2);
    H << 1, 0;

    // Process noise covariance matrix (Q)
    MatrixXd Q(2, 2);
    Q << 1, 0,
         0, 1;

    // Measurement noise covariance matrix (R)
    MatrixXd R(1, 1);
    R << 1;

    // Initial error covariance matrix (P)
    MatrixXd P(2, 2);
    P << 1, 0,
         0, 1;

    // Kalman Filter object
    KalmanFilter kf(dt, A, B, H, Q, R, P);

    // Initial state
    VectorXd x0(2);
    x0 << 0, 0;
    kf.init(x0);

    // Control input (acceleration) and measurement data
    VectorXd u(1);
    u << 1; // Constant acceleration

    VectorXd z(1); // Measurement data

    for (int i = 0; i < 10; ++i) {
        kf.predict(u);

        // Example measurement (adding a bit of random noise to the true value)
        z << i + 0.1 * ((double)rand() / RAND_MAX - 0.5);

        kf.update(z);

        std::cout << "Estimated state: " << kf.state().transpose() << std::endl;
    }

    return 0;
}

```

In this code, we implemented a Kalman filter using the Eigen library. Eigen allows us to easily perform linear algebra operations. We defined all the matrices and the initial state required for creating the Kalman filter. In the main function, the prediction and update steps are executed over a certain time step, and the results are printed.

##  Advantages of the Kalman Filter

Optimal State Estimation: Under the assumptions of linearity and Gaussian distributions, the Kalman filter provides the best minimum mean square error (MMSE) estimate.
Noise Reduction: It accounts for both process and measurement noise, yielding more accurate predictions.

Real-Time Application: Being computationally efficient, it can be applied in real-time systems.

## Application Example

In our autopilot system, the Kalman filter is applied using the vehicle's current position and speed alongside measurements from the GPS. The state transition model includes the vehicle's dynamics and control inputs (e.g., acceleration or steering angle), while the observation model represents the accuracy of the data coming from the GPS sensor. The Kalman filter combines these two pieces of information to provide a more accurate estimation of the vehicle’s position and speed.

For instance, after predicting the position and speed based on control inputs, the measurements from the GPS are used to correct these predictions. If the GPS measurements are noisy, the Kalman filter takes this noise into account, offering a more reliable estimate. Thus, the vehicle’s position and speed can be tracked with higher accuracy, enabling necessary decisions for safe driving.


```python
import numpy as np
import matplotlib.pyplot as plt

class KalmanFilter:
    def __init__(self, A, B, H, Q, R, P, x):
        """
        Initializes the Kalman Filter.
        
        Parameters:
        - A: State transition matrix
        - B: Control input matrix
        - H: Observation matrix
        - Q: Process noise covariance
        - R: Measurement noise covariance
        - P: Initial estimation error covariance
        - x: Initial state estimate
        """
        self.A = A
        self.B = B
        self.H = H
        self.Q = Q
        self.R = R
        self.P = P
        self.x = x

    def predict(self, u):
        """
        Predicts the next state and estimation error covariance.
        
        Parameters:
        - u: Control input
        """
        self.x = np.dot(self.A, self.x) + np.dot(self.B, u)
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q

    def update(self, z):
        """
        Updates the state estimate and estimation error covariance using the measurement z.
        
        Parameters:
        - z: Measurement vector
        """
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        y = z - np.dot(self.H, self.x)
        self.x = self.x + np.dot(K, y)
        I = np.eye(self.A.shape[0])
        self.P = np.dot((I - np.dot(K, self.H)), self.P)

def simulate_data(num_steps, dt, initial_position, initial_velocity, acceleration, gps_noise_std):
    """
    Simulates true state and noisy GPS measurements.
    
    Parameters:
    - num_steps: Number of time steps
    - dt: Time step duration
    - initial_position: Initial position of the vehicle
    - initial_velocity: Initial velocity of the vehicle
    - acceleration: Constant acceleration of the vehicle
    - gps_noise_std: Standard deviation of GPS measurement noise
    
    Returns:
    - true_positions: Array of true positions
    - true_velocities: Array of true velocities
    - measurements: Array of GPS measurements
    """
    true_positions = []
    true_velocities = []
    measurements = []

    position = initial_position
    velocity = initial_velocity

    for _ in range(num_steps):
        # Update true state
        velocity += acceleration * dt
        position += velocity * dt

        true_positions.append(position)
        true_velocities.append(velocity)

        # Simulate GPS measurement with noise
        gps_measurement = position + np.random.normal(0, gps_noise_std)
        measurements.append(gps_measurement)

    return np.array(true_positions), np.array(true_velocities), np.array(measurements)

def main():
    # Simulation parameters
    num_steps = 50
    dt = 1.0  # time step (seconds)
    initial_position = 0.0  # meters
    initial_velocity = 20.0  # meters/second
    acceleration = 1.0  # meters/second^2
    gps_noise_std = 10.0  # meters

    # Simulate true state and measurements
    true_positions, true_velocities, measurements = simulate_data(
        num_steps, dt, initial_position, initial_velocity, acceleration, gps_noise_std
    )

    # Define Kalman Filter matrices
    A = np.array([[1, dt],
                  [0, 1]])  # State transition matrix

    B = np.array([[0.5 * dt**2],
                  [dt]])  # Control input matrix

    H = np.array([[1, 0]])  # Observation matrix

    Q = np.array([[1, 0],
                  [0, 1]])  # Process noise covariance

    R = np.array([[gps_noise_std**2]])  # Measurement noise covariance

    P = np.array([[1000, 0],
                  [0, 1000]])  # Initial estimation error covariance

    x_initial = np.array([[0],
                          [0]])  # Initial state estimate

    # Control input (acceleration)
    u = np.array([[acceleration]])

    # Initialize Kalman Filter
    kf = KalmanFilter(A, B, H, Q, R, P, x_initial)

    # Lists to store estimates
    estimated_positions = []
    estimated_velocities = []

    for i in range(num_steps):
        # Predict
        kf.predict(u)

        # Update with measurement
        z = np.array([[measurements[i]]])
        kf.update(z)

        # Store estimates
        estimated_positions.append(kf.x[0, 0])
        estimated_velocities.append(kf.x[1, 0])

    # Plotting the results
    time_steps = np.arange(num_steps) * dt

    plt.figure(figsize=(12, 6))

    # Position plot
    plt.subplot(2, 1, 1)
    plt.plot(time_steps, true_positions, label='True Position', color='g')
    plt.scatter(time_steps, measurements, label='GPS Measurements', color='r', marker='x')
    plt.plot(time_steps, estimated_positions, label='Kalman Filter Estimate', color='b')
    plt.title('Kalman Filter Position Estimation')
    plt.xlabel('Time [s]')
    plt.ylabel('Position [m]')
    plt.legend()
    plt.grid(True)

    # Velocity plot
    plt.subplot(2, 1, 2)
    plt.plot(time_steps, true_velocities, label='True Velocity', color='g')
    plt.plot(time_steps, estimated_velocities, label='Kalman Filter Estimate', color='b')
    plt.title('Kalman Filter Velocity Estimation')
    plt.xlabel('Time [s]')
    plt.ylabel('Velocity [m/s]')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()


```

### Graphics

<div class="row justify-content-sm-center">
    <div class="col-sm-10 mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/Figure_1.png" title="Kalman Filter Measurement Image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>


<div class="row justify-content-sm-center">
    <div class="col-sm-10 mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/Figure_2.png" title="Kalman Filter Measurement Image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>


## Conclusion
The Kalman Filter minimizes uncertainties in autonomous driving systems, providing more accurate and reliable state estimations. Through the state transition model and the observation model, it effectively combines the vehicle's dynamics and environmental data. This enables both understanding the current state and predicting future movements. Successful implementation of the Kalman Filter requires advanced mathematical techniques such as modeling linear dynamic systems, matrix algebra, Markov processes, and covariance analyses.


Kalman Filter PDF [https://web.mit.edu/kirtley/kirtley/binlustuff/literature/control/Kalman%20filter.pdf]

Tayyarg GitHub - Kalman Filtering [https://tayyarg.github.io/kalman-filtreleme/]
