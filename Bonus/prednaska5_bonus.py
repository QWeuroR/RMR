import numpy as np

def kalman_filter_step(x_k_1, P_k_1, u_k, z_k, A_k, B_k, Q_k, R_k, H_k):
    # Prediction step
    x_k_predict = A_k @ x_k_1 + B_k @ u_k
    P_k_predict = A_k @ P_k_1 @ A_k.T + Q_k
    
    # Use provided measurement
    z_k_used = z_k
    
    # Update step (Kalman gain)
    K_k = P_k_predict @ H_k.T @ np.linalg.inv(H_k @ P_k_predict @ H_k.T + R_k)
    
    # State and covariance update
    x_k = x_k_predict + K_k @ (z_k_used - H_k @ x_k_predict)
    P_k = (np.eye(len(P_k_predict)) - K_k @ H_k) @ P_k_predict

    return x_k, P_k, x_k_predict, P_k_predict, K_k, z_k_used


def kalman_filter_batch(measurements, control_inputs,
                        A_k, B_k, Q_k, R_k, H_k, x_init, P_init, dimensions):
   
    results = {
        'state_estimate': [],
        'covariance_estimate': [],
        'state_predict': [],
        'covariance_predict': [],
        'kalman_gain': [],
        'measurement': []
    }
    
    x_k_1 = x_init
    P_k_1 = P_init
    
    for k in range(len(measurements)):
        # Reshape inputs to column vectors based on dimensions
        if dimensions == 1:
            z_k = np.array([[measurements[k]]])
            u_k = np.array([[control_inputs[k]]])
        else:
            z_k = np.array([[m] for m in measurements[k]]) if isinstance(measurements[k], (list, tuple)) else np.array([[measurements[k]]])
            u_k = np.array([[c] for c in control_inputs[k]]) if isinstance(control_inputs[k], (list, tuple)) else np.array([[control_inputs[k]]])
        
        x_k, P_k, x_k_predict, P_k_predict, K_k, z_k_used = kalman_filter_step(
            x_k_1, P_k_1, u_k, z_k, A_k, B_k, Q_k, R_k, H_k
        )
        
        # Store results - flatten if 1D for display
        if dimensions == 1:
            results['state_estimate'].append(x_k[0, 0])
            results['covariance_estimate'].append(P_k[0, 0])
            results['state_predict'].append(x_k_predict[0, 0])
            results['covariance_predict'].append(P_k_predict[0, 0])
            results['kalman_gain'].append(K_k[0, 0])
            results['measurement'].append(z_k_used[0, 0])
        else:
            results['state_estimate'].append(x_k.flatten())
            results['covariance_estimate'].append(P_k)
            results['state_predict'].append(x_k_predict.flatten())
            results['covariance_predict'].append(P_k_predict)
            results['kalman_gain'].append(K_k)
            results['measurement'].append(z_k_used.flatten())
        
        # Update for next iteration
        x_k_1 = x_k
        P_k_1 = P_k
    
    return results


def setup_kalman_1d(measurement_noise=15.0, process_noise=30.0, sampling_frequency=10.0):
    """Setup 1D Kalman filter parameters
    
    Args:
        measurement_noise: Standard deviation of measurement noise v_k [mm]
        process_noise: Standard deviation of process noise w_k [mm/s] (velocity noise)
        sampling_frequency: Sampling frequency [Hz]
    
    Returns:
        A_k, B_k, Q_k, R_k, H_k, x_init, P_init
    """
    # Sampling time
    delta_t = 1.0 / sampling_frequency
    
    # Calculate covariances from noise parameters
    # R_k = variance of measurement noise = (std_dev)^2
    R_k = np.array([[measurement_noise ** 2]])
    
    # Q_k = process noise covariance
    # Q = (w * Δt)² / 3, where w is velocity noise and Δt is sampling time
    # The factor of 3 accounts for integration of white noise
    Q_k = np.array([[(process_noise * delta_t) ** 2 / 3.0]])
    
    A_k = np.array([[1.0]])
    B_k = np.array([[0.1]])
    H_k = np.array([[1.0]])
    x_init = np.array([[1000.0]])
    
    # P_init = base covariance + process noise contribution
    # P_init = 100000 + Q
    base_P = 100000.0
    P_init = np.array([[base_P + Q_k[0, 0]]])
    
    return A_k, B_k, Q_k, R_k, H_k, x_init, P_init


def setup_kalman_3d(measurement_noise_xy=15.0, measurement_noise_theta=0.5, 
                    process_noise_xy=30.0, process_noise_theta=0.1, sampling_frequency=10.0):
    """Setup 3D Kalman filter parameters (x, y, θ)
    
    Args:
        measurement_noise_xy: Std dev of x,y measurement noise [mm]
        measurement_noise_theta: Std dev of θ measurement noise [rad]
        process_noise_xy: Std dev of x,y process noise [mm/s] (velocity noise)
        process_noise_theta: Std dev of θ process noise [rad/s] (angular velocity noise)
        sampling_frequency: Sampling frequency [Hz]
    
    Returns:
        A_k, B_k, Q_k, R_k, H_k, x_init, P_init
    """
    # Sampling time
    delta_t = 1.0 / sampling_frequency
    
    A_k = np.eye(3)  # Identity for simple 1st order model
    B_k = np.eye(3)  # Identity control matrix
    
    # Calculate covariances from noise parameters using Q = (w * Δt)² / 3
    Q_k = np.diag([
        (process_noise_xy * delta_t) ** 2 / 3.0,
        (process_noise_xy * delta_t) ** 2 / 3.0,
        (process_noise_theta * delta_t) ** 2 / 3.0
    ])
    R_k = np.diag([measurement_noise_xy ** 2, measurement_noise_xy ** 2, measurement_noise_theta ** 2])
    
    H_k = np.eye(3)  # Full state observation
    x_init = np.array([[1000.0], [1000.0], [0.0]])  # Initial state [x, y, θ]
    P_init = np.diag([100000.0, 100000.0, 100.0])  # Initial covariance
    
    return A_k, B_k, Q_k, R_k, H_k, x_init, P_init


if __name__ == "__main__":
    # Choose dimensions
    dimensions = 1  # Change to 3 for 3D case
    
    if dimensions == 1:
        # ========== 1D CASE ==========
        sample_frequency = 10  # 10Hz
        A_k, B_k, Q_k, R_k, H_k, x_init, P_init = setup_kalman_1d(
            measurement_noise=15.0,
            process_noise=30.0,
            sampling_frequency=sample_frequency
        )
        
        # Example data from your table
        measurements = [997, 1005, 950, 910, 845, 789, 800]
        control_inputs = [0, 0, 500, 400, 500, 500, -250]
        
        # Run Kalman filter
        results = kalman_filter_batch(measurements, control_inputs,
                                       A_k, B_k, Q_k, R_k, H_k, x_init, P_init, dimensions)
        
        # Print results
        print("=" * 120)
        print("KALMAN FILTER - 1D CASE")
        print("=" * 120)
        print("Timestep | Measurement | Control Input | State Predict | P' Predict | Kalman Gain | State Estimate")
        print("-" * 120)
        for k in range(len(measurements)):
            print(f"{k:8d} | {measurements[k]:11.1f} | {control_inputs[k]:13.1f} | "
                  f"{results['state_predict'][k]:13.3f} | {results['covariance_predict'][k]:10.3f} | "
                  f"{results['kalman_gain'][k]:11.4f} | {results['state_estimate'][k]:15.3f}")
    
    elif dimensions == 3:
        # ========== 3D CASE ==========
        sample_frequency = 10  # 10Hz
        A_k, B_k, Q_k, R_k, H_k, x_init, P_init = setup_kalman_3d(
            measurement_noise_xy=15.0,
            measurement_noise_theta=0.5,
            process_noise_xy=30.0,
            process_noise_theta=0.1,
            sampling_frequency=sample_frequency
        )
        
        # Example 3D data (x, y, θ)
        measurements = [
            [1000, 2000, 0.0],
            [1050, 2100, 0.1],
            [1100, 2200, 0.2],
            [1150, 2300, 0.3]
        ]
        control_inputs = [
            [50, 100, 0.1],
            [50, 100, 0.1],
            [50, 100, 0.1],
            [50, 100, 0.1]
        ]
        
        # Run Kalman filter
        results = kalman_filter_batch(measurements, control_inputs,
                                       A_k, B_k, Q_k, R_k, H_k, x_init, P_init, dimensions)
        
        # Print results
        print("=" * 140)
        print("KALMAN FILTER - 3D CASE (x, y, θ)")
        print("=" * 140)
        print("Timestep | Measurement (x,y,θ)        | State Predict (x,y,θ)      | State Estimate (x,y,θ)")
        print("-" * 140)
        for k in range(len(measurements)):
            meas = results['measurement'][k]
            pred = results['state_predict'][k]
            est = results['state_estimate'][k]
            print(f"{k:8d} | ({meas[0]:7.1f}, {meas[1]:7.1f}, {meas[2]:6.3f}) | "
                  f"({pred[0]:7.1f}, {pred[1]:7.1f}, {pred[2]:6.3f}) | "
                  f"({est[0]:7.1f}, {est[1]:7.1f}, {est[2]:6.3f})")
    else:
        print(f"Error: dimensions must be 1 or 3, got {dimensions}")
