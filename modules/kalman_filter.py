Add Kalman filter module for player position smoothing

import numpy as np

class KalmanFilter:
    def __init__(self, dt=1, process_noise=1e-3, measurement_noise=1e-1):
        self.x = np.zeros((4, 1))
        self.F = np.array([[1, 0, dt, 0],
                           [0, 1, 0, dt],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]])
        self.P = np.eye(4) * 500
        self.Q = np.eye(4) * process_noise
        self.R = np.eye(2) * measurement_noise

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z):
        z = np.reshape(z, (2, 1))
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P

    def get_state(self):
        return self.x[:2].flatten()
