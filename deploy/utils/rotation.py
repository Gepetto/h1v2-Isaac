import math

import numpy as np


class Quaternion:
    """
    A quaternion class (w, x, y, z) for rotations, using NumPy arrays internally.
    """

    def __init__(self, w, x, y, z):
        self.coeffs = np.array([w, x, y, z], dtype=np.float64)

    @classmethod
    def from_coeffs(cls, coeffs):
        """Creates a Quaternion from a NumPy array [w, x, y, z]."""
        if len(coeffs) != 4:
            err_msg = "Quaternion coefficients must be a 4-element array."
            raise ValueError(err_msg)
        return cls(coeffs[0], coeffs[1], coeffs[2], coeffs[3])

    @classmethod
    def from_euler_z(cls, angle_rad):
        """
        Creates a quaternion from a Z-axis Euler angle (yaw).
        Quaternion (w, x, y, z) for rotation around Z by theta is:
        w = cos(theta/2)
        x = 0
        y = 0
        z = sin(theta/2)
        """
        half_angle = angle_rad / 2.0
        return cls(math.cos(half_angle), 0.0, 0.0, math.sin(half_angle))

    def as_rotation_matrix(self):
        """
        Converts the quaternion to a 3x3 rotation matrix using NumPy.
        Assumes normalized quaternion.
        """
        w, x, y, z = self.coeffs

        # Ensure it's a unit quaternion to avoid scaling issues
        norm_sq = np.dot(self.coeffs, self.coeffs)
        if abs(norm_sq - 1.0) > 1e-6:  # Check if close to 1.0
            norm = np.sqrt(norm_sq)
            w /= norm
            x /= norm
            y /= norm
            z /= norm

        return np.array(
            [
                [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
                [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
                [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y],
            ],
            dtype=np.float64,
        )

    @classmethod
    def from_rotation_matrix(cls, r_matrix):
        """
        Converts a 3x3 rotation matrix (NumPy array) to a quaternion (w, x, y, z).
        Handles different cases for numerical stability.
        """
        trace = np.trace(r_matrix)

        if trace > 0:
            s = 2 * np.sqrt(trace + 1.0)
            w = 0.25 * s
            x = (r_matrix[2, 1] - r_matrix[1, 2]) / s
            y = (r_matrix[0, 2] - r_matrix[2, 0]) / s
            z = (r_matrix[1, 0] - r_matrix[0, 1]) / s
        elif (r_matrix[0, 0] > r_matrix[1, 1]) and (r_matrix[0, 0] > r_matrix[2, 2]):
            s = 2 * np.sqrt(1.0 + r_matrix[0, 0] - r_matrix[1, 1] - r_matrix[2, 2])
            w = (r_matrix[2, 1] - r_matrix[1, 2]) / s
            x = 0.25 * s
            y = (r_matrix[0, 1] + r_matrix[1, 0]) / s
            z = (r_matrix[0, 2] + r_matrix[2, 0]) / s
        elif r_matrix[1, 1] > r_matrix[2, 2]:
            s = 2 * np.sqrt(1.0 + r_matrix[1, 1] - r_matrix[0, 0] - r_matrix[2, 2])
            w = (r_matrix[0, 2] - r_matrix[2, 0]) / s
            x = (r_matrix[0, 1] + r_matrix[1, 0]) / s
            y = 0.25 * s
            z = (r_matrix[1, 2] + r_matrix[2, 1]) / s
        else:
            s = 2 * np.sqrt(1.0 + r_matrix[2, 2] - r_matrix[0, 0] - r_matrix[1, 1])
            w = (r_matrix[1, 0] - r_matrix[0, 1]) / s
            x = (r_matrix[0, 2] + r_matrix[2, 0]) / s
            y = (r_matrix[1, 2] + r_matrix[2, 1]) / s
            z = 0.25 * s

        return cls(w, x, y, z)

    def as_array_wxyz(self):
        """Returns quaternion coefficients as a NumPy array [w, x, y, z]."""
        return self.coeffs


def transform_imu_data(waist_yaw, waist_yaw_omega, imu_quat, imu_omega):
    """
    Transforms IMU data using NumPy for matrix and vector operations.

    Args:
        waist_yaw (float): Yaw angle of the waist in radians.
        waist_yaw_omega (float): Angular velocity of the waist around Z-axis.
        imu_quat (np.ndarray): IMU quaternion in [w, x, y, z] format.
        imu_omega (np.ndarray): IMU angular velocity vector [wx, wy, wz].

    Returns:
        tuple: A tuple containing:
            - np.ndarray: Transformed pelvis quaternion in [w, x, y, z] format.
            - np.ndarray: Transformed angular velocity vector [wx, wy, wz].
    """
    rz_waist = Quaternion.from_euler_z(waist_yaw).as_rotation_matrix()
    imu_quat_obj = Quaternion.from_coeffs(imu_quat)
    r_torso = imu_quat_obj.as_rotation_matrix()
    r_pelvis = r_torso @ rz_waist.T
    q_pelvis = Quaternion.from_rotation_matrix(r_pelvis).as_array_wxyz()

    w_transformed = rz_waist @ imu_omega
    waist_omega_vector = np.array([0, 0, waist_yaw_omega], dtype=np.float64)
    w_final = w_transformed - waist_omega_vector

    return q_pelvis, w_final
