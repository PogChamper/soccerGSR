import numpy as np

from app.vendor.boxmot.motion.kalman_filters.base import BaseKalmanFilter


class KalmanFilterXYWH(BaseKalmanFilter):
    """Kalman filter for the [x, y, w, h] bounding-box state."""

    def __init__(self) -> None:
        super().__init__(ndim=4)

    def _get_initial_covariance_std(self, measurement: np.ndarray) -> np.ndarray:
        return [
            2 * self._std_weight_position * measurement[2],
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[2],
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[2],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[2],
            10 * self._std_weight_velocity * measurement[3],
        ]

    def _get_measurement_noise_std(self, mean: np.ndarray, confidence: float) -> np.ndarray:
        return [
            self._std_weight_position * mean[2],
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[2],
            self._std_weight_position * mean[3],
        ]

    def _get_multi_process_noise_std(self, mean: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        std_pos = [
            self._std_weight_position * mean[:, 2],
            self._std_weight_position * mean[:, 3],
            self._std_weight_position * mean[:, 2],
            self._std_weight_position * mean[:, 3],
        ]
        std_vel = [
            self._std_weight_velocity * mean[:, 2],
            self._std_weight_velocity * mean[:, 3],
            self._std_weight_velocity * mean[:, 2],
            self._std_weight_velocity * mean[:, 3],
        ]
        return std_pos, std_vel

    def initiate(self, measurement: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        measurement = np.asarray(measurement, dtype=float).copy()
        mean, covariance = super().initiate(measurement)
        return self._enforce_state_geometry(mean, positive_indices=(2, 3)), covariance

    def multi_predict(
        self, mean: np.ndarray, covariance: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        mean, covariance = super().multi_predict(mean, covariance)
        mean[:, 2] = np.maximum(mean[:, 2], 1e-4)
        mean[:, 3] = np.maximum(mean[:, 3], 1e-4)
        return mean, covariance

    def update(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
        measurement: np.ndarray,
        confidence: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        new_mean, new_covariance = super().update(mean, covariance, measurement, confidence)
        return self._enforce_state_geometry(new_mean, positive_indices=(2, 3)), new_covariance
