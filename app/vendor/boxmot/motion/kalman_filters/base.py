import numpy as np
import scipy.linalg


class BaseKalmanFilter:
    """
    Base class for Kalman filters tracking bounding boxes in image space.
    """

    def __init__(self, ndim: int):
        # Constant-velocity model over [position..., velocity...].
        self._motion_mat = np.eye(2 * ndim)
        self._motion_mat[:ndim, ndim:] += np.eye(ndim)
        self._update_mat = np.eye(ndim, 2 * ndim)

        # Motion and observation uncertainty weights.
        self._std_weight_position = 1.0 / 20
        self._std_weight_velocity = 1.0 / 160

    @staticmethod
    def _enforce_state_geometry(
        mean: np.ndarray,
        *,
        positive_indices: tuple[int, ...],
        min_size: float = 1e-4,
    ) -> np.ndarray:
        """Clamp the geometry dimensions of a state vector positive."""
        if mean.ndim == 1:
            for idx in positive_indices:
                mean[idx] = max(float(mean[idx]), min_size)
            return mean

        for idx in positive_indices:
            mean[idx, :] = np.maximum(mean[idx, :], min_size)
        return mean

    def initiate(self, measurement: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Create track from unassociated measurement.
        """
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        std = self._get_initial_covariance_std(measurement)
        covariance = np.diag(np.square(std))
        return mean, covariance

    def _get_initial_covariance_std(self, measurement: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def _get_measurement_noise_std(self, mean: np.ndarray, confidence: float) -> np.ndarray:
        raise NotImplementedError

    def _get_multi_process_noise_std(self, mean: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def project(
        self, mean: np.ndarray, covariance: np.ndarray, confidence: float = 0.0
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Project state distribution to measurement space.
        """
        std = self._get_measurement_noise_std(mean, confidence)

        # NSA Kalman (GIAOTracker): scale the preset measurement noise by
        # (1 - confidence), so a confident detection is trusted more.
        std = [(1 - confidence) * x for x in std]

        innovation_cov = np.diag(np.square(std))

        mean = np.dot(self._update_mat, mean)
        covariance = np.linalg.multi_dot((self._update_mat, covariance, self._update_mat.T))
        return mean, covariance + innovation_cov

    def multi_predict(
        self, mean: np.ndarray, covariance: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Run Kalman filter prediction step (Vectorized version).
        """
        std_pos, std_vel = self._get_multi_process_noise_std(mean)
        sqr = np.square(np.r_[std_pos, std_vel]).T

        motion_cov = [np.diag(sqr[i]) for i in range(len(mean))]
        motion_cov = np.asarray(motion_cov)

        mean = np.dot(mean, self._motion_mat.T)
        left = np.dot(self._motion_mat, covariance).transpose((1, 0, 2))
        covariance = np.dot(left, self._motion_mat.T) + motion_cov

        return mean, covariance

    def update(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
        measurement: np.ndarray,
        confidence: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Run Kalman filter correction step.
        """
        projected_mean, projected_cov = self.project(mean, covariance, confidence)

        chol_factor, lower = scipy.linalg.cho_factor(projected_cov, lower=True, check_finite=False)
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower),
            np.dot(covariance, self._update_mat.T).T,
            check_finite=False,
        ).T
        innovation = measurement - projected_mean

        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_covariance = covariance - np.linalg.multi_dot(
            (kalman_gain, projected_cov, kalman_gain.T)
        )
        return new_mean, new_covariance
