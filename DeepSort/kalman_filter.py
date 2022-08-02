# vim: expandtab:ts=4:sw=4
import numpy as np
from rsa import PrivateKey
import scipy.linalg



"""
Table for the 0.95 quantile of the chi-square distribution with N degrees of
freedom (contains values for N=1, ..., 9). Taken from MATLAB/Octave's chi2inv
function and used as Mahalanobis gating threshold.
"""
chi2inv95 = {
    1: 3.8415,
    2: 5.9915,
    3: 7.8147,
    4: 9.4877,
    5: 11.070,
    6: 12.592,
    7: 14.067,
    8: 15.507,
    9: 16.919}


class KalmanFilter(object):
    """
    A simple Kalman filter for tracking bounding boxes in image space.
    The 8-dimensional state space
        x, y, a, h, vx, vy, va, vh
    contains the bounding box center position (x, y), aspect ratio a, height h,
    and their respective velocities.
    Object motion follows a constant velocity model. The bounding box location
    (x, y, a, h) is taken as direct observation of the state space (linear
    observation model).
    """

    def __init__(self) -> None:
        # dimension of observation of the bounding box
        ndim, dt = 4, 1.

        # Create Kalman filter model matrices.
        # x_k+1 = F*x_k + v
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        
        
        # matrix to project x to measurement space
        # z_k = H*x_k + w
        self._update_mat = np.eye(ndim, 2 * ndim)


        # Motion and observation uncertainty are chosen relative to the current
        # state estimate. These weights control the amount of uncertainty in
        # the model. This is a bit hacky.
        self._std_weight_position = 1./20
        self._std_weight_velocity = 1./160

    def initiate(self, measurement):
        """Create track from unassociated measurement.
        Parameters
        ----------
        measurement : ndarray
            Bounding box coordinates (x, y, a, h) with center position (x, y),
            aspect ratio a, and height h.
        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector (8 dimensional) and covariance matrix (8x8
            dimensional) of the new track. Unobserved velocities are initialized
            to 0 mean.
        """

        # the object's position is equal to the position at the time of initialization
        mean_pos = measurement
        # assume the object's velocity is zeri 
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        # Create covariance matrix is a diagon matrix
        # it means no correlation between attribute (x, y, ...) at the initialization stage

        # here is standard deviation of each attribute 
        #   x, y, a, h, vx, vy, va, vh
        # value is big to present an uncertainty about the unobservable initial velocities
        std = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            1e-2, 
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[3],  
            10 * self._std_weight_velocity * measurement[3],
            1e-5,
            10 * self._std_weight_velocity * measurement[3]]
        
        # create covariance matrix by square(standard deviation)
        covariance = np.diag(np.square(std))

        return mean, covariance

    def predict(self, mean, covariance):
        """Run Kalman filter prediction step.
        Parameters
        ----------
        mean : ndarray
            The 8 dimensional mean vector of the object state at the previous
            time step.
        covariance : ndarray
            The 8x8 dimensional covariance matrix of the object state at the
            previous time step.
        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector and covariance matrix of the predicted
            state. Unobserved velocities are initialized to 0 mean.

        """

        # standard deviation of noisy 
        # x_k+1 = F*x_k + v => we calculate cov(v)
        std_pos = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-2,
            self._std_weight_position * mean[3]]
        std_vel = [
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[3],
            1e-5,
            self._std_weight_velocity * mean[3]]

        # concate std_pos and std_vel
        # element-wise square (std)^2 = variace
        # make a 2D diagon matrix from 1D matrix
        noisy_cov = np.diag(np.square(np.r_[std_pos, std_vel]))

        # E(x_k+1) = F*E(x_k)
        mean = np.matmul(self._motion_mat, mean)
        # Cov(x_k+1) = F*Cov(x_k)F.T + Cov(v)
        covariance = np.matmul(np.matmul(self._motion_mat, covariance), self._motion_mat.T) + noisy_cov

        return mean, covariance

    def project(self, mean, covariance):
        """Project state distribution to measurement space.
        Parameters
        ----------
        mean : ndarray
            The state's mean vector (8 dimensional array).
        covariance : ndarray
            The state's covariance matrix (8x8 dimensional).
        Returns
        -------
        (ndarray, ndarray)
            Returns the projected mean and covariance matrix of the given state
            estimate.
        """ 
        # standard deviation of noisy 
        # z_k = H*x_k + w => we calculate Cov(w)
        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3]]

        noisy_cov = np.diag(np.square(std))

        # E(z) = H*E(x)
        mean = np.matmul(self._update_mat, mean)

        # Cov(z) = H*Cov(x)*H.T + Cov(w)
        covariance = np.matmul(np.matmul(self._update_mat, covariance), self._update_mat.T) + noisy_cov

        return mean, covariance

    def update(self, mean, covariance, measurement):
        """Run Kalman filter correction step.
        Parameters
        ----------
        mean : ndarray
            The predicted state's mean vector (8 dimensional).
        covariance : ndarray
            The state's covariance matrix (8x8 dimensional).
        measurement : ndarray
            The 4 dimensional measurement vector (x, y, a, h), where (x, y)
            is the center position, a the aspect ratio, and h the height of the
            bounding box.
        Returns
        -------
        (ndarray, ndarray)
            Returns the measurement-corrected state distribution.
        """
        projected_mean, projected_cov = self.project(mean, covariance)

        # Cov(x_k): covariance
        # Cov(z_k): projected_cov
        # H: self._update_mat
        # K: kalman_gain
        # K = Cov(x_k)*H.T(Cov(w) + H*Cov(x_k)*H.T)^-1
        # K*(Cov(w) + H*Cov(x_k)*H.T) = Cov(x_k)*H.T
        # (Cov(w) + H*Cov(x_k)*H.T).T*K.T = (Cov(x_k)*H.T).T
        # Cov(z)*K.T = (Cov(x_k)*H.T).T
        # A*x = B, x: K

        chol_factor, lower = scipy.linalg.cho_factor(
            projected_cov, lower=True, check_finite=False)
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower), np.matmul(covariance, self._update_mat.T).T,
            check_finite=False).T
        innovation = measurement - projected_mean

        # x_k+1 = x_k + K(z - H*x_k)
        new_mean = mean + np.matmul(innovation, kalman_gain.T)
        # Cov(x+1) = Cov(x)*(I - K*H)
        # Cov(x)*K*H = K*Cov(z)*K.T
        new_covariance = covariance - np.linalg.multi_dot((
            kalman_gain, projected_cov, kalman_gain.T))
        return new_mean, new_covariance

    def gating_distance(self, mean, covariance, measurements,
                        only_position=False):
        """Compute gating distance between state distribution and measurements.
        A suitable distance threshold can be obtained from `chi2inv95`. If
        `only_position` is False, the chi-square distribution has 4 degrees of
        freedom, otherwise 2.
        Parameters
        ----------
        mean : ndarray
            Mean vector over the state distribution (8 dimensional).
        covariance : ndarray
            Covariance of the state distribution (8x8 dimensional).
        measurements : ndarray
            An Nx4 dimensional matrix of N measurements, each in
            format (x, y, a, h) where (x, y) is the bounding box center
            position, a the aspect ratio, and h the height.
        only_position : Optional[bool]
            If True, distance computation is done with respect to the bounding
            box center position only.
        Returns
        -------
        ndarray
            Returns an array of length N, where the i-th element contains the
            squared Mahalanobis distance between (mean, covariance) and
            `measurements[i]`.
        """
        mean, covariance = self.project(mean, covariance)

        if only_position:
            mean, covariance = mean[:2], covariance[:2]
            measurements = measurements[:, :2]

        # d = measurement - mean
        # maha = d*Cov(x)^-1*d.T
        # Cov(x)*d^-1*maha = d

        cholesky_factor = np.linalg.cholesky(covariance)
        d = measurements - mean
        z = scipy.linalg.solve_triangular(
            cholesky_factor, d.T, lower=True, check_finite=False,
            overwrite_b=True)
        squared_maha = np.sum(z * z, axis=0)
        return squared_maha
