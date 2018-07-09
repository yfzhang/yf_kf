#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Kalman filter implementation.
    1. Bicycle model
    2. Unicycle model
"""

import numpy as np
import math


class Filter(object):
    def __init__(self, debug=False):
        self.debug = debug
        self._X = None
        self._P = None
        self._Q = None
        self._R = None

    def set_init_state(self, init_state):
        self._X = init_state

    def set_init_covariance(self, init_covariance):
        self._P = init_covariance

    def set_process_noise(self, noise):
        self._Q = noise

    def set_measurement_noise(self, noise):
        self._R = noise

    def is_init_state_set(self):
        return self._X is not None

    @property
    def X(self):
        return self._X

    @property
    def P(self):
        return self._P


class LinearKalmanFilter(Filter):
    """
    simple linear Kalman filter.
    state is [x, y, vel_x, vel_y]
    measurement is [x, y]
    """

    def __init__(self, debug=False):
        super(LinearKalmanFilter, self).__init__(debug)
        self.A = None
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])

        # process noise covariance
        self._Q = np.array([[0.1, 0, 0, 0], [0, 0.1, 0, 0], [0, 0, 0.1, 0], [0, 0, 0, 0.1]])

        # measurement noise covariance
        self._R = np.array([[0.05, 0],
                            [0, 0.05]])

        # initial state covariance
        self._P = np.array([[1.0, 0, 0, 0],
                            [0, 1.0, 0, 0],
                            [0, 0, 1.0, 0],
                            [0, 0, 0, 1.0]])

    def predict(self, delta_t):
        assert (self._X is not None)

        self.A = np.array([[1, 0, delta_t, 0],
                           [0, 1, 0, delta_t],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])

        self._X = np.dot(self.A, self._X)
        self._P = self.A.dot(self._P).dot(self.A.T) + self._Q

        if self.debug:
            print("delta_t: {}".format(delta_t))

    def correct(self, Z):
        K = self._P.dot(self.H.T).dot(np.linalg.inv(self.H.dot(self._P).dot(self.H.T) + self._R))

        self._X = self._X + K.dot(Z - self.H.dot(self._X))

        self._P = (np.identity(4) - K.dot(self.H)).dot(self._P)

        if self.debug:
            print("P:\n {}".format(self._P))
            print("X:\n {}".format(self._X))
            print("K:\n {}".format(K))


class BicycleKalmanFilter(Filter):
    """
    bicyle EKF
    state is [x, y, yaw, vel, beta], beta is the angle of the current velocity of the center of mass with respect to the longitudinal axis of the car
    measurement is [x, y]
    """

    def __init__(self, cm2rear_len=2.0, debug=False):
        super(BicycleKalmanFilter, self).__init__(debug)

        np.set_printoptions(precision=5)
        self.debug = debug
        self.cm2rear_len = cm2rear_len  # distance from center of mass to rear wheel
        self.A = None
        self.H = np.array([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0]])
        self._X = None

        # TODO: the non-diagonal element for Q may not be zero
        self._Q = np.array([[1e-7, 0, 0, 0, 0],
                            [0, 1e-7, 0, 0, 0],
                            [0, 0, 1e-7, 0, 0],
                            [0, 0, 0, 1e-3, 0],
                            [0, 0, 0, 0, 1e-2]])

        # self.R = np.array([[0.5, 0], [0, 0.5]])
        # self.R = np.array([[5.0, 0], [0, 5.0]])
        self._R = np.array([[100.0, 0],
                            [0, 100.0]])

        # TODO: the non-diagonal element for P may not be zero
        self._P = np.array([[10.0, 0, 0, 0, 0],
                            [0, 10.0, 0, 0, 0],
                            [0, 0, 10.0, 0, 0],
                            [0, 0, 0, 10.0, 0],
                            [0, 0, 0, 0, 10.0]])
        self.K = None

    def predict(self, delta_t):
        assert (self._X is not None)

        # state update
        # X[3] and X[4] are vel and beta. both are kept as constant.
        self._X[0] = self._X[0] + self._X[3] * math.cos(self._X[2] + self._X[4]) * delta_t
        self._X[1] = self._X[1] + self._X[3] * math.sin(self._X[2] + self._X[4]) * delta_t
        self._X[2] = self._X[2] + self._X[3] / self.cm2rear_len * math.sin(self._X[4]) * delta_t

        # compute Jacobian
        j13 = -delta_t * math.sin(self._X[2] + self._X[4]) * self._X[3]
        j14 = delta_t * math.cos(self.X[2] + self.X[4])
        j15 = -delta_t * math.sin(self.X[2] + self.X[4]) * self.X[3]

        j23 = delta_t * math.cos(self.X[2] + self.X[4]) * self.X[3]
        j24 = delta_t * math.sin(self.X[2] + self.X[4])
        j25 = delta_t * math.cos(self.X[2] + self.X[4]) * self.X[3]

        j34 = delta_t * math.sin(self.X[4]) / self.cm2rear_len
        j35 = delta_t * math.cos(self.X[4]) / self.cm2rear_len * self.X[3]

        jocobian = np.array([[1, 0, j13, j14, j15],
                             [0, 1, j23, j24, j25],
                             [0, 0, 1, j34, j35],
                             [0, 0, 0, 1, 0],
                             [0, 0, 0, 0, 1]])
        # covariance update
        self._P = np.matmul(np.matmul(jocobian, self._P), jocobian.T) + self._Q

        if self.debug:
            print("delta_t: {}".format(delta_t))
            print("P prior:\n {}".format(self._P))
            print("X prior:\n {}".format(self.X))
            print('psi prior {} deg'.format(math.degrees(self.X[2])))
            print('beta prior {} deg'.format(math.degrees(self.X[4])))

    def correct(self, Z):
        inv_tmp = np.linalg.inv(np.matmul(np.matmul(self.H, self._P), self.H.T) + self._R)
        self.K = np.matmul(np.matmul(self._P, self.H.T), inv_tmp)

        self._X = self._X + np.matmul(self.K, Z - np.matmul(self.H, self._X))
        self._P = np.matmul(np.identity(5) - np.matmul(self.K, self.H), self._P)
        if self.debug:
            print("P:\n {}".format(self._P))
            print("X:\n {}".format(self._X))
            print("K:\n {}".format(self.K))
            print('psi {} deg'.format(math.degrees(self._X[2])))
            print('beta {} deg'.format(math.degrees(self._X[4])))


class UnicycleKalmanFilter:
    """TODO: WIP, still have bugs"""

    def __init__(self, debug=False):
        raise NotImplementedError("not implemented yet")
        self.debug = debug
        self.A = None
        self.H = np.array([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0]])
        self.X = np.array([0, 0, 0, 0, 0]).T
        self.X_prior = None

        # TODO: the non-diagonal element for Q may not be zero
        self.Q = np.array(
            [[0.1, 0, 0, 0, 0], [0, 0.1, 0, 0, 0], [0, 0, 0.1, 0, 0], [0, 0, 0, 0.1, 0], [0, 0, 0, 0, 0.5]])

        self.R = np.array([[0.001, 0], [0, 0.001]])

        # TODO: the non-diagonal element for P may not be zero
        self.P = np.array(
            [[1.0, 0, 0, 0, 0], [0, 1.0, 0, 0, 0], [0, 0, 1.0, 0, 0], [0, 0, 0, 1.0, 0], [0, 0, 0, 0, 1.0]])
        self.P_prior = None
        self.K = None

    def predict(self, delta_t):
        self.A = np.array([[1, 0, delta_t * -self.X[3] * math.sin(self.X[2]), delta_t * math.cos(self.X[2]), 0],
                           [0, 1, delta_t * self.X[3] * math.cos(self.X[2]), delta_t * math.sin(self.X[2]), 0],
                           [0, 0, 1, 0, delta_t],
                           [0, 0, 0, 1, 0],
                           [0, 0, 0, 0, 1]])
        self.X_prior = np.dot(self.A, self.X)
        self.P_prior = self.A.dot(self.P).dot(self.A.T) + self.Q
        if self.debug:
            print("delta_t: {}".format(delta_t))

    def correct(self, Z):
        self.K = self.P_prior.dot(self.H.T).dot(np.linalg.inv(self.H.dot(self.P_prior).dot(self.H.T) + self.R))
        self.X = self.X_prior + self.K.dot(Z - self.H.dot(self.X_prior))
        self.P = (np.identity(5) - self.K.dot(self.H)).dot(self.P_prior)
        if self.debug:
            print("P:\n {}".format(self.P))
            print("X:\n {}".format(self.X))
            print("K:\n {}".format(self.K))
