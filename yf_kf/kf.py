import numpy as np
import math
import abc


class KalmanFilterBase(object):
    """
    this is the abstract kalman filter class that defines the common interface
    """
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __init__(self, debug=False):
        self.debug = debug
        self._X = None
        self._P = None
        self._Q = None
        self._R = None

    @abc.abstractmethod
    def predict(self, delta_t):
        """
        use the motion model to predict
        :param delta_t: float
        :return:
        """
        pass

    @abc.abstractmethod
    def correct(self, Z):
        """
        correct the state using measurement model
        :param Z:
        :return:
        """
        pass

    ##########################################################
    # methods implemented below are common for all sub-classes

    def is_init_state_set(self):
        return self._X is not None

    def set_init_state(self, init_state):
        """
        set the initial states
        :param init_state: numpy array
        :return:
        """
        self._X = init_state.copy()

    def set_init_covariance(self, init_covariance):
        """
        set the initial covariance
        :param init_covariance: numpy array
        :return:
        """
        self._P = init_covariance.copy()

    def set_process_noise(self, p_noise):
        """
        set the process noise for motion update
        :param p_noise:
        :return:
        """
        self._Q = p_noise.copy()

    def set_measurement_noise(self, m_noise):
        """
        set the measurement noise for measurement model
        :param m_noise:
        :return:
        """
        self._R = m_noise.copy()


class LinearKalmanFilter(KalmanFilterBase):
    """
    simple 2D linear Kalman filter.
    state is [x, y, vel_x, vel_y]
    measurement is [x, y]
    """

    def __init__(self, debug=False):
        super(LinearKalmanFilter, self).__init__(debug)
        self._A = None
        self._H = np.array([[1, 0, 0, 0],
                            [0, 1, 0, 0]], dtype=np.float64)

        # process noise covariance
        self._Q = np.array([[0.1, 0, 0, 0],
                            [0, 0.1, 0, 0],
                            [0, 0, 0.1, 0],
                            [0, 0, 0, 0.1]], dtype=np.float64)

        # measurement noise covariance
        self._R = np.array([[0.05, 0],
                            [0, 0.05]], dtype=np.float64)

        # initial state covariance
        self._P = np.array([[1.0, 0, 0, 0],
                            [0, 1.0, 0, 0],
                            [0, 0, 1.0, 0],
                            [0, 0, 0, 1.0]], dtype=np.float64)

    def predict(self, delta_t):
        assert (self._X is not None)

        self._A = np.array([[1, 0, delta_t, 0],
                            [0, 1, 0, delta_t],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]], dtype=np.float64)

        self._X = np.dot(self._A, self._X)
        self._P = self._A.dot(self._P).dot(self._A.T) + self._Q

        if self.debug:
            print("delta_t: {}".format(delta_t))

    def correct(self, Z):
        K = self._P.dot(self._H.T).dot(np.linalg.inv(self._H.dot(self._P).dot(self._H.T) + self._R))

        self._X = self._X + K.dot(Z - self._H.dot(self._X))

        self._P = (np.identity(4) - K.dot(self._H)).dot(self._P)

        if self.debug:
            print("P:\n {}".format(self._P))
            print("X:\n {}".format(self._X))
            print("K:\n {}".format(K))


class BicycleKalmanFilter(KalmanFilterBase):
    """
    kinematic bicyle EKF
    state is [x, y, yaw, vel, beta], beta is the angle of the current velocity of the center of mass with respect to the longitudinal axis of the car
    measurement is [x, y]
    """

    def __init__(self, cm2rear_len=2.0, enable_yaw_observation=False, debug=False):
        super(BicycleKalmanFilter, self).__init__(debug)

        np.set_printoptions(precision=5)
        self.debug = debug
        self.cm2rear_len = cm2rear_len  # distance from center of mass to rear wheel
        self._A = None
        if enable_yaw_observation:
            self._H = np.array([[1, 0, 0, 0, 0],
                                [0, 1, 0, 0, 0],
                                [0, 0, 1, 0, 0]], dtype=np.float64)
            self._R = np.array([[1, 0, 0],
                                [0, 1, 0],
                                [0, 0, 1]], dtype=np.float64)
        else:
            self._H = np.array([[1, 0, 0, 0, 0],
                                [0, 1, 0, 0, 0]], dtype=np.float64)
            self._R = np.array([[1.0, 0],
                                [0, 1.0]], dtype=np.float64)

        self._X = None

        self._Q = np.array([[1e-7, 0, 0, 0, 0],
                            [0, 1e-7, 0, 0, 0],
                            [0, 0, 1e-7, 0, 0],
                            [0, 0, 0, 1e-3, 0],
                            [0, 0, 0, 0, 1e-2]], dtype=np.float64)

        self._P = np.array([[1, 0, 0, 0, 0],
                            [0, 1, 0, 0, 0],
                            [0, 0, 1, 0, 0],
                            [0, 0, 0, 1, 0],
                            [0, 0, 0, 0, 1]], dtype=np.float64)

    def predict(self, delta_t):
        assert (self._X is not None)

        # state update
        self._X[0] = self._X[0] + self._X[3] * math.cos(self._X[2] + self._X[4]) * delta_t
        self._X[1] = self._X[1] + self._X[3] * math.sin(self._X[2] + self._X[4]) * delta_t
        self._X[2] = self._X[2] + self._X[3] / self.cm2rear_len * math.sin(self._X[4]) * delta_t
        # X[3] and X[4] are vel and beta. both are kept as constant.

        # compute Jacobian
        j13 = -delta_t * math.sin(self._X[2] + self._X[4]) * self._X[3]
        j14 = delta_t * math.cos(self._X[2] + self._X[4])
        j15 = -delta_t * math.sin(self._X[2] + self._X[4]) * self._X[3]

        j23 = delta_t * math.cos(self._X[2] + self._X[4]) * self._X[3]
        j24 = delta_t * math.sin(self._X[2] + self._X[4])
        j25 = delta_t * math.cos(self._X[2] + self._X[4]) * self._X[3]

        j34 = delta_t * math.sin(self._X[4]) / self.cm2rear_len
        j35 = delta_t * math.cos(self._X[4]) / self.cm2rear_len * self._X[3]

        F = np.array([[1, 0, j13, j14, j15],
                      [0, 1, j23, j24, j25],
                      [0, 0, 1, j34, j35],
                      [0, 0, 0, 1, 0],
                      [0, 0, 0, 0, 1]])

        # covariance update
        # self._P = np.matmul(np.matmul(F, self._P), F.T) + self._Q
        self._P = F.dot(self._P).dot(F.T) + self._Q

        if self.debug:
            print("delta_t: {}".format(delta_t))
            print("P prior:\n {}".format(self._P))
            print("X prior:\n {}".format(self._X))
            print('psi prior {} deg'.format(math.degrees(self._X[2])))
            print('beta prior {} deg'.format(math.degrees(self._X[4])))

    def correct(self, Z):
        inv_tmp = np.linalg.inv(np.matmul(np.matmul(self._H, self._P), self._H.T) + self._R)
        K = np.matmul(np.matmul(self._P, self._H.T), inv_tmp)

        self._X = self._X + np.matmul(K, Z - np.matmul(self._H, self._X))
        self._P = np.matmul(np.identity(5) - np.matmul(K, self._H), self._P)
        if self.debug:
            print("P:\n {}".format(self._P))
            print("X:\n {}".format(self._X))
            print("K:\n {}".format(K))
            print('psi {} deg'.format(math.degrees(self._X[2])))
            print('beta {} deg'.format(math.degrees(self._X[4])))
