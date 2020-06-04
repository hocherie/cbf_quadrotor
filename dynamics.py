"""dynamics.py
Simulate Simple Quadrotor Dynamics

`python dynamics.py` to see hovering drone
"""

import numpy as np
import numpy.matlib
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from visualize_dynamics import *
from sim_utils import *
from controller import *
import time

# Physical constants
g = -9.81  # FLU
m = 0.5
L = 0.25
k = 3e-6
b = 1e-7
I = np.diag([5e-3, 5e-3, 10e-3])
kd = 0.001
dt = 0.1
maxrpm = 10000
maxthrust = k*np.sum(np.array([maxrpm**2] * 4))
param_dict = {"g": g, "m": m, "L": L, "k": k, "b": b, "I": I,
              "kd": kd, "dt": dt, "maxRPM": maxrpm, "maxthrust": maxthrust}


def init_state():
    """Initialize state dictionary. """
    state = {"x": np.array([5, 0, 10]),
             "xdot": np.zeros(3,),
             "xdd": np.zeros(3,),
             "theta": np.radians(np.array([0, 0, 0])),  # ! hardcoded
             "thetadot": np.radians(np.array([0, 0, 0]))  # ! hardcoded
             }
    return state


class QuadDynamics:
    def __init__(self):
        self.param_dict = param_dict

    def step_dynamics(self, state, u):
        """Step dynamics given current state and input. Updates state dict.
        
        Parameters
        ----------
        state : dict 
            contains current x, xdot, theta, thetadot

        u : (4, ) np.ndarray
            control input - (angular velocity)^squared of motors (rad^2/s^2)

        Updates
        -------
        state : dict 
            updates with next x, xdot, xdd, theta, thetadot  
        """
        # Compute angular velocity vector from angular velocities
        omega = self.thetadot2omega(state["thetadot"], state["theta"])

        # Compute linear and angular accelerations given input and state
        a = self.calc_acc(u, state["theta"], state["xdot"], m, g, k, kd)
        omegadot = self.calc_ang_acc(u, omega, I, L, b, k)

        # Compute next state
        omega = omega + dt * omegadot
        thetadot = self.omega2thetadot(omega, state["theta"])
        theta = state["theta"] + dt * state["thetadot"]
        xdot = state["xdot"] + dt * a
        x = state["x"] + dt * xdot

        # Update state dictionary
        state["x"] = x
        state["xdot"] = xdot
        state["xdd"] = a
        state["theta"] = theta
        state["thetadot"] = thetadot

        return state

    def compute_thrust(self, u, k):
        """Compute total thrust (in body frame) given control input and thrust coefficient. Used in calc_acc().
        Clips if above maximum rpm (10000).

        thrust = k * sum(u)
        
        Parameters
        ----------
        u : (4, ) np.ndarray
            control input - (angular velocity)^squared of motors (rad^2/s^2)

        k : float
            thrust coefficient

        Returns
        -------
        T : (3, ) np.ndarray
            thrust in body frame
        """

        u = np.clip(u, 0, self.param_dict["maxRPM"]**2)
        T = np.array([0, 0, k*np.sum(u)])
        # print("u", u)
        # print("T", T)

        return T

    def calc_torque(self, u, L, b, k):
        """Compute torque (body-frame), given control input, and coefficients. Used in calc_ang_acc()
        
        Parameters
        ----------
        u : (4, ) np.ndarray
            control input - (angular velocity)^squared of motors (rad^2/s^2)
        L : float
            distance from center of quadcopter to any propellers, to find torque (m).
        
        b : float # TODO: description
        
        k : float
            thrust coefficient

        Returns
        -------
        tau : (3,) np.ndarray
            torque in body frame (Nm)

        """
        tau = np.array([
            L * k * (u[0]-u[2]),
            L * k * (u[1]-u[3]),
            b * (u[0]-u[1] + u[2]-u[3])
        ])

        return tau

    def calc_acc(self, u, theta, xdot, m, g, k, kd):
        """Computes linear acceleration (in inertial frame) given control input, gravity, thrust and drag.
        a = g + T_b+Fd/m

        Parameters
        ----------
        u : (4, ) np.ndarray
            control input - (angular velocity)^squared of motors (rad^2/s^2)
        theta : (3, ) np.ndarray 
            rpy angle in body frame (radian) 
        xdot : (3, ) np.ndarray
            linear velocity in body frame (m/s), for drag calc 
        m : float
            mass of quadrotor (kg)
        g : float
            gravitational acceleration (m/s^2)
        k : float
            thrust coefficient
        kd : float
            drag coefficient

        Returns
        -------
        a : (3, ) np.ndarray 
            linear acceleration in inertial frame (m/s^2)
        """
        gravity = np.array([0, 0, g])
        R = get_rot_matrix(theta)
        thrust = self.compute_thrust(u, k)
        T = np.dot(R, thrust)
        Fd = -kd * xdot
        a = gravity + 1/m * T + Fd
        return a

    def calc_ang_acc(self, u, omega, I, L, b, k):
        """Computes angular acceleration (in body frame) given control input, angular velocity vector, inertial matrix.
        
        omegaddot = inv(I) * (torque - w x (Iw))

        Parameters
        ----------
        u : (4, ) np.ndarray
            control input - (angular velocity)^squared of motors (rad^2/s^2)
        omega : (3, ) np.ndarray 
            angular velcoity vector in body frame
        I : (3, 3) np.ndarray 
            inertia matrix
        L : float
            distance from center of quadcopter to any propellers, to find torque (m).
        b : float # TODO: description
        k : float
            thrust coefficient


        Returns
        -------
        omegaddot : (3, ) np.ndarray
            rotational acceleration in body frame #TODO: units
        """
        # Calculate torque given control input and physical constants
        tau = self.calc_torque(u, L, b, k)

        # Calculate body frame angular acceleration using Euler's equation
        omegaddot = np.dot(np.linalg.inv(
            I), (tau - np.cross(omega, np.dot(I, omega))))

        return omegaddot

    def omega2thetadot(self, omega, theta):
        """Compute angle rate from angular velocity vector and euler angle.

        Uses Tait Bryan's z-y-x/yaw-pitch-roll.

        Parameters
        ----------

        omega: (3, ) np.ndarray
            angular velocity vector

        theta: (3, ) np.ndarray
            euler angles in body frame (roll, pitch, yaw)

        Returns
        ---------
        thetadot: (3, ) np.ndarray
            time derivative of euler angles (roll rate, pitch rate, yaw rate)
        """
        mult_matrix = np.array(
            [
                [1, 0, -np.sin(theta[1])],
                [0, np.cos(theta[0]), np.cos(theta[1])*np.sin(theta[0])],
                [0, -np.sin(theta[0]), np.cos(theta[1])*np.cos(theta[0])]
            ], dtype='float')

        mult_inv = np.linalg.inv(mult_matrix)
        thetadot = np.dot(mult_inv, omega)

        return thetadot

    def thetadot2omega(self, thetadot, theta):
        """Compute angular velocity vector from euler angle and associated rates.
        
        Uses Tait Bryan's z-y-x/yaw-pitch-roll. 

        Parameters
        ----------
        
        thetadot: (3, ) np.ndarray
            time derivative of euler angles (roll rate, pitch rate, yaw rate)

        theta: (3, ) np.ndarray
            euler angles in body frame (roll, pitch, yaw)

        Returns
        ---------
        w: (3, ) np.ndarray
            angular velocity vector (in body frame)
        
        """
        roll = theta[0]
        pitch = theta[1]
        yaw = theta[2]

        mult_matrix = np.array(
            [
                [1, 0, -np.sin(pitch)],
                [0, np.cos(roll), np.cos(pitch)*np.sin(roll)],
                [0, -np.sin(roll), np.cos(pitch)*np.cos(roll)]
            ]

        )

        w = np.dot(mult_matrix, thetadot)

        return w


def basic_input():
    """Return arbritrary input to test simulator"""
    return np.power(np.array([950, 700, 700, 700]), 2)


class QuadHistory():
    """Keeps track of quadrotor history for plotting."""

    def __init__(self):
        self.hist_theta = []
        self.hist_des_theta = []
        self.hist_thetadot = []
        self.hist_xdot = [[0, 0, 0]]
        self.hist_xdotdot = []
        self.hist_x = []
        self.hist_y = []
        self.hist_z = []
        self.hist_pos = []
        self.hist_des_xdot = []
        self.hist_des_x = []

    def update_history(self, state, des_theta_deg_i, des_xdot_i, des_x_i, dt):
        """Appends current state and desired theta for plotting."""
        x = state["x"]
        xdot = state["xdot"]
        xdotdot = (xdot - np.array(self.hist_xdot[-1])) / dt
        self.hist_x.append(x[0])
        self.hist_y.append(x[1])
        self.hist_z.append(x[2])
        self.hist_theta.append(np.degrees(state["theta"]))
        self.hist_thetadot.append(np.degrees(state["thetadot"]))
        # if des_xdot_i is None:
        #     des_xdot_i = [0,0,0]
        # if des_x_i is None:
        #     des_x_i = [0, 0, 0]
        self.hist_des_theta.append(des_theta_deg_i)
        self.hist_xdot.append(state["xdot"])
        self.hist_des_xdot.append(des_xdot_i)
        self.hist_des_x.append(des_x_i)
        self.hist_pos.append(x)

        self.hist_xdotdot.append(state["xdd"])


def main():
    print("start")
    t_start = time.time()

    # Set desired position
    des_pos = np.array([3, -3, 9])

    # Initialize Robot State
    state = init_state()

    # Initialize quadrotor history tracker
    quad_hist = QuadHistory()

    # Initialize visualization
    fig = plt.figure()
    ax = fig.add_subplot(2, 3, 1, projection='3d')
    ax_x_error = fig.add_subplot(2, 3, 2)
    ax_xd_error = fig.add_subplot(2, 3, 3)
    ax_xdd_error = fig.add_subplot(2, 3, 4)
    ax_th_error = fig.add_subplot(2, 3, 5)
    ax_thr_error = fig.add_subplot(2, 3, 6)

    # Initialize controller errors
    integral_p_err = None
    integral_v_err = None

    # Initialize quad dynamics
    quad_dyn = QuadDynamics()

    sim_iter = 100
    # Step through simulation
    for t in range(sim_iter):

        if t * dt > 20:
            des_pos = np.array([0, 0, 10])
        ax.cla()
        des_vel, integral_p_err = pi_position_control(
            state, des_pos, integral_p_err)
        des_thrust, des_theta, integral_v_err = pi_velocity_control(
            state, des_vel, integral_v_err)  # attitude control
        des_theta_deg = np.degrees(des_theta)  # for logging
        u = pi_attitude_control(
            state, des_theta, des_thrust, param_dict)  # attitude control
        # Step dynamcis and update state dict
        state = quad_dyn.step_dynamics(state, u)
        # update history for plotting
        quad_hist.update_history(state, des_theta_deg, des_vel, des_pos, dt)

    for t in range(100):
    # # Visualize quadrotor and angle error
        ax.cla()
        visualize_quad_quadhist(ax, quad_hist, t)
        visualize_error_quadhist(
            ax_x_error, ax_xd_error, ax_th_error, ax_thr_error, ax_xdd_error, quad_hist, t, dt)

    print("Time Elapsed:", time.time() - t_start)


if __name__ == '__main__':
    main()
