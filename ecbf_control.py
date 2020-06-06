import numpy as np
import matplotlib.pyplot as plt
from cvxopt import matrix
from cvxopt import solvers

class ECBFControl():
    def __init__(self, state, goal=np.array([[0], [10]]), Kp=3, Kd=4):
        self.state = state
        self.shape_a = 1  # Ellipsoid ECBF Shape
        self.shape_b = 2
        self.shape_safedist = 2
        self.K = np.array([Kp, Kd])  # ECBF Gain
        self.goal = goal
        self.use_safe = True

    @np.vectorize
    def h_func(r1, r2, a, b, safety_dist):
        # Ellipsoid ECBF Function
        hr = np.power(r1, 4)/np.power(a, 4) + \
            np.power(r2, 4)/np.power(b, 4) - safety_dist
        return hr

    def compute_plot_z(self, obs):
        plot_x = np.arange(-7.5, 7.5, 0.1)
        plot_y = np.arange(-7.5, 7.5, 0.1)
        xx, yy = np.meshgrid(plot_x, plot_y, sparse=True)
        z = self.h_func(xx - obs[0], yy - obs[1], self.shape_a,
                        self.shape_b, self.shape_safedist) > 0
        p = {"x": plot_x, "y": plot_y, "z": z}
        return p

    def plot_h(self, plot_x, plot_y, z):
        h = plt.contourf(plot_x, plot_y, z,
                         [-1, 0, 1], colors=["grey", "white"])
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.legend()
        plt.pause(0.00000001)

    def compute_h(self, obs=np.array([[0], [0]]).T):
        rel_r = np.atleast_2d(self.state["x"][:2]).T - obs
        hr = self.h_func(rel_r[0], rel_r[1], self.shape_a,
                         self.shape_b, self.shape_safedist)
        return hr

    def compute_hd(self, obs):
        rel_r = np.atleast_2d(self.state["x"][:2]).T - obs
        rd = np.atleast_2d(self.state["xdot"][:2]).T
        term1 = (4 * np.power(rel_r[0], 3) * rd[0])/(np.power(self.shape_a, 4))
        term2 = (4 * np.power(rel_r[1], 3) * rd[1])/(np.power(self.shape_b, 4))
        return term1+term2

    def compute_A(self, obs):
        rel_r = np.atleast_2d(self.state["x"][:2]).T - obs
        A0 = (4 * np.power(rel_r[0], 3))/(np.power(self.shape_a, 4))
        A1 = (4 * np.power(rel_r[1], 3))/(np.power(self.shape_b, 4))

        return np.array([np.hstack((A0, A1))])

    def compute_h_hd(self, obs):
        h = self.compute_h(obs)
        hd = self.compute_hd(obs)

        return np.vstack((h, hd)).astype(np.double)

    def compute_b(self, obs):
        """extra + K * [h hd]"""
        rel_r = np.atleast_2d(self.state["x"][:2]).T - obs
        rd = np.array(np.array(self.state["xdot"])[:2])
        extra = -(
            (12 * np.square(rel_r[0]) * np.square(rd[0]))/np.power(self.shape_a, 4) +
            (12 * np.square(rel_r[1]) * np.square(rd[1])) /
            np.power(self.shape_b, 4)
        )

        b_ineq = extra - self.K @ self.compute_h_hd(obs)
        return b_ineq

    def compute_safe_control(self, obs):
        if self.use_safe:
            A = self.compute_A(obs)
            assert(A.shape == (1, 2))

            b_ineq = self.compute_b(obs)

            #Make CVXOPT quadratic programming problem
            P = matrix(np.eye(2), tc='d')
            q = -1 * matrix(self.compute_nom_control(), tc='d')
            G = -1 * matrix(A.astype(np.double), tc='d')

            h = -1 * matrix(b_ineq.astype(np.double), tc='d')
            solvers.options['show_progress'] = False
            # get dictionary for solution
            sol = solvers.qp(P, q, G, h, verbose=False)

            optimized_u = sol['x']

        else:
            optimized_u = self.compute_nom_control()

        return optimized_u

    def compute_nom_control(self, Kn=np.array([-0.08, -0.2])):
        #! mock
        vd = Kn[0]*(np.atleast_2d(self.state["x"][:2]).T - self.goal)
        u_nom = Kn[1]*(np.atleast_2d(self.state["xdot"][:2]).T - vd)

        if np.linalg.norm(u_nom) > 0.01:
            u_nom = (u_nom/np.linalg.norm(u_nom)) * 0.01
        return u_nom.astype(np.double)
