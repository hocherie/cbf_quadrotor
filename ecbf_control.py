from dynamics import QuadDynamics
from controller import *
import numpy as np
import matplotlib.pyplot as plt
from cvxopt import matrix
from cvxopt import solvers
from matplotlib.patches import Ellipse
import time
import warnings

# warnings.filterwarnings("ignore")

a = 1
b = 1
safety_dist = 1
robot_radius = 0.5
is_crash = False # Sets title as Crashed when crashed once

class ECBF_control():
    def __init__(self, state, goal=np.array([[0], [10]])):
        self.state = state
        self.shape_dict = {} #TODO: a, b
        Kp = 6
        Kd = 8
        self.K = np.array([Kp, Kd])
        self.goal=goal
        self.use_safe = True

    def compute_plot_z(self, obs):
        plot_x = np.arange(-7.5, 7.5, 0.4)
        plot_y = np.arange(-7.5, 7.5, 0.4)
        xx, yy = np.meshgrid(plot_x, plot_y, sparse=True)
        z = np.zeros(xx.shape)
        for i in range(obs.shape[1]):
            ztemp = h_func(xx - obs[0][i], yy - obs[1][i], a, b, safety_dist) > 0
            z = z + ztemp
        z = z / (obs.shape[1]-1)
        p = {"x":plot_x, "y":plot_y, "z":z}
        return p
        
        
    def plot_h(self, plot_x, plot_y, z):
        h = plt.contourf(plot_x, plot_y, z, [-1, 0, 1],colors=['#808080', '#A0A0A0', '#C0C0C0'])
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.pause(0.00000001)



    def compute_h(self, obs=np.array([[0], [0]]).T):
        h = np.zeros((obs.shape[1], 1))
        for i in range(obs.shape[1]):
            rel_r = np.atleast_2d(self.state["x"][:2]).T - obs[:, i].reshape(2,1)
            # TODO: a, safety_dist, obs, b
            hr = h_func(rel_r[0], rel_r[1], a, b, safety_dist)
            h[i] = hr
        return h

    def compute_hd(self, obs, obs_v):
        hd = np.zeros((obs.shape[1], 1))
        for i in range(obs.shape[1]):
            rel_r = np.atleast_2d(self.state["x"][:2]).T - obs[:, i].reshape(2,1)
            rd = np.atleast_2d(self.state["xdot"][:2]).T - obs_v[:, i].reshape(2,1)
            term1 = (4 * np.power(rel_r[0],3) * rd[0])/(np.power(a,4))
            term2 = (4 * np.power(rel_r[1],3) * rd[1])/(np.power(b,4))
            hd[i] = term1 + term2
        return hd

    def compute_A(self, obs):
        A = np.empty((0,2))
        for i in range(obs.shape[1]):
            rel_r = np.atleast_2d(self.state["x"][:2]).T - obs[:, i].reshape(2,1)
            A0 = (4 * np.power(rel_r[0], 3))/(np.power(a, 4))
            A1 = (4 * np.power(rel_r[1], 3))/(np.power(b, 4))
            Atemp = np.array([np.hstack((A0, A1))])
            
            A = np.array(np.vstack((A, Atemp)))
        
        A = -1 * matrix(A.astype(np.double), tc='d')
        return A

    def compute_h_hd(self, obs, obs_v):
        h = self.compute_h(obs)
        hd = self.compute_hd(obs, obs_v)
        return np.vstack((h, hd)).astype(np.double)

    def compute_b(self, obs, obs_v):
        """extra + K * [h hd]"""
        rel_r = np.atleast_2d(self.state["x"][:2]).T - obs
        rd = np.atleast_2d(self.state["xdot"][:2]).T - obs_v

        extra = -( (12 * np.square(rel_r[0]) * np.square(rd[0]))/np.power(a, 4) + (12 * np.square(rel_r[1]) * np.square(rd[1]))/np.power(b, 4) )
        extra = extra.reshape(obs.shape[1], 1)

        b_ineq =  extra - ( self.K[0] * self.compute_h(obs) + self.K[1] * self.compute_hd(obs, obs_v) )
        b_ineq = -1 * matrix(b_ineq.astype(np.double), tc='d')
        return b_ineq



    def compute_safe_control(self,obs, obs_v, id):
        # control in R^2
        if self.use_safe:
            try:
                A = self.compute_A(obs) # For Exercise 1
                b = self.compute_b(obs, obs_v) # For Exercise 1
                u_des = self.compute_nom_control() # For Exercise 1

                # optimized_u = u_des #! REPLACE!! Exercise 1: Write Minimum Interventional Control

                # Solution to Exercise 1
                P = np.eye(2)
                q = -1 * u_des
                G = A 
                h = b 
                Sol = solve_qp(P,q,G,h)
                optimized_u = Sol['x']

            except:
                print("Robot "+str(id)+": NO SOLUTION!!!")
                optimized_u = [[0], [0]]
            

        else:
            optimized_u = self.compute_nom_control()

        
        return optimized_u

    def compute_nom_control(self, Kn=np.array([-0.08, -0.2])):
        vd = Kn[0]*(np.atleast_2d(self.state["x"][:2]).T - self.goal)
        u_nom = Kn[1]*(np.atleast_2d(self.state["xdot"][:2]).T - vd)

        if np.linalg.norm(u_nom) > 0.1:
            u_nom = (u_nom/np.linalg.norm(u_nom))* 0.1
        return matrix(u_nom, tc='d')




class Robot_Sim():
    def __init__(self, x_init, goal_init, robot_id):
        self.id = robot_id
        self.state = {"x": x_init,
                "xdot": np.zeros(3,),
                "theta": np.radians(np.array([0, 0, 0])),  # ! hardcoded
                "thetadot": np.radians(np.array([0, 0, 0]))  # ! hardcoded
                }
        self.dyn = QuadDynamics()
        self.goal = goal_init
        self.ecbf = ECBF_control(self.state, self.goal)


        self.state_hist = []
        self.state_hist.append(self.state["x"])

        self.new_obs = np.array([[1], [1]])
    def robot_step(self, new_obs, obs_v):
        u_hat_acc = self.ecbf.compute_safe_control(obs=new_obs, obs_v=obs_v, id=self.id)
        u_hat_acc = np.ndarray.flatten(np.array(np.vstack((u_hat_acc,np.zeros((1,1))))))  # acceleration
        assert(u_hat_acc.shape == (3,))
        u_motor = go_to_acceleration(self.state, u_hat_acc, self.dyn.param_dict) # desired motor rate ^2

        self.state = self.dyn.step_dynamics(self.state, u_motor)
        self.ecbf.state = self.state
        self.state_hist.append(self.state["x"])
        return u_hat_acc

    def update_obstacles(self, robots, obs, noisy = False):
        obst = []
        obs_v = []
        for robot in robots:
            if robot.id == self.id:
                continue
            if np.linalg.norm( robot.state["x"][:2] - self.state["x"][:2]) < robot_radius:
                print("CRASH!!!!!!!!!!!!!!!!!!!!")
                global is_crash 
                is_crash = True
            
            obst_temp = robot.state["x"][:2] 
            if noisy:
                obst_temp = obst_temp + (np.random.random(2)*2-1) # + np.array([[0.5], [0.5]]).T 
            obst.append(obst_temp.reshape(2,1))
            obs_v.append(robot.state["xdot"][:2].reshape(2,1))
        if not len(obs):
            return {"obs":obst, "obs_v":obs_v}
        if obs.ndim == 1:
            obst.append(obs.reshape(2,1))
            obs_v.append(np.array([[0], [0]]))
            return {"obs":obst, "obs_v":obs_v}
        for i in range(obs.shape[0]):
            obst.append(obs[i].reshape(2,1))
            obs_v.append(np.array([[0], [0]]))
        
        obstacles = {"obs":obst, "obs_v":obs_v}
        return obstacles

        



@np.vectorize
def h_func(r1, r2, a, b, safety_dist):
    hr = np.power(r1,4)/np.power(a, 4) + \
        np.power(r2, 4)/np.power(b, 4) - safety_dist
    return hr


def plot_step(id, ecbf, new_obs, u_hat_acc, state_hist, plot_handle):
    state_hist_plot = np.array(state_hist)
    nom_cont = ecbf.compute_nom_control()
    multiplier_const = 15
    plot_handle.plot([state_hist_plot[-1, 0], state_hist_plot[-1, 0] + multiplier_const *
                u_hat_acc[0]],
                [state_hist_plot[-1, 1], state_hist_plot[-1, 1] + multiplier_const * u_hat_acc[1]], label="Safe", color='b')
    plot_handle.plot([state_hist_plot[-1, 0], state_hist_plot[-1, 0] + multiplier_const *
                nom_cont[0]],
                [state_hist_plot[-1, 1], state_hist_plot[-1, 1] + multiplier_const * nom_cont[1]],label="Nominal",color='orange')

    plot_handle.plot(state_hist_plot[:, 0], state_hist_plot[:, 1])
    plot_handle.plot(ecbf.goal[0], ecbf.goal[1], '*r')
    # plot_handle.plot(state_hist_plot[-1, 0], state_hist_plot[-1, 1], '8k') # current
    plot_handle.text(state_hist_plot[-1,0]+0.2, state_hist_plot[-1,1]+0.2, str(id))
    if is_crash:
        plot_handle.set_title("CRASHED!")
    for i in range(new_obs.shape[1]):
        plot_handle.plot(new_obs[0, i], new_obs[1, i], '8k') # obs
    

    ell = Ellipse((state_hist_plot[-1, 0], state_hist_plot[-1, 1]), a*safety_dist+0.5, b*safety_dist+0.5, 0)
    ell.set_alpha(0.3)
    ell.set_facecolor(np.array([0, 1, 0]))
    
    plot_handle.add_artist(ell)

    ell = Ellipse((state_hist_plot[-1, 0], state_hist_plot[-1, 1]), robot_radius+0.5, robot_radius+0.5, 0)
    ell.set_alpha(0.8)
    ell.set_facecolor(np.array([1, 0, 0]))
    
    plot_handle.add_artist(ell)

    plot_handle.set_xlim([-10, 10])
    plot_handle.set_ylim([-10, 10])

def solve_qp(P,q,G,h):
    # Custom wrapper cvxopt.solvers.qp
    # Takes in numpy array Converts to matrix double
    P = matrix(P,tc='d')
    q = matrix(q,tc='d')
    G = matrix(G,tc='d')
    h = matrix(h,tc='d')
    Sol = solvers.qp(P,q,G,h)
    return Sol