from dynamics import QuadDynamics
from controller import *
import numpy as np
import matplotlib.pyplot as plt
from cvxopt import matrix
from cvxopt import solvers

a = 1
b = 2
safety_dist = 2
class SimpleDynamics():
    def __init__(self):
        ## State space
        r = np.array([np.array([1,-4])]).T # position
        rd = np.array([np.array([0, 0])]).T  # velocity
        self.state = {"r":r, "rd":rd}
        ## Params
        self.dt = 10e-3

        # self.u = zeros(2,1) # acceleration, control input

    def step(self, u):
        # rdd = self.u
        rd = self.state["rd"] + self.dt * u - self.state["rd"] * 0.02
        r = self.state["r"] + self.dt * self.state["rd"]

        self.state["rd"] = rd
        self.state["r"]  = r

class ECBF_control():
    def __init__(self, state, goal=np.array([[0], [10]])):
        self.state = state
        self.shape_dict = {} #TODO: a, b
        # self.gain_dict = {} #TODO: Kp, Kd
        Kp = 6
        Kd = 8
        self.K = np.array([Kp, Kd])
        self.goal=goal
        self.use_safe = True
        self.ecbf_safety_dist = 2.0

        # pass

    def compute_plot_z(self, obs):
        # obs = np.array([0,1]) #! mock

        plot_x = np.arange(-10, 10, 0.1)
        plot_y = np.arange(-10, 10, 0.1)
        xx, yy = np.meshgrid(plot_x, plot_y, sparse=True)
        z = h_func(xx - obs[0], yy - obs[1], a, b, safety_dist) > 0
        p = {"x":plot_x, "y":plot_y, "z":z}
        return p
        
        # plt.show()
    def plot_h(self, plot_x, plot_y, z):
        h = plt.contourf(plot_x, plot_y, z, [-1, 0, 1])
        # h = plt.contourf(plot_x, plot_y, z)
        plt.xlabel("X")
        plt.ylabel("Y")
        proxy = [plt.Rectangle((0, 0), 1, 1, fc=pc.get_facecolor()[0])
                 for pc in h.collections]
        # plt.legend(proxy, ["Unsafe: range(-1 to 0)","Safe: range(0 to 1)"])
        plt.legend()
        plt.pause(0.00000001)



    def compute_A(self,obs):
        # print("pos: ", self.pos)
        # print("col pos: ", self.col_pos)
        del_p = np.atleast_2d(self.state["x"][:2]).T - obs
        A1 = np.hstack((-del_p.T, del_p.T))
        # A2 = np.hstack((del_p, -del_p))
        # A = np.vstack((A1, A2))
        # A = -del_p
        print(A1.shape[1])
        A1 = A1.reshape(1, A1.shape[1])
        return A1

    def compute_b(self, obs, obs_v):
        gamma = 0.005
        alp1 = 1
        alp2 = 1
        
        h = self.compute_h(obs, obs_v, 1)
        del_pp = np.atleast_2d(self.state["x"][:2]).T - obs
        del_vp = np.atleast_2d(self.state["xdot"][:2]).T - obs_v
        b1 = gamma * h**3 * np.linalg.norm(del_pp) 
        # b1 = self.gamma * np.exp(h) * np.linalg.norm(del_pp) 
        b2 = (np.matmul(np.transpose(del_vp), del_pp))**2 / ((np.linalg.norm(del_pp))**2) 
        alp12 = (alp1 + alp2)
        b3 = alp12*np.matmul(np.transpose(del_vp), del_pp) / np.sqrt(2*alp12*(np.linalg.norm(del_pp) - self.ecbf_safety_dist))
        b4 = np.linalg.norm(del_vp)**2
        b11 = b1 - b2 + b3 + b4
        h = self.compute_h(obs, obs_v, 2)
        del_pm = -np.atleast_2d(self.state["x"][:2]).T + obs
        del_vm = -np.atleast_2d(self.state["xdot"][:2]).T + obs_v
        b1 = gamma * h**3 * np.linalg.norm(del_pm) 
        # b1 = self.gamma * np.exp(h) * np.linalg.norm(del_pm) 
        b2 = (np.matmul(np.transpose(del_vm), del_pm))**2 / ((np.linalg.norm(del_pm))**2) 
        alp12 = (alp1 + alp2)
        b3 = alp12*np.matmul(np.transpose(del_vm), del_pm) / np.sqrt(2*alp12*(np.linalg.norm(del_pm) - self.ecbf_safety_dist))
        b4 = np.linalg.norm(del_vm)**2
        b = np.vstack((b11, b1 - b2 + b3 + b4))
        return b11
    def compute_h(self, obs, obs_v, i):
        alp1 = 1
        alp2 = 1
        if i == 1:
            del_p = np.atleast_2d(self.state["x"][:2]).T - obs
            del_v = np.atleast_2d(self.state["xdot"][:2]).T - obs_v

            h1 = np.sqrt(2*(alp1+alp2)*(np.linalg.norm(del_p) - self.ecbf_safety_dist))
            h2 = np.matmul(np.transpose(del_p), del_v)/np.linalg.norm(del_p)
        elif i == 2:
            del_p = obs - np.atleast_2d(self.state["x"][:2]).T
            del_v = obs_v - np.atleast_2d(self.state["xdot"][:2]).T

            h1 = np.sqrt(2*(alp1+alp2)*(np.linalg.norm(del_p) - self.ecbf_safety_dist))
            h2 = np.matmul(np.transpose(del_p), del_v)/np.linalg.norm(del_p)

        # print("in  = "+str(2*(self.alp1+self.alp2)*(np.linalg.norm(del_p) - self.ecbf_safety_dist)))
        # print("h1  = "+str(h1))
        # print("h2 = "+str(h2))
        return h1+h2

    def compute_safe_control(self, obs, obs_v):
        if self.use_safe:
            A = self.compute_A(obs)
            assert(A.shape == (1,4))

            b_ineq = self.compute_b(obs, obs_v)

            #Make CVXOPT quadratic programming problem
            P = matrix(np.eye(2), tc='d')
            q = -1 * matrix(self.compute_nom_control(), tc='d')
            G = matrix(A.astype(np.double), tc='d')

            h = matrix(b_ineq.astype(np.double), tc='d')
            solvers.options['show_progress'] = False
            sol = solvers.qp(P,q,G, h, verbose=False) # get dictionary for solution

            optimized_u = sol['x']

        else:
            optimized_u = self.compute_nom_control()


        return optimized_u
        # u = np.linalg.pinv(A) @ b_ineq

        # return u

    def compute_nom_control(self, Kn=np.array([-0.08, -0.2])):
        #! mock
        vd = Kn[0]*(np.atleast_2d(self.state["x"][:2]).T - self.goal)
        u_nom = Kn[1]*(np.atleast_2d(self.state["xdot"][:2]).T - vd)

        if np.linalg.norm(u_nom) > 0.01:
            u_nom = (u_nom/np.linalg.norm(u_nom))* 0.01
        return u_nom.astype(np.double)

    # def compute_control(self, obs):

@np.vectorize
def h_func(r1, r2, a, b, safety_dist):
    hr = np.power(r1,4)/np.power(a, 4) + \
        np.power(r2, 4)/np.power(b, 4) - safety_dist
    return hr


def robot_step(state, state_hist, dyn, ecbf, new_obs, obs_v):
    u_hat_acc = ecbf.compute_safe_control(obs=new_obs, obs_v=obs_v)
    u_hat_acc = np.ndarray.flatten(np.array(np.vstack((u_hat_acc,np.zeros((1,1))))))  # acceleration
    assert(u_hat_acc.shape == (3,))
    u_motor = go_to_acceleration(state, u_hat_acc, dyn.param_dict) # desired motor rate ^2

    state = dyn.step_dynamics(state, u_motor)
    ecbf.state = state
    state_hist.append(state["x"])
    return u_hat_acc

def plot_step(ecbf, new_obs, u_hat_acc, state_hist):
    state_hist_plot = np.array(state_hist)
    nom_cont = ecbf.compute_nom_control()
    multiplier_const = 100
    plt.plot([state_hist_plot[-1, 0], state_hist_plot[-1, 0] + multiplier_const *
                u_hat_acc[0]],
                [state_hist_plot[-1, 1], state_hist_plot[-1, 1] + multiplier_const * u_hat_acc[1]], label="Safe")
    plt.plot([state_hist_plot[-1, 0], state_hist_plot[-1, 0] + multiplier_const *
                nom_cont[0]],
                [state_hist_plot[-1, 1], state_hist_plot[-1, 1] + multiplier_const * nom_cont[1]],label="Nominal")

    plt.plot(state_hist_plot[:, 0], state_hist_plot[:, 1],'k')
    plt.plot(ecbf.goal[0], ecbf.goal[1], '*r')
    plt.plot(state_hist_plot[-1, 0], state_hist_plot[-1, 1], '8k') # current

    
    

def main():
    # pass
    # dyn = SimpleDynamics()
    ### Robot 1
    state1 = {"x": np.array([3, -5, 10]),
                "xdot": np.zeros(3,),
                "theta": np.radians(np.array([0, 0, 0])),  # ! hardcoded
                "thetadot": np.radians(np.array([0, 0, 0]))  # ! hardcoded
                }
    dyn = QuadDynamics()
    goal1 = np.array([[-6], [4]])
    ecbf1 = ECBF_control(state1, goal1)


    state1_hist = []
    state1_hist.append(state1["x"])

    new_obs1 = np.array([[1], [1]])

    ### Robot 2
    state2 = {"x": np.array([-5, 3, 10]),
                "xdot": np.zeros(3,),
                "theta": np.radians(np.array([0, 0, 0])),  # ! hardcoded
                "thetadot": np.radians(np.array([0, 0, 0]))  # ! hardcoded
                }
    dyn = QuadDynamics()
    goal2 = np.array([[4], [-6]])
    ecbf2 = ECBF_control(state2, goal2)

    state2_hist = []
    state2_hist.append(state2["x"])

    new_obs2 = np.array([[1], [1]])
    for tt in range(20000):
        new_obs1 = state2["x"][:2].reshape(2,1)
        new_obs2 = state1["x"][:2].reshape(2,1)
        obs_v1 = state2["xdot"][:2].reshape(2,1)
        obs_v2 = state1["xdot"][:2].reshape(2,1)
        # print(new_obs1.shape)
        u_hat_acc1 = robot_step(state1, state1_hist, dyn, ecbf1, new_obs1, obs_v1)
        u_hat_acc2 = robot_step(state2, state2_hist, dyn, ecbf2, new_obs2, obs_v2)

        if(tt % 10 == 0):
            print(tt)
            plt.cla()

            plot_step(ecbf1, new_obs1, u_hat_acc1, state1_hist)
            plot_step(ecbf2, new_obs2, u_hat_acc2, state2_hist)

            p1 = ecbf1.compute_plot_z(new_obs1)
            p2 = ecbf1.compute_plot_z(new_obs2)
            x = (p1["x"] + p2["x"]) / 2
            y = (p1["y"] + p2["y"]) / 2
            z = (p1["z"] + p2["z"]) / 2
            ecbf1.plot_h(x, y, z)









if __name__=="__main__":
    main()
