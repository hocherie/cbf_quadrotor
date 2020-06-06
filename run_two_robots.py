"""run_two_robots.py
Exponential Control Barrier Function (ECBF) with Quadrotor Dynamics

`python run_two_robots.py` to see two robots navigating towards each other, with ECBF safe control to maintain safety
"""
from dynamics import QuadDynamics
from ecbf_control import ECBFControl
from controller import *
from sim_utils import *
import numpy as np
import matplotlib.pyplot as plt 


# TODO: Move to another ecbf_sim specific file?
def robot_ecbf_step(state, state_hist, dyn, ecbf, new_obs):
    u_hat_acc = ecbf.compute_safe_control(obs=new_obs)
    u_hat_acc = np.ndarray.flatten(np.array(np.vstack((u_hat_acc,np.zeros((1,1))))))  # acceleration
    assert(u_hat_acc.shape == (3,))
    u_motor = go_to_acceleration(state, u_hat_acc, dyn.param_dict) # desired motor rate ^2

    state = dyn.step_dynamics(state, u_motor)
    ecbf.state = state
    state_hist.append(state["x"])
    return u_hat_acc

# TODO: Move to another ecbf_sim specific file?
def plot_ecbf_step(ecbf, new_obs, u_hat_acc, state_hist):
    state_hist_plot = np.array(state_hist)
    nom_cont = ecbf.compute_nom_control()
    multiplier_const = 100
    plt.plot([state_hist_plot[-1, 0], state_hist_plot[-1, 0] + multiplier_const *
                u_hat_acc[0]],
             [state_hist_plot[-1, 1], state_hist_plot[-1, 1] + multiplier_const * u_hat_acc[1]], color='r', linewidth=2, label="Safe Control")
    plt.plot([state_hist_plot[-1, 0], state_hist_plot[-1, 0] + multiplier_const *
                nom_cont[0]],
                [state_hist_plot[-1, 1], state_hist_plot[-1, 1] + multiplier_const * nom_cont[1]],color='b',linewidth=2,label="Nominal Control")

    plt.plot(state_hist_plot[:, 0], state_hist_plot[:, 1],'k')
    plt.plot(ecbf.goal[0], ecbf.goal[1], '*r')
    plt.plot(state_hist_plot[-1, 0], state_hist_plot[-1, 1], '8k') # current

    
    

def main():
    ### Initialize Robot 1 (state, dynamics, goal, ecbf control)
    state1 = {"x": np.array([3, -5, 10]),
                "xdot": np.zeros(3,),
                "theta": np.radians(np.array([0, 0, 0])),  
                "thetadot": np.radians(np.array([0, 0, 0]))  
                }
    dyn = QuadDynamics()
    goal1 = np.array([[-6], [4]])
    ecbf1 = ECBFControl(state1, goal1)

    ## Save robot position history for plotting and analysis
    state1_hist = []
    state1_hist.append(state1["x"])

    new_obs1 = np.array([[1], [1]]) # center of obstacle

    ### Initialize Robot 2 (state, dynamics, goal, ecbf control)
    state2 = {"x": np.array([-5, 3, 10]),
                "xdot": np.zeros(3,),
                "theta": np.radians(np.array([0, 0, 0])),  # ! hardcoded
                "thetadot": np.radians(np.array([0, 0, 0]))  # ! hardcoded
                }
    dyn = QuadDynamics()
    goal2 = np.array([[4], [-6]])
    ecbf2 = ECBFControl(state2, goal2)
    ## Save robot position history for plotting and analysis
    state2_hist = []
    state2_hist.append(state2["x"])

    new_obs2 = np.array([[1], [1]])
    for tt in range(20000): # update ECBF obstacle with other robot position
        new_obs1 = state2["x"][:2].reshape(2,1)
        new_obs2 = state1["x"][:2].reshape(2,1)
        # print(new_obs1.shape)
        u_hat_acc1 = robot_ecbf_step(state1, state1_hist, dyn, ecbf1, new_obs1)
        u_hat_acc2 = robot_ecbf_step(state2, state2_hist, dyn, ecbf2, new_obs2)

        if(tt % 25 == 0):
            print("Timestep", tt)
            plt.cla()

            plot_ecbf_step(ecbf1, new_obs1, u_hat_acc1, state1_hist)
            plot_ecbf_step(ecbf2, new_obs2, u_hat_acc2, state2_hist)

            p1 = ecbf1.compute_plot_z(new_obs1)
            p2 = ecbf1.compute_plot_z(new_obs2)
            x = (p1["x"] + p2["x"]) / 2
            y = (p1["y"] + p2["y"]) / 2
            z = (p1["z"] + p2["z"]) / 2
            ecbf1.plot_h(x, y, z)









if __name__=="__main__":
    main()
